"""Contract tests every QueueBackend must pass.

The memory backend always runs. The Redis backend runs when REDIS_URL is set
(it is skipped otherwise), so the same behaviour is pinned for both.
"""

import os
import threading
import uuid
from types import SimpleNamespace

import pytest

from fastmlapi.queue.memory_backend import MemoryBackend


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "redis",
            marks=pytest.mark.skipif(
                not os.environ.get("REDIS_URL"), reason="REDIS_URL not set"
            ),
        ),
    ]
)
def env(request):
    clock = FakeClock()
    if request.param == "memory":
        yield SimpleNamespace(
            backend=MemoryBackend(result_ttl=100, clock=clock), clock=clock
        )
        return
    from fastmlapi.queue.redis_backend import RedisBackend

    prefix = f"fastmlapi-test:{uuid.uuid4().hex}:"
    backend = RedisBackend(
        os.environ["REDIS_URL"], prefix=prefix, result_ttl=100, clock=clock
    )
    yield SimpleNamespace(backend=backend, clock=clock)
    for key in backend._r.scan_iter(match=prefix + "*"):
        backend._r.delete(key)


def test_enqueue_then_get_job_is_queued(env):
    job_id = env.backend.enqueue({"x": 1})
    job = env.backend.get_job(job_id)
    assert job.status == "queued"
    assert job.payload == {"x": 1}
    assert job.attempts == 0
    assert env.backend.pending_count() == 1


def test_get_unknown_job_returns_none(env):
    assert env.backend.get_job("nope") is None


def test_claim_returns_none_when_empty(env):
    assert env.backend.claim("w1") is None


def test_claim_is_fifo_and_marks_running(env):
    b = env.backend
    first = b.enqueue(1)
    second = b.enqueue(2)
    job = b.claim("w1")
    assert job.id == first
    assert job.status == "running"
    assert job.worker == "w1"
    assert job.attempts == 1
    assert job.payload == 1
    assert b.claim("w2").id == second
    assert b.pending_count() == 0


def test_complete_stores_result(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    assert b.complete("w1", job_id, {"label": "ok"}) is True
    job = b.get_job(job_id)
    assert job.status == "succeeded"
    assert job.result == {"label": "ok"}
    assert job.finished_at is not None


def test_complete_by_non_owner_is_rejected(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    assert b.complete("w2", job_id, "nope") is False
    assert b.get_job(job_id).status == "running"


def test_fail_records_error(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    assert b.fail("w1", job_id, "boom", "ValueError") is True
    job = b.get_job(job_id)
    assert job.status == "failed"
    assert job.error == "boom"
    assert job.error_type == "ValueError"


def test_complete_with_unserializable_result_raises_and_job_stays_running(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    with pytest.raises(TypeError):
        b.complete("w1", job_id, object())
    assert b.get_job(job_id).status == "running"
    # The worker can still report the failure afterwards.
    assert b.fail("w1", job_id, "not serializable", "ResultNotSerializable") is True
    assert b.get_job(job_id).status == "failed"


def test_heartbeat_refreshes_timestamp_and_rejects_non_owner(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    env.clock.advance(10)
    assert b.heartbeat("w1", job_id) is True
    assert b.get_job(job_id).heartbeat == env.clock.now
    assert b.heartbeat("w2", job_id) is False


def test_requeue_stale_returns_job_to_pending(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    env.clock.advance(31)
    assert b.requeue_stale(30, 3) == 1
    job = b.get_job(job_id)
    assert job.status == "queued"
    assert job.attempts == 1
    assert job.worker is None
    assert b.pending_count() == 1
    assert b.claim("w2").attempts == 2


def test_requeue_stale_skips_job_with_fresh_heartbeat(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    env.clock.advance(20)
    b.heartbeat("w1", job_id)
    env.clock.advance(20)
    assert b.requeue_stale(30, 3) == 0
    assert b.get_job(job_id).status == "running"


def test_requeue_stale_fails_job_after_max_attempts(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    env.clock.advance(31)
    assert b.requeue_stale(30, 2) == 1
    b.claim("w2")
    env.clock.advance(31)
    assert b.requeue_stale(30, 2) == 1
    job = b.get_job(job_id)
    assert job.status == "failed"
    assert job.error_type == "WorkerLost"
    assert b.pending_count() == 0


def test_zombie_worker_cannot_overwrite_new_owner(env):
    b = env.backend
    job_id = b.enqueue({})
    b.claim("w1")
    env.clock.advance(31)
    b.requeue_stale(30, 3)
    b.claim("w2")
    assert b.complete("w1", job_id, "late") is False
    assert b.fail("w1", job_id, "late", "X") is False
    assert b.complete("w2", job_id, "fresh") is True
    assert b.get_job(job_id).result == "fresh"


def test_concurrent_claims_never_hand_out_the_same_job(env):
    b = env.backend
    ids = {b.enqueue(i) for i in range(50)}
    claimed = []
    lock = threading.Lock()

    def drain(worker_id):
        while True:
            job = b.claim(worker_id)
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=drain, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 50
    assert set(claimed) == ids


def test_ping(env):
    assert env.backend.ping() is True

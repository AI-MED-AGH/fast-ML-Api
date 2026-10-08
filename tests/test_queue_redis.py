"""RedisBackend-specific behaviour. The shared contract is in test_queue_backends.py."""

import os
import sys
import uuid

import pytest

from fastmlapi.queue.backend import QueueUnavailable

needs_redis = pytest.mark.skipif(
    not os.environ.get("REDIS_URL"), reason="REDIS_URL not set"
)


@needs_redis
def test_finished_jobs_get_a_ttl_and_queued_jobs_do_not():
    from fastmlapi.queue.redis_backend import RedisBackend

    prefix = f"fastmlapi-test:{uuid.uuid4().hex}:"
    b = RedisBackend(os.environ["REDIS_URL"], prefix=prefix, result_ttl=100)
    try:
        job_id = b.enqueue({})
        assert b._r.ttl(f"{prefix}job:{job_id}") == -1  # no expiry while queued
        b.claim("w1")
        assert b._r.ttl(f"{prefix}job:{job_id}") == -1  # nor while running
        b.complete("w1", job_id, "done")
        assert 0 < b._r.ttl(f"{prefix}job:{job_id}") <= 100
    finally:
        for key in b._r.scan_iter(match=prefix + "*"):
            b._r.delete(key)


def test_unreachable_redis_raises_queue_unavailable():
    pytest.importorskip("redis")
    from fastmlapi.queue.redis_backend import RedisBackend

    b = RedisBackend("redis://127.0.0.1:1/0")  # nothing listens on port 1
    with pytest.raises(QueueUnavailable):
        b.enqueue({})
    with pytest.raises(QueueUnavailable):
        b.get_job("x")
    assert b.ping() is False


def test_missing_redis_package_raises_helpful_import_error(monkeypatch):
    from fastmlapi.queue.redis_backend import RedisBackend

    monkeypatch.setitem(sys.modules, "redis", None)  # makes `import redis` fail
    with pytest.raises(ImportError, match=r"fastmlapi\[queue\]"):
        RedisBackend("redis://localhost:6379/0")


def _fresh_client(url):
    import redis

    return redis.Redis.from_url(url, decode_responses=True)


@needs_redis
def test_second_sweep_with_stale_snapshot_does_not_requeue_twice():
    """Two sweepers read the same processing list; only the first may requeue."""
    from fastmlapi.queue.redis_backend import RedisBackend

    url = os.environ["REDIS_URL"]
    prefix = f"fastmlapi-test:{uuid.uuid4().hex}:"
    now = [1000.0]
    b = RedisBackend(url, prefix=prefix, result_ttl=100, clock=lambda: now[0])
    proc_key = f"{prefix}processing:w1"
    try:
        b.enqueue({})
        b.claim("w1")
        now[0] += 31
        snapshot = b._r.lrange(proc_key, 0, -1)  # what the second sweeper saw
        # Replay the stale snapshot for both sweeps, as a concurrent sweeper would.
        b._r.lrange = lambda *a, **k: snapshot
        b._r.scan_iter = lambda *a, **k: [proc_key]
        assert b.requeue_stale(30, 3) == 1
        assert b.requeue_stale(30, 3) == 0
        assert b.pending_count() == 1
    finally:
        r = _fresh_client(url)
        for key in r.scan_iter(match=prefix + "*"):
            r.delete(key)


@needs_redis
def test_job_completed_after_snapshot_is_not_resurrected():
    """A worker completes the job between the sweeper's snapshot and its requeue."""
    from fastmlapi.queue.redis_backend import RedisBackend

    url = os.environ["REDIS_URL"]
    prefix = f"fastmlapi-test:{uuid.uuid4().hex}:"
    now = [1000.0]
    b = RedisBackend(url, prefix=prefix, result_ttl=100, clock=lambda: now[0])
    proc_key = f"{prefix}processing:w1"
    try:
        job_id = b.enqueue({})
        b.claim("w1")
        now[0] += 31
        snapshot = b._r.lrange(proc_key, 0, -1)  # sweeper snapshots: job is in flight
        assert b.complete("w1", job_id, "done") is True
        b._r.lrange = lambda *a, **k: snapshot
        b._r.scan_iter = lambda *a, **k: [proc_key]
        assert b.requeue_stale(30, 3) == 0
        job = b.get_job(job_id)
        assert job.status == "succeeded"
        assert job.result == "done"
        assert b.pending_count() == 0
    finally:
        r = _fresh_client(url)
        for key in r.scan_iter(match=prefix + "*"):
            r.delete(key)

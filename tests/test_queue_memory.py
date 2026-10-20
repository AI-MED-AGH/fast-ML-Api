"""MemoryBackend-specific behaviour."""

from fastmlapi.queue.memory_backend import MemoryBackend
from tests.test_queue_backends import FakeClock


def test_finished_jobs_expire_after_result_ttl():
    clock = FakeClock()
    b = MemoryBackend(result_ttl=100, clock=clock)
    job_id = b.enqueue({})
    b.claim("w1")
    b.complete("w1", job_id, "done")
    clock.advance(99)
    assert b.get_job(job_id) is not None
    clock.advance(2)
    assert b.get_job(job_id) is None


def test_failed_jobs_expire_after_result_ttl():
    clock = FakeClock()
    b = MemoryBackend(result_ttl=100, clock=clock)
    job_id = b.enqueue({})
    b.claim("w1")
    b.fail("w1", job_id, "boom", "ValueError")
    clock.advance(101)
    assert b.get_job(job_id) is None


def test_queued_and_running_jobs_never_expire():
    clock = FakeClock()
    b = MemoryBackend(result_ttl=100, clock=clock)
    queued = b.enqueue({})
    running = b.enqueue({})
    b.claim("w1")  # claims `queued` first (FIFO)
    clock.advance(10_000)
    assert b.get_job(queued).status == "running"
    assert b.get_job(running).status == "queued"

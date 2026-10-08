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

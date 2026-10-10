"""Proves the Redis backend works for a user that may NOT run scripts or SCAN.

Opt in with REDIS_ADMIN_URL=redis://:<admin password>@host:port/0 (a throwaway Redis >= 7; it is flushed).
"""
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

ADMIN = os.environ.get("REDIS_ADMIN_URL")
pytestmark = pytest.mark.skipif(not ADMIN, reason="set REDIS_ADMIN_URL to run against a real Redis")


@pytest.fixture
def restricted():
    import redis

    from fastmlapi.queue.redis_backend import REDIS_COMMANDS_USED, RedisBackend

    admin = redis.Redis.from_url(ADMIN, decode_responses=True)
    model = f"acl{uuid.uuid4().hex[:8]}"
    prefix = f"fastmlapi:{model}:"
    user = f"m_{model}"
    rules = ["reset", "on", ">pw", f"~{prefix}*", "resetchannels", "-@all"]
    rules += [f"+{c.lower()}" for c in REDIS_COMMANDS_USED]
    rules += ["+hello", "+auth", "+client|setinfo", "+client|setname", "+client|id"]
    admin.execute_command("ACL", "SETUSER", user, *rules)
    parts = urlsplit(ADMIN)
    url = urlunsplit((parts.scheme, f"{user}:pw@{parts.hostname}:{parts.port}", parts.path, "", ""))
    other = f"fastmlapi:someone-else:{uuid.uuid4().hex}"
    admin.set(other, "private")
    now = [1000.0]
    backend = RedisBackend(url, prefix=prefix, result_ttl=100, clock=lambda: now[0])
    yield backend, now, redis.Redis.from_url(url, decode_responses=True), other
    admin.execute_command("ACL", "DELUSER", user)
    for key in list(admin.scan_iter(match=prefix + "*")) + [other]:
        admin.delete(key)
    admin.close()


def test_the_whole_job_lifecycle_works_without_scripts_or_scanning(restricted):
    b, now, _, _ = restricted
    assert b.ping() is True
    ok, bad, lost = b.enqueue({"n": 1}), b.enqueue({"n": 2}), b.enqueue({"n": 3})
    assert b.pending_count() == 3
    assert b.claim("w1").id == ok and b.claim("w1").id == bad and b.claim("w2").id == lost
    assert b.heartbeat("w1", ok) is True and b.heartbeat("w2", ok) is False
    assert b.complete("w1", ok, {"r": 1}) is True
    assert b.fail("w1", bad, "boom", "ValueError") is True
    now[0] += 31
    assert b.requeue_stale(30, 3) == 1                      # w2 never finished `lost`
    assert b.get_job(lost).status == "queued" and b.get_job(lost).payload == {"n": 3}
    assert b.get_job(ok).payload is None and b.get_job(bad).payload is None
    b.worker_alive("w1", model_loaded=True, ttl=30)
    assert [(w.worker_id, w.model_loaded) for w in b.list_workers()] == [("w1", True)]
    b.worker_gone("w1")
    assert b.list_workers() == []


def test_the_restricted_user_really_cannot_script_scan_or_read_other_keys(restricted):
    import redis

    _, _, raw, other = restricted
    for call in (lambda: raw.eval("return 1", 0), lambda: list(raw.scan_iter()), lambda: raw.keys("*"),
                 lambda: raw.get(other), lambda: raw.flushall()):
        with pytest.raises(redis.exceptions.ResponseError):
            call()

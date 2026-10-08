"""Redis queue backend.

Keys (all under `prefix`):
  pending                list of job ids, enqueue LPUSH / claim RPOP (FIFO)
  processing:{worker}    list of job ids claimed by that worker
  job:{id}               hash with the job record

Every state transition that must be atomic is a Lua script, so two workers can
never claim the same job and a stalled worker cannot overwrite a job that was
requeued to someone else (completion is fenced by worker id).
"""

from __future__ import annotations

import functools
import json
import uuid
from typing import Any, Callable, List, Mapping, Optional, cast

from .backend import QUEUED, Job, QueueBackend, QueueUnavailable

# KEYS: pending, processing:{worker}   ARGV: now, job key prefix, worker id
_CLAIM = """
local id = redis.call('RPOP', KEYS[1])
if not id then return nil end
redis.call('LPUSH', KEYS[2], id)
local k = ARGV[2] .. id
redis.call('HSET', k, 'status', 'running', 'started_at', ARGV[1],
           'heartbeat', ARGV[1], 'worker', ARGV[3])
redis.call('HINCRBY', k, 'attempts', 1)
return id
"""

# KEYS: job   ARGV: worker id, now
_HEARTBEAT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'worker') ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[1], 'heartbeat', ARGV[2])
return 1
"""

# KEYS: job, processing:{worker}   ARGV: worker id, job id, now, ttl, result json
_COMPLETE = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'worker') ~= ARGV[1] then return 0 end
redis.call('LREM', KEYS[2], 0, ARGV[2])
redis.call('HSET', KEYS[1], 'status', 'succeeded', 'result', ARGV[5],
           'finished_at', ARGV[3])
redis.call('HDEL', KEYS[1], 'error', 'error_type')
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
"""

# KEYS: job, processing:{worker}   ARGV: worker id, job id, now, ttl, error, error type
_FAIL = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'worker') ~= ARGV[1] then return 0 end
redis.call('LREM', KEYS[2], 0, ARGV[2])
redis.call('HSET', KEYS[1], 'status', 'failed', 'error', ARGV[5],
           'error_type', ARGV[6], 'finished_at', ARGV[3])
redis.call('HDEL', KEYS[1], 'result')
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
"""

# KEYS: job, processing:{worker}, pending
# ARGV: now, visibility timeout, max attempts, job id, ttl
# returns 0 untouched, 1 requeued, 2 failed (worker lost)
_REQUEUE_IF_STALE = """
local hb = tonumber(redis.call('HGET', KEYS[1], 'heartbeat'))
if not hb then
  redis.call('LREM', KEYS[2], 0, ARGV[4])
  return 0
end
if tonumber(ARGV[1]) - hb <= tonumber(ARGV[2]) then return 0 end
redis.call('LREM', KEYS[2], 0, ARGV[4])
local attempts = tonumber(redis.call('HGET', KEYS[1], 'attempts') or '0')
if attempts >= tonumber(ARGV[3]) then
  redis.call('HSET', KEYS[1], 'status', 'failed',
             'error', 'Worker lost after ' .. attempts .. ' attempt(s)',
             'error_type', 'WorkerLost', 'finished_at', ARGV[1])
  redis.call('EXPIRE', KEYS[1], ARGV[5])
  return 2
end
redis.call('HSET', KEYS[1], 'status', 'queued')
redis.call('HDEL', KEYS[1], 'worker')
redis.call('RPUSH', KEYS[3], ARGV[4])
return 1
"""


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn Redis connection errors into QueueUnavailable."""

    @functools.wraps(fn)
    def wrapper(self: "RedisBackend", *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except self._connection_errors as exc:
            raise QueueUnavailable(f"Redis unavailable: {exc}") from exc

    return wrapper


def _to_job(job_id: str, h: Mapping[Any, Any]) -> Job:
    def num(key: str) -> Optional[float]:
        value = h.get(key)
        return float(value) if value not in (None, "") else None

    return Job(
        id=job_id,
        status=h["status"],
        payload=json.loads(h["payload"]) if "payload" in h else None,
        result=json.loads(h["result"]) if "result" in h else None,
        error=h.get("error"),
        error_type=h.get("error_type"),
        attempts=int(h.get("attempts", 0)),
        worker=h.get("worker"),
        created_at=float(h.get("created_at", 0.0)),
        started_at=num("started_at"),
        finished_at=num("finished_at"),
        heartbeat=num("heartbeat"),
    )


class RedisBackend(QueueBackend):
    def __init__(
        self,
        url: str,
        *,
        prefix: str = "fastmlapi:",
        result_ttl: float = 3600.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisBackend requires the 'redis' package. "
                "Install it with: pip install fastmlapi[queue]"
            ) from exc
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._connection_errors = (
            redis.exceptions.ConnectionError,
            redis.exceptions.TimeoutError,
        )
        self._prefix = prefix
        self._result_ttl = int(result_ttl)
        self._clock = clock
        self._claim = self._r.register_script(_CLAIM)
        self._heartbeat = self._r.register_script(_HEARTBEAT)
        self._complete = self._r.register_script(_COMPLETE)
        self._fail = self._r.register_script(_FAIL)
        self._requeue = self._r.register_script(_REQUEUE_IF_STALE)

    def _k(self, name: str) -> str:
        return self._prefix + name

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        seconds, micros = self._r.time()
        return float(seconds) + float(micros) / 1_000_000

    @_guard
    def enqueue(self, payload: Any) -> str:
        encoded = json.dumps(payload)
        job_id = uuid.uuid4().hex
        pipe = self._r.pipeline(transaction=True)
        pipe.hset(
            self._k(f"job:{job_id}"),
            mapping={
                "status": QUEUED,
                "payload": encoded,
                "attempts": 0,
                "created_at": self._now(),
            },
        )
        pipe.lpush(self._k("pending"), job_id)
        pipe.execute()
        return job_id

    @_guard
    def claim(self, worker_id: str) -> Optional[Job]:
        job_id = self._claim(
            keys=[self._k("pending"), self._k(f"processing:{worker_id}")],
            args=[self._now(), self._k("job:"), worker_id],
        )
        return self.get_job(job_id) if job_id else None

    @_guard
    def heartbeat(self, worker_id: str, job_id: str) -> bool:
        return bool(
            self._heartbeat(
                keys=[self._k(f"job:{job_id}")], args=[worker_id, self._now()]
            )
        )

    @_guard
    def complete(self, worker_id: str, job_id: str, result: Any) -> bool:
        encoded = json.dumps(result)  # raises before touching Redis
        return bool(
            self._complete(
                keys=[self._k(f"job:{job_id}"), self._k(f"processing:{worker_id}")],
                args=[worker_id, job_id, self._now(), self._result_ttl, encoded],
            )
        )

    @_guard
    def fail(self, worker_id: str, job_id: str, error: str, error_type: str) -> bool:
        return bool(
            self._fail(
                keys=[self._k(f"job:{job_id}"), self._k(f"processing:{worker_id}")],
                args=[worker_id, job_id, self._now(), self._result_ttl, error, error_type],
            )
        )

    @_guard
    def get_job(self, job_id: str) -> Optional[Job]:
        h = self._r.hgetall(self._k(f"job:{job_id}"))
        return _to_job(job_id, h) if h else None

    @_guard
    def requeue_stale(self, visibility_timeout: float, max_attempts: int) -> int:
        now = self._now()
        touched = 0
        for key in self._r.scan_iter(match=self._k("processing:*")):
            # decode_responses=True, so the ids are str at runtime
            for job_id in cast(List[str], self._r.lrange(key, 0, -1)):
                outcome = self._requeue(
                    keys=[self._k(f"job:{job_id}"), key, self._k("pending")],
                    args=[now, visibility_timeout, max_attempts, job_id, self._result_ttl],
                )
                if outcome in (1, 2):
                    touched += 1
        return touched

    @_guard
    def pending_count(self) -> int:
        return int(self._r.llen(self._k("pending")))

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except self._connection_errors:
            return False

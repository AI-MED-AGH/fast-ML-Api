"""Redis queue backend.

Keys (all under `prefix`):
  pending                list of job ids: enqueue LPUSH, claim takes from the right
  processing:{worker}    list of job ids claimed by that worker
  job:{id}               hash with the job record
  worker_index           set of worker ids (so no key scanning is ever needed)
  workers:{id}           hash published by a live worker, with a TTL: model_loaded, started_at, last_seen

No Lua scripts and no SCAN/KEYS. Moving a job off `pending` is one atomic command (LMOVE); every other state change is an
optimistic transaction (WATCH the job, then MULTI/EXEC), so two sweepers, or a stalled worker and the sweeper, can never both
win. That keeps the backend inside the small set of commands in REDIS_COMMANDS_USED, which is what lets an operator give a
model a Redis user that may only touch its own keys and cannot run scripts or list other tenants' key names.

Requires Redis 6.2+ (LMOVE).
"""

from __future__ import annotations

import functools
import json
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .backend import FAILED, QUEUED, RUNNING, SUCCEEDED, Job, QueueBackend, QueueUnavailable, WorkerInfo

# Every Redis command this backend issues. Give a restricted Redis user exactly these (plus connection setup such as
# AUTH/HELLO/CLIENT SETINFO, which redis-py sends itself).
REDIS_COMMANDS_USED = (
    "LMOVE", "LPUSH", "RPUSH", "LREM", "LRANGE", "LLEN", "LPOS",
    "HSET", "HSETNX", "HGET", "HMGET", "HGETALL", "HDEL", "HINCRBY",
    "EXPIRE", "SADD", "SREM", "SMEMBERS", "DEL", "EXISTS",
    "WATCH", "UNWATCH", "MULTI", "EXEC", "DISCARD", "TIME", "PING",
)

_TX_RETRIES = 25


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn any Redis error (connection loss, timeout, OOM, READONLY after a
    failover, ...) into QueueUnavailable."""

    @functools.wraps(fn)
    def wrapper(self: "RedisBackend", *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except self._redis_errors as exc:
            raise QueueUnavailable(f"Redis unavailable: {exc}") from exc

    return wrapper


def _to_job(job_id: str, h: Mapping[Any, Any]) -> Job:
    def num(key: str) -> Optional[float]:
        value = h.get(key)
        return None if value is None or value == "" else float(value)

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
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisBackend requires the 'redis' package. "
                "Install it with: pip install fastmlapi[queue]"
            ) from exc
        # Without timeouts (redis-py < 8 default) a partitioned Redis hangs
        # /health, /jobs and workers. Options set in the URL take precedence.
        parsed = urlparse(url)
        in_url = parse_qs(parsed.query)
        options = [("socket_timeout", socket_timeout)]
        if parsed.scheme != "unix":
            # redis-py 4.x unix-socket connections reject this keyword
            options.append(("socket_connect_timeout", socket_connect_timeout))
        timeouts = {name: value for name, value in options if name not in in_url}
        # Any: with decode_responses=True replies are str, but the redis stubs type them as bytes | str, or as
        # Awaitable[...] | T, depending on the version
        self._r: Any = redis.Redis.from_url(url, decode_responses=True, **timeouts)
        self._redis_errors = redis.exceptions.RedisError
        self._watch_error = redis.exceptions.WatchError
        self._prefix = prefix
        self._result_ttl = max(1, int(result_ttl))  # EXPIRE 0 would delete at once
        self._clock = clock
        # (processing list, job id) -> when this process first saw a job there that was never marked running
        self._unmarked: Dict[Tuple[str, str], float] = {}

    def _k(self, name: str) -> str:
        return self._prefix + name

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        seconds, micros = self._r.time()
        return float(seconds) + float(micros) / 1_000_000

    def _transaction(self, watch_keys: List[str], body: Callable[[Any], Tuple[Any, Optional[Callable[[Any], None]]]]) -> Any:
        """Optimistic transaction.

        `body(pipe)` runs while `watch_keys` are watched (reads on `pipe` execute immediately) and returns
        `(result, writes)`. `writes(pipe)` queues the commands to run atomically, or is None to change nothing. If any
        watched key changed in between, the whole thing is retried against fresh data."""
        for _ in range(_TX_RETRIES):
            with self._r.pipeline() as pipe:
                try:
                    pipe.watch(*watch_keys)
                    result, writes = body(pipe)
                    if writes is None:
                        pipe.unwatch()
                        return result
                    pipe.multi()
                    writes(pipe)
                    pipe.execute()
                    return result
                except self._watch_error:
                    continue
        raise QueueUnavailable("could not update the job: it kept changing underneath us")

    # -- QueueBackend ------------------------------------------------------

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
        # one atomic command moves the job off `pending`, so two workers can never take the same one
        job_id = self._r.lmove(
            self._k("pending"), self._k(f"processing:{worker_id}"), "RIGHT", "LEFT"
        )
        if not job_id:
            return None
        now = self._now()
        pipe = self._r.pipeline(transaction=True)
        pipe.sadd(self._k("worker_index"), worker_id)
        pipe.hset(
            self._k(f"job:{job_id}"),
            mapping={"status": RUNNING, "started_at": now, "heartbeat": now, "worker": worker_id},
        )
        pipe.hincrby(self._k(f"job:{job_id}"), "attempts", 1)
        pipe.execute()
        return self.get_job(job_id)

    @_guard
    def heartbeat(self, worker_id: str, job_id: str) -> bool:
        job = self._k(f"job:{job_id}")
        now = self._now()

        def body(pipe: Any):
            status, owner = pipe.hmget(job, "status", "worker")
            if status != RUNNING or owner != worker_id:
                return False, None
            return True, lambda p: p.hset(job, "heartbeat", now)

        return bool(self._transaction([job], body))

    @_guard
    def complete(self, worker_id: str, job_id: str, result: Any) -> bool:
        encoded = json.dumps(result)  # raises before touching Redis
        job, processing = self._k(f"job:{job_id}"), self._k(f"processing:{worker_id}")
        now = self._now()

        def body(pipe: Any):
            status, owner = pipe.hmget(job, "status", "worker")
            if status != RUNNING or owner != worker_id:
                return False, None

            def writes(p: Any) -> None:
                p.lrem(processing, 0, job_id)
                p.hset(job, mapping={"status": SUCCEEDED, "result": encoded, "finished_at": now})
                p.hdel(job, "error", "error_type", "payload")  # the request body is no longer needed
                p.expire(job, self._result_ttl)

            return True, writes

        return bool(self._transaction([job], body))

    @_guard
    def fail(self, worker_id: str, job_id: str, error: str, error_type: str) -> bool:
        job, processing = self._k(f"job:{job_id}"), self._k(f"processing:{worker_id}")
        now = self._now()

        def body(pipe: Any):
            status, owner = pipe.hmget(job, "status", "worker")
            if status != RUNNING or owner != worker_id:
                return False, None

            def writes(p: Any) -> None:
                p.lrem(processing, 0, job_id)
                p.hset(
                    job,
                    mapping={"status": FAILED, "error": error, "error_type": error_type, "finished_at": now},
                )
                p.hdel(job, "result", "payload")
                p.expire(job, self._result_ttl)

            return True, writes

        return bool(self._transaction([job], body))

    @_guard
    def get_job(self, job_id: str) -> Optional[Job]:
        h = self._r.hgetall(self._k(f"job:{job_id}"))
        return _to_job(job_id, h) if h else None

    @_guard
    def requeue_stale(self, visibility_timeout: float, max_attempts: int) -> int:
        now = self._now()
        touched = 0
        for worker in sorted(self._r.smembers(self._k("worker_index"))):
            processing = self._k(f"processing:{worker}")
            job_ids = self._r.lrange(processing, 0, -1)
            for job_id in job_ids:
                if self._sweep_one(worker, processing, job_id, now, visibility_timeout, max_attempts):
                    touched += 1
            if not job_ids:
                self._forget_worker_if_idle(worker, processing)
        return touched

    def _sweep_one(
        self, worker: str, processing: str, job_id: str, now: float, visibility_timeout: float, max_attempts: int
    ) -> bool:
        job, pending = self._k(f"job:{job_id}"), self._k("pending")
        marker = (processing, job_id)

        def requeue(p: Any) -> None:
            p.lrem(processing, 0, job_id)
            p.hset(job, "status", QUEUED)
            p.hdel(job, "worker")
            p.rpush(pending, job_id)

        def body(pipe: Any):
            # every decision is made on fresh data, and only while this job and list are unchanged (WATCH)
            if pipe.lpos(processing, job_id) is None:
                self._unmarked.pop(marker, None)
                return 0, None  # someone else already dealt with it
            status, owner, heartbeat, attempts = pipe.hmget(job, "status", "worker", "heartbeat", "attempts")
            if status is None:  # the job record expired
                self._unmarked.pop(marker, None)
                return 0, lambda p: p.lrem(processing, 0, job_id)
            if status == RUNNING and owner == worker:
                self._unmarked.pop(marker, None)
                if now - float(heartbeat or 0.0) <= visibility_timeout:
                    return 0, None
                made = int(attempts or 0)
                if made >= max_attempts:
                    def fail_job(p: Any) -> None:
                        p.lrem(processing, 0, job_id)
                        p.hset(
                            job,
                            mapping={
                                "status": FAILED,
                                "error": f"Worker lost after {made} attempt(s)",
                                "error_type": "WorkerLost",
                                "finished_at": now,
                            },
                        )
                        p.hdel(job, "payload")
                        p.expire(job, self._result_ttl)

                    return 2, fail_job
                return 1, requeue
            if status == QUEUED and owner is None:
                # taken off `pending` but never marked running: its worker may have died in between, or may be about to
                # mark it. Only act once it has stayed that way for a whole visibility timeout.
                first = self._unmarked.setdefault(marker, now)
                if now - first <= visibility_timeout:
                    return 0, None
                self._unmarked.pop(marker, None)
                return 1, requeue
            # finished, or now owned by another worker: this id is only a leftover in this list
            self._unmarked.pop(marker, None)
            return 0, lambda p: p.lrem(processing, 0, job_id)

        return self._transaction([job, processing], body) in (1, 2)

    def _forget_worker_if_idle(self, worker: str, processing: str) -> None:
        """Drop a worker from the index once it has no jobs and no liveness entry."""
        alive = self._k(f"workers:{worker}")

        def body(pipe: Any):
            if pipe.llen(processing) or pipe.exists(alive):
                return False, None
            return True, lambda p: p.srem(self._k("worker_index"), worker)

        self._transaction([processing, alive], body)

    @_guard
    def pending_count(self) -> int:
        return int(self._r.llen(self._k("pending")))

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except self._redis_errors:
            return False

    # -- worker liveness ---------------------------------------------------

    @_guard
    def worker_alive(self, worker_id: str, model_loaded: bool, ttl: float) -> None:
        key, now = self._k(f"workers:{worker_id}"), self._now()
        pipe = self._r.pipeline(transaction=True)
        pipe.hsetnx(key, "started_at", now)
        pipe.hset(key, mapping={"model_loaded": "true" if model_loaded else "false", "last_seen": now})
        pipe.expire(key, max(1, int(ttl)))
        pipe.sadd(self._k("worker_index"), worker_id)
        pipe.execute()

    @_guard
    def worker_gone(self, worker_id: str) -> None:
        self._r.delete(self._k(f"workers:{worker_id}"))

    @_guard
    def list_workers(self) -> List[WorkerInfo]:
        found: List[WorkerInfo] = []
        for worker in sorted(self._r.smembers(self._k("worker_index"))):
            h = self._r.hgetall(self._k(f"workers:{worker}"))
            if not h:
                continue  # expired: it is dropped from the index by the next sweep
            found.append(
                WorkerInfo(
                    worker_id=worker,
                    model_loaded=h.get("model_loaded") == "true",
                    started_at=float(h["started_at"]) if "started_at" in h else None,
                    last_seen=float(h["last_seen"]) if "last_seen" in h else None,
                )
            )
        return found

# Redis Job Queue and Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `MLController` run heavy/long inference out-of-process: the API enqueues jobs into Redis and returns a job id; worker processes consume jobs, run the model, and store results that clients poll.

**Architecture:** A new `fastmlapi.queue` package holds a `QueueBackend` interface with `MemoryBackend` (dev/tests) and `RedisBackend` (Lua scripts for atomic claim/complete/requeue, fenced by worker id), a `Worker` loop (heartbeat thread, stale-job sweep, graceful shutdown), and the `/jobs` routes. `MLController` gains opt-in queue mode (`use_queue`, `run(queue_url=...)`, `run_worker(...)`); without it nothing changes.

**Tech Stack:** Python >=3.9, FastAPI/Starlette, Pydantic v2, `redis>=4.2.0` (optional extra), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-08-redis-job-queue-design.md`

## Global Constraints

- Python `>=3.9`: no runtime `X | Y` unions, no `match`; use `Optional[...]`. `from __future__ import annotations` is fine everywhere except `queue/api.py` (it needs real annotation objects for the dynamic request body type).
- `redis` is an optional dependency: extra `queue = ["redis>=4.2.0"]`, also added to `dev`. Importing `fastmlapi` or `fastmlapi.queue` must work without `redis` installed; only constructing `RedisBackend` needs it.
- Synchronous `/predict` behaviour is unchanged when queue mode is not enabled. All existing tests must pass unmodified.
- Payloads and results are JSON only. Redis key prefix is `fastmlapi:{model_name}:`.
- Defaults (class attributes on `MLController`): `result_ttl = 3600.0`, `visibility_timeout = 30.0`, `max_attempts = 3`, `heartbeat_interval = 5.0`.
- CI runs `ruff check src/` and `mypy src/ --ignore-missing-imports`; both must stay clean.
- Run tests with `uv run --extra dev pytest -q` from the repo root (there is no `python`/`pytest` on PATH).
- Every commit ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (shown as a second `-m` below).
- Work on the current branch `redis-job-queue-spec`.

## Review Focus

Failure modes the spec implies that a user is most likely to hit. Each has a test in the task named in brackets.

1. A result that is not JSON-serializable must end as `failed`/`ResultNotSerializable`, never stuck `running` [Task 1, Task 3].
2. A paused/zombie worker whose job was requeued must not overwrite the new owner's result [Task 1].
3. Backend down: `POST /jobs`, `GET /jobs/{id}` and `/health` answer 503 with a clear body, not a bare 500 [Task 4].
4. Unknown/expired job id gives 404; missing `data` or non-JSON body gives 422 and enqueues nothing [Task 4].
5. `fastmlapi[queue]` not installed: building a `RedisBackend` raises an `ImportError` that names the extra [Task 2].

## File Structure

| File | Responsibility |
|---|---|
| `src/fastmlapi/queue/__init__.py` | Public exports of the package |
| `src/fastmlapi/queue/backend.py` | `Job`, status constants, `QueueUnavailable`, `QueueBackend` ABC |
| `src/fastmlapi/queue/memory_backend.py` | Thread-safe in-process backend |
| `src/fastmlapi/queue/redis_backend.py` | Redis backend (Lua scripts) |
| `src/fastmlapi/queue/worker.py` | `Worker`: claim loop, heartbeat, sweep, shutdown |
| `src/fastmlapi/queue/api.py` | `register_queue_routes`: `POST /jobs`, `GET /jobs/{id}` |
| `src/fastmlapi/types.py` | + `JobSubmitResponse`, `JobStatusResponse`, extended `HealthResponse` |
| `src/fastmlapi/controller.py` | Queue-mode wiring, `use_queue`, `run(queue_url=)`, `run_worker` |
| `tests/test_queue_backends.py` | Backend contract tests (memory always; Redis when `REDIS_URL` set) |
| `tests/test_queue_memory.py` | Memory-only behaviour (TTL via fake clock) |
| `tests/test_queue_redis.py` | Redis-only behaviour (skipped without `REDIS_URL`) |
| `tests/test_queue_worker.py` | Worker tests |
| `tests/test_queue_api.py` | API + controller integration tests |
| `examples/queue_worker.py`, `README.md`, `CHANGELOG.md` | Docs |

---

### Task 1: Backend interface and MemoryBackend

**Files:**
- Create: `src/fastmlapi/queue/__init__.py`, `src/fastmlapi/queue/backend.py`, `src/fastmlapi/queue/memory_backend.py`
- Create: `tests/test_queue_backends.py`, `tests/test_queue_memory.py`
- Modify: `pyproject.toml` (optional extras)

**Interfaces:**
- Produces (used by every later task):
  - constants `QUEUED="queued"`, `RUNNING="running"`, `SUCCEEDED="succeeded"`, `FAILED="failed"`
  - `class QueueUnavailable(Exception)`
  - `@dataclass Job(id: str, status: str, payload: Any=None, result: Any=None, error: Optional[str]=None, error_type: Optional[str]=None, attempts: int=0, worker: Optional[str]=None, created_at: float=0.0, started_at: Optional[float]=None, finished_at: Optional[float]=None, heartbeat: Optional[float]=None)`
  - `QueueBackend` abstract methods: `enqueue(payload) -> str`; `claim(worker_id) -> Optional[Job]` (non-blocking, FIFO); `heartbeat(worker_id, job_id) -> bool`; `complete(worker_id, job_id, result) -> bool` (raises `TypeError`/`ValueError` if `result` is not JSON-serializable, leaving the job untouched); `fail(worker_id, job_id, error, error_type) -> bool`; `get_job(job_id) -> Optional[Job]`; `requeue_stale(visibility_timeout, max_attempts) -> int`; `pending_count() -> int`; `ping() -> bool`. `heartbeat`/`complete`/`fail` return `False` (and change nothing) unless the job is `running` and owned by `worker_id`.
  - `MemoryBackend(result_ttl: float = 3600.0, clock: Callable[[], float] = time.time)`

- [ ] **Step 1: Add the optional extras**

In `pyproject.toml`, under `[project.optional-dependencies]` add a `queue` extra and add redis to `dev`:

```toml
queue = [
    "redis>=4.2.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
    "numpy>=1.20.0",
    "redis>=4.2.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```

(`dev` keeps all its existing entries; only `redis>=4.2.0` is new.)

- [ ] **Step 2: Write the failing contract tests**

Create `tests/test_queue_backends.py`:

```python
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
```

Create `tests/test_queue_memory.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_queue_backends.py tests/test_queue_memory.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'fastmlapi.queue'`.

- [ ] **Step 4: Implement the interface**

Create `src/fastmlapi/queue/backend.py`:

```python
"""Queue backend interface shared by the API and the workers."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Optional

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


class QueueUnavailable(Exception):
    """The queue backend cannot be reached (e.g. Redis is down)."""


@dataclasses.dataclass
class Job:
    id: str
    status: str
    payload: Any = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attempts: int = 0
    worker: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    heartbeat: Optional[float] = None


class QueueBackend(abc.ABC):
    """Storage and hand-off of jobs.

    `heartbeat`, `complete` and `fail` only act when the job is `running` and
    owned by `worker_id`; otherwise they return False and change nothing. This
    stops a stalled worker from overwriting a job that was requeued and taken
    by someone else.
    """

    @abc.abstractmethod
    def enqueue(self, payload: Any) -> str:
        """Store a job (payload must be JSON-serializable) and return its id."""

    @abc.abstractmethod
    def claim(self, worker_id: str) -> Optional[Job]:
        """Atomically take the oldest pending job (non-blocking)."""

    @abc.abstractmethod
    def heartbeat(self, worker_id: str, job_id: str) -> bool:
        """Refresh the job's heartbeat."""

    @abc.abstractmethod
    def complete(self, worker_id: str, job_id: str, result: Any) -> bool:
        """Mark succeeded. Raises TypeError/ValueError, leaving the job
        untouched, if `result` is not JSON-serializable."""

    @abc.abstractmethod
    def fail(self, worker_id: str, job_id: str, error: str, error_type: str) -> bool:
        """Mark failed."""

    @abc.abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Return the job, or None if unknown or expired."""

    @abc.abstractmethod
    def requeue_stale(self, visibility_timeout: float, max_attempts: int) -> int:
        """Requeue running jobs whose heartbeat is older than
        `visibility_timeout`; jobs already claimed `max_attempts` times are
        failed with error_type "WorkerLost" instead. Returns how many jobs
        were touched."""

    @abc.abstractmethod
    def pending_count(self) -> int:
        """Number of jobs waiting to be claimed."""

    @abc.abstractmethod
    def ping(self) -> bool:
        """True if the backend is reachable."""
```

Create `src/fastmlapi/queue/memory_backend.py`:

```python
"""In-process queue backend for local development and tests."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .backend import FAILED, QUEUED, RUNNING, SUCCEEDED, Job, QueueBackend


class MemoryBackend(QueueBackend):
    def __init__(
        self,
        result_ttl: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ):
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._pending: Deque[str] = deque()  # claim pops from the right
        self._processing: Dict[str, List[str]] = {}
        self._expiry: Dict[str, float] = {}
        self._result_ttl = result_ttl
        self._clock = clock

    # -- helpers (call with the lock held) ---------------------------------

    def _expire(self) -> None:
        now = self._clock()
        for job_id in [j for j, t in self._expiry.items() if t <= now]:
            self._jobs.pop(job_id, None)
            self._expiry.pop(job_id, None)

    def _owned(self, worker_id: str, job_id: str) -> Optional[Job]:
        self._expire()
        job = self._jobs.get(job_id)
        if job and job.status == RUNNING and job.worker == worker_id:
            return job
        return None

    def _finish(self, worker_id: str, job: Job) -> None:
        now = self._clock()
        job.finished_at = now
        ids = self._processing.get(worker_id, [])
        if job.id in ids:
            ids.remove(job.id)
        self._expiry[job.id] = now + self._result_ttl

    # -- QueueBackend ------------------------------------------------------

    def enqueue(self, payload: Any) -> str:
        payload = json.loads(json.dumps(payload))  # same JSON semantics as Redis
        with self._lock:
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = Job(
                id=job_id, status=QUEUED, payload=payload, created_at=self._clock()
            )
            self._pending.appendleft(job_id)
            return job_id

    def claim(self, worker_id: str) -> Optional[Job]:
        with self._lock:
            self._expire()
            if not self._pending:
                return None
            job_id = self._pending.pop()
            job = self._jobs[job_id]
            now = self._clock()
            job.status = RUNNING
            job.worker = worker_id
            job.attempts += 1
            job.started_at = now
            job.heartbeat = now
            self._processing.setdefault(worker_id, []).append(job_id)
            return dataclasses.replace(job)

    def heartbeat(self, worker_id: str, job_id: str) -> bool:
        with self._lock:
            job = self._owned(worker_id, job_id)
            if job is None:
                return False
            job.heartbeat = self._clock()
            return True

    def complete(self, worker_id: str, job_id: str, result: Any) -> bool:
        encoded = json.dumps(result)  # raises before any state changes
        with self._lock:
            job = self._owned(worker_id, job_id)
            if job is None:
                return False
            job.status = SUCCEEDED
            job.result = json.loads(encoded)
            job.error = None
            job.error_type = None
            self._finish(worker_id, job)
            return True

    def fail(self, worker_id: str, job_id: str, error: str, error_type: str) -> bool:
        with self._lock:
            job = self._owned(worker_id, job_id)
            if job is None:
                return False
            job.status = FAILED
            job.error = error
            job.error_type = error_type
            self._finish(worker_id, job)
            return True

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._expire()
            job = self._jobs.get(job_id)
            return dataclasses.replace(job) if job else None

    def requeue_stale(self, visibility_timeout: float, max_attempts: int) -> int:
        with self._lock:
            self._expire()
            now = self._clock()
            touched = 0
            for ids in self._processing.values():
                for job_id in list(ids):
                    job = self._jobs.get(job_id)
                    if job is None:
                        ids.remove(job_id)
                        continue
                    if now - (job.heartbeat or 0.0) <= visibility_timeout:
                        continue
                    ids.remove(job_id)
                    if job.attempts >= max_attempts:
                        job.status = FAILED
                        job.error = f"Worker lost after {job.attempts} attempt(s)"
                        job.error_type = "WorkerLost"
                        job.finished_at = now
                        self._expiry[job_id] = now + self._result_ttl
                    else:
                        job.status = QUEUED
                        job.worker = None
                        self._pending.append(job_id)
                    touched += 1
            return touched

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def ping(self) -> bool:
        return True
```

Create `src/fastmlapi/queue/__init__.py`:

```python
"""Job queue support: submit work to the API, run it on separate workers."""

from .backend import (
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    Job,
    QueueBackend,
    QueueUnavailable,
)
from .memory_backend import MemoryBackend
from .redis_backend import RedisBackend
from .worker import Worker

__all__ = [
    "FAILED",
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "Job",
    "MemoryBackend",
    "QueueBackend",
    "QueueUnavailable",
    "RedisBackend",
    "Worker",
]
```

This `__init__` imports modules created in later tasks. For this task only, create it with just the `.backend` and `.memory_backend` imports and the matching `__all__` entries; Task 2 and Task 3 add the other two lines.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_queue_backends.py tests/test_queue_memory.py -q`
Expected: all memory cases PASS; Redis cases reported as skipped.

- [ ] **Step 6: Run the full suite (regression)**

Run: `uv run --extra dev pytest -q`
Expected: PASS (existing tests untouched).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/fastmlapi/queue tests/test_queue_backends.py tests/test_queue_memory.py
git commit -m "feat(queue): add QueueBackend interface and MemoryBackend" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: RedisBackend

**Files:**
- Create: `src/fastmlapi/queue/redis_backend.py`, `tests/test_queue_redis.py`
- Modify: `src/fastmlapi/queue/__init__.py` (add `RedisBackend` import/export)
- Modify: `.github/workflows/tests.yml`, `.github/workflows/test.yml` (Redis service)

**Interfaces:**
- Consumes: everything produced by Task 1.
- Produces: `RedisBackend(url: str, *, prefix: str = "fastmlapi:", result_ttl: float = 3600.0, clock: Optional[Callable[[], float]] = None)`. When `clock` is None, timestamps use the Redis server clock (`TIME`), so workers on different machines agree. Redis connection errors surface as `QueueUnavailable`. Attribute `_r` is the `redis.Redis` client (tests use it).

- [ ] **Step 1: Write the Redis-only tests**

Create `tests/test_queue_redis.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_queue_redis.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fastmlapi.queue.redis_backend'`.

- [ ] **Step 3: Implement RedisBackend**

Create `src/fastmlapi/queue/redis_backend.py`:

```python
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
from typing import Any, Callable, Dict, Optional

from .backend import FAILED, QUEUED, Job, QueueBackend, QueueUnavailable

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


def _to_job(job_id: str, h: Dict[str, str]) -> Job:
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
            for job_id in self._r.lrange(key, 0, -1):
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
```

(`FAILED` is imported for symmetry with the other backend but unused; remove it from the import if ruff flags it.)

- [ ] **Step 4: Export from the package**

In `src/fastmlapi/queue/__init__.py` add `from .redis_backend import RedisBackend` and `"RedisBackend"` to `__all__`.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra dev pytest tests/test_queue_redis.py tests/test_queue_backends.py -q`
Expected without Redis: `test_unreachable_redis_raises_queue_unavailable` and `test_missing_redis_package_raises_helpful_import_error` PASS; Redis-marked cases skipped.

If a Redis server is available (e.g. `docker run --rm -p 6379:6379 redis:7`), also run:
`REDIS_URL=redis://localhost:6379/0 uv run --extra dev pytest tests/test_queue_redis.py tests/test_queue_backends.py -q`
Expected: every case PASS, including the Redis parametrization of the whole contract suite. If no Redis is available, say so when reporting; do not claim Redis was verified.

- [ ] **Step 6: Add Redis to CI**

In both `.github/workflows/tests.yml` and `.github/workflows/test.yml`, under the `test:` job (next to `runs-on`), add:

```yaml
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 10
    env:
      REDIS_URL: redis://localhost:6379/0
```

- [ ] **Step 7: Lint and commit**

Run: `uv run --extra dev ruff check src/` then `uv run --extra dev mypy src/ --ignore-missing-imports`
Expected: both clean (fix anything they report).

```bash
git add src/fastmlapi/queue tests/test_queue_redis.py .github/workflows
git commit -m "feat(queue): add Redis backend with atomic Lua claim/complete/requeue" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Worker

**Files:**
- Create: `src/fastmlapi/queue/worker.py`, `tests/test_queue_worker.py`
- Modify: `src/fastmlapi/queue/__init__.py` (add `Worker`)
- Modify: `src/fastmlapi/controller.py` (class attributes only, so the worker can read them)

**Interfaces:**
- Consumes: `QueueBackend`, `Job`, `QueueUnavailable` (Task 1); `MLController._initialize()` and `async MLController.predict(data)` (existing).
- Produces: `Worker(controller, backend, *, worker_id=None, poll_interval=0.5, sweep_interval=None, backoff_initial=1.0, backoff_max=30.0)` with `run()` (initializes the controller, loops until stopped; exceptions from initialization propagate), `run_once() -> bool` (claim and process one job; False if queue empty; the controller must already be initialized), `stop()`. `controller` must expose `result_ttl`, `visibility_timeout: float`, `max_attempts: int`, `heartbeat_interval: float`. `sweep_interval` defaults to `max(visibility_timeout / 2, 1.0)`.

- [ ] **Step 1: Add the controller class attributes**

In `src/fastmlapi/controller.py`, directly after the `enable_docs: bool = True` line add:

```python
    # Job queue tuning (only used in queue mode / by workers)
    result_ttl: float = 3600.0          # seconds a finished job's result is kept
    visibility_timeout: float = 30.0    # seconds without a heartbeat before a job is requeued
    max_attempts: int = 3               # claims before a job is failed as "worker lost"
    heartbeat_interval: float = 5.0     # seconds between worker heartbeats
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_queue_worker.py`:

```python
"""Worker tests (MemoryBackend, real controllers)."""

import threading
import time

import pytest

from fastmlapi import MLController, postprocessing, prediction, preprocessing
from fastmlapi.queue.backend import QueueUnavailable
from fastmlapi.queue.memory_backend import MemoryBackend
from fastmlapi.queue.worker import Worker
from tests.test_queue_backends import FakeClock


class Doubler(MLController):
    model_name = "doubler"

    @preprocessing
    def preprocess(self, data):
        return data["n"]

    @prediction
    def infer(self, n):
        if n == 0:
            raise ValueError("boom")
        return n * 2

    @postprocessing
    def postprocess(self, out):
        return {"doubled": out}


def make_worker(controller=None, backend=None, **kwargs):
    controller = controller or Doubler()
    backend = backend or MemoryBackend()
    controller._initialize()
    kwargs.setdefault("poll_interval", 0.01)
    return controller, backend, Worker(controller, backend, **kwargs)


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_run_once_returns_false_when_queue_is_empty():
    _, _, worker = make_worker()
    assert worker.run_once() is False


def test_run_once_processes_a_job():
    _, backend, worker = make_worker()
    job_id = backend.enqueue({"n": 4})
    assert worker.run_once() is True
    job = backend.get_job(job_id)
    assert job.status == "succeeded"
    assert job.result == {"doubled": 8}


def test_model_exception_fails_the_job_and_the_worker_keeps_going():
    _, backend, worker = make_worker()
    bad = backend.enqueue({"n": 0})
    good = backend.enqueue({"n": 1})
    worker.run_once()
    worker.run_once()
    failed = backend.get_job(bad)
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.error_type == "ValueError"
    assert backend.get_job(good).status == "succeeded"


def test_unserializable_result_fails_the_job_instead_of_leaving_it_running():
    class Unserializable(Doubler):
        @postprocessing
        def postprocess(self, out):
            return {"obj": object()}

    _, backend, worker = make_worker(Unserializable())
    job_id = backend.enqueue({"n": 1})
    worker.run_once()
    job = backend.get_job(job_id)
    assert job.status == "failed"
    assert job.error_type == "ResultNotSerializable"


def test_heartbeat_is_refreshed_while_a_job_runs():
    class Slow(Doubler):
        heartbeat_interval = 0.05
        seen = None
        backend = None
        job_id = None

        @prediction
        def infer(self, n):
            first = self.backend.get_job(self.job_id).heartbeat
            time.sleep(0.4)
            self.seen = (first, self.backend.get_job(self.job_id).heartbeat)
            return n

    controller = Slow()
    backend = MemoryBackend()
    controller.backend = backend
    controller.job_id = backend.enqueue({"n": 1})
    _, _, worker = make_worker(controller, backend)
    worker.run_once()
    first, later = controller.seen
    assert later > first


def test_run_loop_processes_jobs_until_stopped():
    controller = Doubler()
    backend = MemoryBackend()
    worker = Worker(controller, backend, poll_interval=0.01)
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        ids = [backend.enqueue({"n": i + 1}) for i in range(3)]
        assert wait_for(lambda: all(backend.get_job(i).status == "succeeded" for i in ids))
    finally:
        worker.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_stop_lets_the_current_job_finish():
    class Slow(Doubler):
        @prediction
        def infer(self, n):
            time.sleep(0.3)
            return n

    backend = MemoryBackend()
    worker = Worker(Slow(), backend, poll_interval=0.01)
    thread = threading.Thread(target=worker.run)
    thread.start()
    job_id = backend.enqueue({"n": 1})
    assert wait_for(lambda: backend.get_job(job_id).status == "running")
    worker.stop()
    thread.join(timeout=5)
    assert backend.get_job(job_id).status == "succeeded"


def test_startup_failure_propagates():
    class BrokenModel(Doubler):
        def load_model(self):
            raise RuntimeError("cannot load")

    worker = Worker(BrokenModel(), MemoryBackend())
    with pytest.raises(RuntimeError, match="cannot load"):
        worker.run()


def test_worker_requeues_jobs_abandoned_by_dead_workers():
    clock = FakeClock()
    backend = MemoryBackend(clock=clock)
    job_id = backend.enqueue({"n": 5})
    backend.claim("dead-worker")
    clock.advance(100)  # far beyond visibility_timeout (30)
    _, _, worker = make_worker(backend=backend, sweep_interval=0)
    assert worker.run_once() is True
    job = backend.get_job(job_id)
    assert job.status == "succeeded"
    assert job.attempts == 2


def test_queue_outage_is_retried_with_backoff():
    class Flaky(MemoryBackend):
        failures = 2

        def claim(self, worker_id):
            if self.failures > 0:
                self.failures -= 1
                raise QueueUnavailable("down")
            return super().claim(worker_id)

    backend = Flaky()
    job_id = backend.enqueue({"n": 3})
    worker = Worker(
        Doubler(), backend, poll_interval=0.01, backoff_initial=0.01, backoff_max=0.05
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert wait_for(lambda: backend.get_job(job_id).status == "succeeded")
    finally:
        worker.stop()
        thread.join(timeout=5)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_queue_worker.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fastmlapi.queue.worker'`.

- [ ] **Step 4: Implement the worker**

Create `src/fastmlapi/queue/worker.py`:

```python
"""Worker loop: claim jobs, run the model pipeline, store results."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from .backend import Job, QueueBackend, QueueUnavailable

if TYPE_CHECKING:
    from ..controller import MLController

logger = logging.getLogger(__name__)


class Worker:
    """Runs one job at a time. Scale by starting more worker processes."""

    def __init__(
        self,
        controller: "MLController",
        backend: QueueBackend,
        *,
        worker_id: Optional[str] = None,
        poll_interval: float = 0.5,
        sweep_interval: Optional[float] = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self.controller = controller
        self.backend = backend
        self.worker_id = worker_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        )
        self.poll_interval = poll_interval
        self.sweep_interval = (
            sweep_interval
            if sweep_interval is not None
            else max(controller.visibility_timeout / 2, 1.0)
        )
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self._backoff = backoff_initial
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_sweep = float("-inf")

    # -- public ------------------------------------------------------------

    def stop(self) -> None:
        """Finish the current job, then leave `run()`."""
        self._stop.set()

    def run(self) -> None:
        """Initialize the model, then process jobs until stopped.

        Exceptions from `setup()`/`load_model()` propagate so the process
        exits non-zero and the orchestrator restarts it.
        """
        self.controller._initialize()
        restore = self._install_signal_handlers()
        logger.info("Worker %s started for %s", self.worker_id, self.controller.model_name)
        try:
            while not self._stop.is_set():
                if not self._tick():
                    self._stop.wait(self.poll_interval)
        finally:
            restore()
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            logger.info("Worker %s stopped", self.worker_id)

    def run_once(self) -> bool:
        """Claim and process one job. Returns False if the queue was empty."""
        job = self.backend.claim(self.worker_id)
        if job is None:
            return False
        self._process(job)
        return True

    # -- internals ---------------------------------------------------------

    def _tick(self) -> bool:
        try:
            self._maybe_sweep()
            processed = self.run_once()
        except QueueUnavailable as exc:
            logger.warning("Queue unavailable (%s); retrying in %.1fs", exc, self._backoff)
            self._stop.wait(self._backoff)
            self._backoff = min(self._backoff * 2, self.backoff_max)
            return True
        self._backoff = self.backoff_initial
        return processed

    def _maybe_sweep(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep < self.sweep_interval:
            return
        touched = self.backend.requeue_stale(
            self.controller.visibility_timeout, self.controller.max_attempts
        )
        self._last_sweep = now
        if touched:
            logger.warning("Requeued or failed %d stale job(s)", touched)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _process(self, job: Job) -> None:
        stop_heartbeat = threading.Event()
        beat = threading.Thread(
            target=self._heartbeat_loop, args=(job.id, stop_heartbeat), daemon=True
        )
        beat.start()
        outcome: Tuple[Any, ...]
        try:
            try:
                result = self._get_loop().run_until_complete(
                    self.controller.predict(job.payload)
                )
                outcome = ("complete", result)
            except Exception as exc:
                logger.exception("Job %s failed", job.id)
                outcome = ("fail", str(exc), type(exc).__name__)
        finally:
            stop_heartbeat.set()
            beat.join()
        self._finish(job, outcome)

    def _finish(self, job: Job, outcome: Tuple[Any, ...]) -> None:
        if outcome[0] == "complete":
            try:
                accepted = self._retry(
                    lambda: self.backend.complete(self.worker_id, job.id, outcome[1])
                )
            except (TypeError, ValueError) as exc:
                accepted = self._retry(
                    lambda: self.backend.fail(
                        self.worker_id,
                        job.id,
                        f"Result is not JSON-serializable: {exc}",
                        "ResultNotSerializable",
                    )
                )
        else:
            accepted = self._retry(
                lambda: self.backend.fail(self.worker_id, job.id, outcome[1], outcome[2])
            )
        if accepted is False:
            logger.warning("Job %s was reassigned before it finished; result dropped", job.id)

    def _retry(self, fn: Callable[[], Any]) -> Any:
        """Retry a result write through Redis outages. If a stop was requested
        meanwhile, give up and let the visibility timeout recover the job."""
        delay = self.backoff_initial
        while True:
            try:
                return fn()
            except QueueUnavailable:
                if self._stop.is_set():
                    raise
                logger.warning("Queue unavailable while saving a result; retrying in %.1fs", delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.backoff_max)

    def _heartbeat_loop(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(self.controller.heartbeat_interval):
            try:
                if not self.backend.heartbeat(self.worker_id, job_id):
                    logger.warning("Job %s is no longer owned by this worker", job_id)
                    return
            except QueueUnavailable:
                logger.warning("Heartbeat for job %s failed: queue unavailable", job_id)

    def _install_signal_handlers(self) -> Callable[[], None]:
        if threading.current_thread() is not threading.main_thread():
            return lambda: None
        previous = {}

        def handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %s; finishing the current job", signum)
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, handler)

        def restore() -> None:
            for sig, old in previous.items():
                signal.signal(sig, old)

        return restore
```

- [ ] **Step 5: Export `Worker`**

In `src/fastmlapi/queue/__init__.py` add `from .worker import Worker` and `"Worker"` to `__all__`.

- [ ] **Step 6: Run the tests**

Run: `uv run --extra dev pytest tests/test_queue_worker.py -q`
Expected: PASS (all 9).

- [ ] **Step 7: Lint, full suite, commit**

Run: `uv run --extra dev ruff check src/ && uv run --extra dev mypy src/ --ignore-missing-imports && uv run --extra dev pytest -q`
Expected: all clean/PASS.

```bash
git add src/fastmlapi tests/test_queue_worker.py
git commit -m "feat(queue): add Worker with heartbeat, stale sweep and graceful shutdown" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `/jobs` API and controller integration

**Files:**
- Create: `src/fastmlapi/queue/api.py`, `tests/test_queue_api.py`
- Modify: `src/fastmlapi/types.py`, `src/fastmlapi/controller.py`

**Interfaces:**
- Consumes: `QueueBackend`, `QueueUnavailable` (Task 1), `RedisBackend` (Task 2), `Worker` (Task 3), controller attributes added in Task 3.
- Produces:
  - `types.JobSubmitResponse(job_id: str)`; `types.JobStatusResponse(job_id, status, result=None, error=None, error_type=None, attempts=0, created_at=None, started_at=None, finished_at=None)`
  - `HealthResponse` gains `queue_connected: Optional[bool] = None`, `queue_depth: Optional[int] = None`; `model_loaded` becomes `Optional[bool] = None`
  - `register_queue_routes(app, controller, backend, request_model)`
  - `MLController.use_queue(backend)` (raises `RuntimeError` once the app exists), `MLController.run(..., queue_url: Optional[str] = None)`, `MLController.run_worker(redis_url=None, *, backend=None, **worker_kwargs)` (raises `ValueError` if neither is given; calls `sys.exit(1)` if the worker fails to start)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_queue_api.py`:

```python
"""API and controller integration for queue mode."""

import uvicorn
import pytest
from fastapi.testclient import TestClient

from fastmlapi import MLController, postprocessing, prediction, preprocessing
from fastmlapi.queue.backend import QueueUnavailable
from fastmlapi.queue.memory_backend import MemoryBackend
from fastmlapi.queue.worker import Worker
from tests.test_queue_backends import FakeClock


class Doubler(MLController):
    model_name = "doubler"
    loads = []

    def load_model(self):
        type(self).loads.append(1)
        return None

    @preprocessing
    def preprocess(self, data):
        return data["n"]

    @prediction
    def infer(self, n):
        if n == 0:
            raise ValueError("boom")
        return n * 2

    @postprocessing
    def postprocess(self, out):
        return {"doubled": out}


@pytest.fixture
def env():
    Doubler.loads = []
    backend = MemoryBackend()
    api = Doubler()
    api.use_queue(backend)
    worker_ctrl = Doubler()
    worker_ctrl._initialize()
    worker = Worker(worker_ctrl, backend, poll_interval=0.01)
    Doubler.loads = []  # only count loads caused by the API from here on
    with TestClient(api.app) as client:
        yield type("Env", (), dict(client=client, backend=backend, worker=worker, api=api))


def submit(client, n):
    r = client.post("/jobs", json={"data": {"n": n}})
    assert r.status_code == 202
    return r.json()["job_id"]


def test_submit_returns_202_and_job_is_queued(env):
    job_id = submit(env.client, 4)
    body = env.client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "queued"
    assert body["job_id"] == job_id


def test_full_submit_run_poll_cycle(env):
    job_id = submit(env.client, 4)
    env.worker.run_once()
    body = env.client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"doubled": 8}


def test_failed_job_exposes_the_error(env):
    job_id = submit(env.client, 0)
    env.worker.run_once()
    body = env.client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert body["error"] == "boom"
    assert body["error_type"] == "ValueError"


def test_unknown_job_id_is_404(env):
    r = env.client.get("/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"]["error_type"] == "JobNotFound"


def test_invalid_body_is_422_and_enqueues_nothing(env):
    assert env.client.post("/jobs", json={"nope": 1}).status_code == 422
    assert env.client.post("/jobs", content=b"not json",
                           headers={"content-type": "application/json"}).status_code == 422
    assert env.backend.pending_count() == 0


def test_predict_is_not_registered_in_queue_mode(env):
    assert env.client.post("/predict", json={"data": {"n": 1}}).status_code in (404, 405)


def test_api_process_does_not_load_the_model(env):
    assert Doubler.loads == []


def test_health_reports_queue_state_without_model_loaded(env):
    submit(env.client, 1)
    r = env.client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["queue_connected"] is True
    assert body["queue_depth"] == 1
    assert "model_loaded" not in body
    assert env.client.get("/livez").status_code == 200


def test_results_expire_after_ttl():
    clock = FakeClock()
    backend = MemoryBackend(result_ttl=100, clock=clock)
    api = Doubler()
    api.use_queue(backend)
    worker_ctrl = Doubler()
    worker_ctrl._initialize()
    worker = Worker(worker_ctrl, backend, poll_interval=0.01)
    with TestClient(api.app) as client:
        job_id = submit(client, 1)
        worker.run_once()
        assert client.get(f"/jobs/{job_id}").status_code == 200
        clock.advance(101)
        assert client.get(f"/jobs/{job_id}").status_code == 404


def test_backend_down_gives_503_not_500():
    class Down(MemoryBackend):
        def enqueue(self, payload):
            raise QueueUnavailable("redis is down")

        def get_job(self, job_id):
            raise QueueUnavailable("redis is down")

        def ping(self):
            return False

    api = Doubler()
    api.use_queue(Down())
    with TestClient(api.app) as client:
        r = client.post("/jobs", json={"data": {"n": 1}})
        assert r.status_code == 503
        assert r.json()["detail"]["error_type"] == "QueueUnavailable"
        assert client.get("/jobs/x").status_code == 503
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["queue_connected"] is False


def test_jobs_routes_absent_in_sync_mode():
    with TestClient(Doubler().app) as client:
        assert client.post("/jobs", json={"data": {"n": 1}}).status_code == 404


def test_use_queue_after_app_creation_raises():
    ctrl = Doubler()
    ctrl.app
    with pytest.raises(RuntimeError):
        ctrl.use_queue(MemoryBackend())


def test_run_with_queue_url_enables_queue_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "fastmlapi.queue.redis_backend.RedisBackend",
        lambda url, **kw: seen.update(url=url, **kw) or MemoryBackend(),
    )
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app))
    Doubler().run(queue_url="redis://example:6379/0")
    assert seen["url"] == "redis://example:6379/0"
    assert seen["prefix"] == "fastmlapi:doubler:"
    assert any(getattr(r, "path", None) == "/jobs" for r in seen["app"].routes)


def test_run_worker_requires_a_backend():
    with pytest.raises(ValueError):
        Doubler().run_worker()


def test_run_worker_exits_nonzero_when_the_model_cannot_load():
    class Broken(Doubler):
        def load_model(self):
            raise RuntimeError("cannot load")

    with pytest.raises(SystemExit) as excinfo:
        Broken().run_worker(backend=MemoryBackend())
    assert excinfo.value.code == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_queue_api.py -q`
Expected: FAIL (`AttributeError: ... has no attribute 'use_queue'`).

- [ ] **Step 3: Extend `types.py`**

In `src/fastmlapi/types.py` replace `HealthResponse` and append the job models:

```python
class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(default="healthy")
    model_loaded: Optional[bool] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    queue_connected: Optional[bool] = Field(default=None)
    queue_depth: Optional[int] = Field(default=None)


class JobSubmitResponse(BaseModel):
    """Returned by POST /jobs."""

    job_id: str


class JobStatusResponse(BaseModel):
    """Returned by GET /jobs/{job_id}."""

    job_id: str
    status: str = Field(description="queued, running, succeeded or failed")
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attempts: int = 0
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
```

- [ ] **Step 4: Create the routes**

Create `src/fastmlapi/queue/api.py` (no `from __future__ import annotations` here; FastAPI needs the real `request_model` object as the body annotation):

```python
"""HTTP routes for queue mode."""

from typing import Any, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..types import JobStatusResponse, JobSubmitResponse
from .backend import QueueBackend, QueueUnavailable


def _unavailable(exc: QueueUnavailable) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"error": str(exc), "error_type": "QueueUnavailable"},
    )


def register_queue_routes(
    app: FastAPI,
    controller: Any,
    backend: QueueBackend,
    request_model: Type[BaseModel],
) -> None:
    @app.post(
        "/jobs",
        status_code=202,
        response_model=JobSubmitResponse,
        tags=["Jobs"],
        summary=f"Submit a {controller.model_name} job",
    )
    async def submit_job(body: request_model):  # type: ignore[valid-type]
        if hasattr(body, "data"):
            data = body.data
        else:
            data = body.model_dump(mode="json")
        try:
            job_id = await run_in_threadpool(backend.enqueue, data)
        except QueueUnavailable as exc:
            raise _unavailable(exc)
        return JobSubmitResponse(job_id=job_id)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["Jobs"],
        summary="Get a job's status and result",
    )
    async def get_job(job_id: str):
        try:
            job = await run_in_threadpool(backend.get_job, job_id)
        except QueueUnavailable as exc:
            raise _unavailable(exc)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Job {job_id} not found or expired",
                    "error_type": "JobNotFound",
                },
            )
        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            result=job.result,
            error=job.error,
            error_type=job.error_type,
            attempts=job.attempts,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
```

- [ ] **Step 5: Wire it into the controller**

All in `src/fastmlapi/controller.py`:

1. Imports: add `import sys` if not already imported; add `from starlette.concurrency import run_in_threadpool`; add `from .queue.backend import QueueBackend, QueueUnavailable`.

2. In `__init__`, after `self._app: Optional[FastAPI] = None` add:
```python
        self._queue_backend: Optional[QueueBackend] = None
```

3. Add these methods right after `add_route`:
```python
    def use_queue(self, backend: QueueBackend) -> None:
        """Run in queue mode: `/jobs` replaces `/predict` and this process
        does not load the model (start workers with `run_worker()`).

        Must be called before the app is created.
        """
        if self._app is not None:
            raise RuntimeError(
                "Cannot enable the job queue after the app has been created. "
                "Call use_queue() (or run(queue_url=...)) before accessing .app"
            )
        self._queue_backend = backend

    def _redis_backend(self, url: str) -> QueueBackend:
        from .queue.redis_backend import RedisBackend

        return RedisBackend(
            url, prefix=f"fastmlapi:{self.model_name}:", result_ttl=self.result_ttl
        )

    def _initialize_api_only(self) -> None:
        """Queue-mode API startup: no model, workers load it."""
        logger.info("Initializing controller (queue mode, model not loaded)")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.setup()
        self._ready = True

    def run_worker(
        self,
        redis_url: Optional[str] = None,
        *,
        backend: Optional[QueueBackend] = None,
        **worker_kwargs: Any,
    ) -> None:
        """Run a queue worker in this process: load the model, then process
        jobs until SIGINT/SIGTERM. Exits non-zero if startup fails."""
        if backend is None:
            if redis_url is None:
                raise ValueError("run_worker() needs redis_url (or backend=)")
            backend = self._redis_backend(redis_url)
        from .queue.worker import Worker

        print(f"Starting worker for {self.model_name} v{self.model_version}...")
        try:
            Worker(self, backend, **worker_kwargs).run()
        except Exception:
            logger.exception("Worker stopped: failed to start or fatal error")
            sys.exit(1)
```

4. Replace `health_check` with:
```python
    def health_check(self) -> Dict[str, Any]:
        """Return health check information."""
        backend = self._queue_backend
        if backend is not None:
            try:
                connected = backend.ping()
                depth = backend.pending_count() if connected else None
            except QueueUnavailable:
                connected, depth = False, None
            return {
                "status": "healthy" if connected else "not_ready",
                "model_name": self.model_name,
                "version": self.model_version,
                "queue_connected": connected,
                "queue_depth": depth,
            }
        ready = self._ready and self.is_loaded
        return {
            "status": "healthy" if ready else "not_ready",
            "model_loaded": ready,
            "model_name": self.model_name,
            "version": self.model_version,
        }
```

5. In `_create_app`:
   - In `lifespan`, replace `controller._initialize()` with:
     ```python
                if controller._queue_backend is not None:
                    controller._initialize_api_only()
                else:
                    controller._initialize()
     ```
   - On the `/health` route add `response_model_exclude_none=True` to the `@app.get(...)` arguments, and change its body to `result = await run_in_threadpool(controller.health_check)`.
   - In `/info`, set `"endpoint": "/jobs" if controller._queue_backend is not None else "/predict"`.
   - Move the predict endpoint (the `async def predict_endpoint(...)` and the following `app.add_api_route("/predict", ...)` call) verbatim into a new method `def _register_predict_route(self, app, RequestModel, ResponseModel)`, replacing `controller` with `self` inside it. In `_create_app`, in its place:
     ```python
        if self._queue_backend is not None:
            from .queue.api import register_queue_routes

            register_queue_routes(app, self, self._queue_backend, RequestModel)
        else:
            self._register_predict_route(app, RequestModel, ResponseModel)
     ```

6. In `run`: add the parameter `queue_url: Optional[str] = None` after `reload: bool = False`; add `queue_url: Redis URL; enables queue mode (see use_queue)` to the docstring Args; at the top of the body (before the host/port resolution) add:
```python
        if queue_url is not None:
            self.use_queue(self._redis_backend(queue_url))
```
   and replace the `Predict endpoint` print with:
```python
        if self._queue_backend is not None:
            print(f"Jobs endpoint: POST http://{host}:{port}/jobs")
        else:
            print(f"Predict endpoint: POST http://{host}:{port}/predict")
```

- [ ] **Step 6: Run the new tests**

Run: `uv run --extra dev pytest tests/test_queue_api.py -q`
Expected: PASS (all 15).

- [ ] **Step 7: Full regression, lint, commit**

Run: `uv run --extra dev pytest -q && uv run --extra dev ruff check src/ && uv run --extra dev mypy src/ --ignore-missing-imports`
Expected: all existing tests (including health/predict tests) still PASS; lint clean. If an existing health test fails because `model_loaded` is now optional, fix the implementation, not the test: sync-mode output must be byte-identical.

```bash
git add src tests/test_queue_api.py
git commit -m "feat(queue): add /jobs API and queue mode in MLController" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Docs, example, spec reconciliation

**Files:**
- Create: `examples/queue_worker.py`
- Modify: `README.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-10-08-redis-job-queue-design.md`

- [ ] **Step 1: Example**

Create `examples/queue_worker.py`:

```python
"""
Queue mode example: one file, started as an API or as a worker.

    pip install fastmlapi[queue]
    docker run --rm -p 6379:6379 redis:7

    python examples/queue_worker.py            # API on :8000 (no model loaded)
    python examples/queue_worker.py worker     # worker (start as many as you like)

    curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
         -d '{"data": {"n": 21}}'               # -> {"job_id": "..."}
    curl localhost:8000/jobs/<job_id>           # queued -> running -> succeeded
"""

import os
import sys
import time

from fastmlapi import MLController, postprocessing, prediction, preprocessing

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class SlowDoubler(MLController):
    model_name = "slow-doubler"

    @preprocessing
    def preprocess(self, data: dict) -> int:
        return data["n"]

    @prediction
    def infer(self, n: int) -> int:
        time.sleep(10)  # stand in for heavy inference
        return n * 2

    @postprocessing
    def postprocess(self, out: int) -> dict:
        return {"doubled": out}


if __name__ == "__main__":
    controller = SlowDoubler()
    if sys.argv[1:] == ["worker"]:
        controller.run_worker(redis_url=REDIS_URL)
    else:
        controller.run(queue_url=REDIS_URL)
```

- [ ] **Step 2: README section**

Add a section "Long-running jobs (queue mode)" to `README.md` after the existing usage sections, covering: install `fastmlapi[queue]`; run API with `run(queue_url=...)` and workers with `run_worker(redis_url=...)` (point to `examples/queue_worker.py`); the endpoint contract (`POST /jobs` -> 202 `{job_id}`; `GET /jobs/{id}` -> `status` `queued|running|succeeded|failed` plus `result`/`error`/`error_type`); that `/predict` is not available in queue mode and the API process does not load the model; tuning attributes `result_ttl`, `visibility_timeout`, `max_attempts`, `heartbeat_interval`; caveats: at-least-once delivery (a job can run twice if a worker dies after finishing but before saving), no hard per-job timeout (Python cannot safely interrupt a running model), a long CPU-bound call that never releases the GIL can delay heartbeats so raise `visibility_timeout`, results must be JSON-serializable, results are deleted after `result_ttl`, workers (not the API) requeue jobs from dead workers.

- [ ] **Step 3: CHANGELOG**

Under `## [Unreleased]` in `CHANGELOG.md` add:

```markdown
### Added

- Queue mode for long-running inference: `run(queue_url=...)` serves `POST /jobs` and `GET /jobs/{id}` and does not load the model; `run_worker(redis_url=...)` runs a worker. Redis-backed (`pip install fastmlapi[queue]`), with heartbeats, crash recovery and graceful shutdown.
- `fastmlapi.queue` package: `QueueBackend`, `MemoryBackend`, `RedisBackend`, `Worker`
- `result_ttl`, `visibility_timeout`, `max_attempts`, `heartbeat_interval` class attributes
- `HealthResponse.queue_connected` / `queue_depth` (queue mode only)

### Changed

- `HealthResponse.model_loaded` is now optional (omitted in queue mode); sync-mode output is unchanged
```

- [ ] **Step 4: Reconcile the spec with what was built**

Edit `docs/superpowers/specs/2026-10-08-redis-job-queue-design.md`:
- Lifecycle step 2 and the Worker section: claiming is a non-blocking atomic Lua script (`RPOP` from `pending` + `LPUSH` to `processing:{worker}` + mark running, in one step) and workers poll every 0.5 s when the queue is empty, replacing `BLMOVE`. This also removes the Redis >= 6.2 requirement.
- Crash recovery: the stale sweep runs in workers only (not the API). Jobs of dead workers are recovered as soon as any worker is running.
- Completion, failure and heartbeat are fenced by worker id (a worker that lost its job cannot overwrite the new owner).
- `POST /jobs` invalid body: 422 from FastAPI body validation (note: today's `/predict` returns 500 for an invalid body; that is not changed).
- `/health` in queue mode: 200 when the backend is reachable and 503 otherwise, with `queue_connected` and `queue_depth`; `model_loaded` omitted.
- Public API: add `use_queue(backend)`.

- [ ] **Step 5: Final verification**

Run: `uv run --extra dev pytest -q && uv run --extra dev ruff check src/ && uv run --extra dev mypy src/ --ignore-missing-imports`
Expected: all PASS/clean. If a Redis server is reachable, also run the suite once with `REDIS_URL=redis://localhost:6379/0` and report the result; otherwise state that Redis-backed tests were not run locally (CI runs them).

Smoke-check the example end to end if Redis is available: start the API and one worker (two terminals), `POST /jobs`, poll `GET /jobs/{id}` until `succeeded`. If Redis is not available, say so.

- [ ] **Step 6: Commit**

```bash
git add README.md CHANGELOG.md examples/queue_worker.py docs/superpowers/specs
git commit -m "docs: document queue mode and add worker example" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

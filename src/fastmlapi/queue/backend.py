"""Queue backend interface shared by the API and the workers."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, List, Optional

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


@dataclasses.dataclass
class WorkerInfo:
    """A live worker, as published by `worker_alive`."""

    worker_id: str
    model_loaded: bool
    started_at: Optional[float] = None
    last_seen: Optional[float] = None


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

    # -- worker liveness (optional: default to doing nothing) ------------------

    def worker_alive(self, worker_id: str, model_loaded: bool, ttl: float) -> None:
        """Publish that this worker is running, and whether its model is loaded.
        The entry disappears by itself after `ttl` seconds unless refreshed, so
        a crashed worker stops being listed."""

    def worker_gone(self, worker_id: str) -> None:
        """Remove this worker's liveness entry (graceful shutdown)."""

    def list_workers(self) -> List[WorkerInfo]:
        """Workers whose liveness entry has not expired."""
        return []

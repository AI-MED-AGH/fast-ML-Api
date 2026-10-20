"""In-process queue backend for local development and tests."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .backend import FAILED, QUEUED, RUNNING, SUCCEEDED, Job, QueueBackend, WorkerInfo


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
        self._workers: Dict[str, Dict[str, Any]] = {}
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
        job.payload = None  # the request body is only needed while the job can still run
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
                        job.payload = None
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

    # -- worker liveness ---------------------------------------------------

    def worker_alive(self, worker_id: str, model_loaded: bool, ttl: float) -> None:
        with self._lock:
            now = self._clock()
            entry = self._workers.get(worker_id)
            started = entry["started_at"] if entry and entry["expires_at"] > now else now
            self._workers[worker_id] = {
                "model_loaded": bool(model_loaded),
                "started_at": started,
                "last_seen": now,
                "expires_at": now + max(1.0, float(ttl)),
            }

    def worker_gone(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def list_workers(self) -> List[WorkerInfo]:
        with self._lock:
            now = self._clock()
            for worker_id in [w for w, e in self._workers.items() if e["expires_at"] <= now]:
                del self._workers[worker_id]
            return [
                WorkerInfo(w, e["model_loaded"], e["started_at"], e["last_seen"])
                for w, e in sorted(self._workers.items())
            ]

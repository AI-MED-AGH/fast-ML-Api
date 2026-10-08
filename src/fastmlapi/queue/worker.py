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
        """Sweep stale jobs if due, then claim and process one job.

        Returns False if the queue was empty.
        """
        self._maybe_sweep()
        job = self.backend.claim(self.worker_id)
        if job is None:
            return False
        self._process(job)
        return True

    # -- internals ---------------------------------------------------------

    def _tick(self) -> bool:
        try:
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
                message = f"Result is not JSON-serializable: {exc}"
                accepted = self._retry(
                    lambda: self.backend.fail(
                        self.worker_id,
                        job.id,
                        message,
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

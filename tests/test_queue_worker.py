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
    job_id = backend.enqueue({"data": {"n": 4}})
    assert worker.run_once() is True
    job = backend.get_job(job_id)
    assert job.status == "succeeded"
    assert job.result == {"doubled": 8}


def test_model_exception_fails_the_job_and_the_worker_keeps_going():
    _, backend, worker = make_worker()
    bad = backend.enqueue({"data": {"n": 0}})
    good = backend.enqueue({"data": {"n": 1}})
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
    job_id = backend.enqueue({"data": {"n": 1}})
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
    controller.job_id = backend.enqueue({"data": {"n": 1}})
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
        ids = [backend.enqueue({"data": {"n": i + 1}}) for i in range(3)]
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
    job_id = backend.enqueue({"data": {"n": 1}})
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
    job_id = backend.enqueue({"data": {"n": 5}})
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
    job_id = backend.enqueue({"data": {"n": 3}})
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


def test_payload_that_fails_request_validation_fails_the_job_and_worker_keeps_going():
    _, backend, worker = make_worker()
    bad = backend.enqueue({"n": 1})  # no `data`: invalid for PredictionRequest
    good = backend.enqueue({"data": {"n": 2}})
    assert worker.run_once() is True
    failed = backend.get_job(bad)
    assert failed.status == "failed"
    assert failed.error_type == "ValidationError"
    assert "data" in failed.error
    assert worker.run_once() is True
    assert backend.get_job(good).result == {"doubled": 4}


def test_heartbeat_thread_survives_an_unexpected_error():
    class Glitchy(MemoryBackend):
        calls = 0
        ok_after_error = 0

        def heartbeat(self, worker_id, job_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("READONLY You can't write against a read only replica")
            self.ok_after_error += 1
            return super().heartbeat(worker_id, job_id)

    class Slow(Doubler):
        heartbeat_interval = 0.02
        backend = None
        job_id = None
        seen = None

        @prediction
        def infer(self, n):
            time.sleep(0.1)  # the first (failing) beat happens in here
            first = self.backend.get_job(self.job_id).heartbeat
            time.sleep(0.2)
            self.seen = (first, self.backend.get_job(self.job_id).heartbeat)
            return n

    controller = Slow()
    backend = Glitchy()
    controller.backend = backend
    controller.job_id = backend.enqueue({"data": {"n": 1}})
    _, _, worker = make_worker(controller, backend)
    worker.run_once()
    assert backend.ok_after_error >= 2
    first, later = controller.seen
    assert later > first
    assert backend.get_job(controller.job_id).status == "succeeded"


def test_heartbeat_keeps_running_while_the_result_write_is_retried():
    class SlowSave(MemoryBackend):
        failures = 3
        saving = False
        beats_while_saving = 0

        def complete(self, worker_id, job_id, result):
            self.saving = True
            if self.failures > 0:
                self.failures -= 1
                raise QueueUnavailable("down")
            return super().complete(worker_id, job_id, result)

        def heartbeat(self, worker_id, job_id):
            if self.saving:
                self.beats_while_saving += 1
            return super().heartbeat(worker_id, job_id)

    class Fast(Doubler):
        heartbeat_interval = 0.01

    backend = SlowSave()
    job_id = backend.enqueue({"data": {"n": 2}})
    # the result write is retried for ~0.05 + 0.1 + 0.2 s
    _, _, worker = make_worker(Fast(), backend, backoff_initial=0.05)
    worker.run_once()
    assert backend.get_job(job_id).status == "succeeded"
    assert backend.beats_while_saving > 0


# -- worker liveness: how an orchestrator learns that a worker has loaded its model ---

class SpyBackend(MemoryBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.alive_calls = []

    def worker_alive(self, worker_id, model_loaded, ttl):
        self.alive_calls.append((model_loaded, ttl))
        super().worker_alive(worker_id, model_loaded, ttl)


def run_in_thread(worker):
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    return t


def test_worker_reports_not_loaded_while_the_model_loads_and_loaded_afterwards():
    backend = SpyBackend(clock=time.time)
    seen = {}

    class SlowStart(Doubler):
        heartbeat_interval = 0.05

        def setup(self):
            seen["during"] = [(w.model_loaded) for w in backend.list_workers()]

    worker = Worker(SlowStart(), backend, poll_interval=0.01)
    t = run_in_thread(worker)
    try:
        assert wait_for(lambda: any(w.model_loaded for w in backend.list_workers()))
        assert seen["during"] == [False]                  # visible as "starting" before the model was ready
        assert backend.alive_calls[0][0] is False and backend.alive_calls[-1][0] is True
    finally:
        worker.stop()
        t.join(5)


def test_liveness_is_refreshed_with_a_ttl_of_three_heartbeats():
    backend = SpyBackend(clock=time.time)

    class Quick(Doubler):
        heartbeat_interval = 0.05

    worker = Worker(Quick(), backend, poll_interval=0.01)
    t = run_in_thread(worker)
    try:
        assert wait_for(lambda: len(backend.alive_calls) >= 4)
        assert all(ttl == pytest.approx(0.15) or ttl >= 1 for _, ttl in backend.alive_calls)
        first = backend.list_workers()[0].last_seen
        time.sleep(0.2)
        assert backend.list_workers()[0].last_seen > first
    finally:
        worker.stop()
        t.join(5)


def test_graceful_stop_removes_the_liveness_entry():
    backend = MemoryBackend(clock=time.time)
    worker = Worker(Doubler(), backend, poll_interval=0.01)
    t = run_in_thread(worker)
    assert wait_for(lambda: backend.list_workers() != [])
    worker.stop()
    t.join(5)
    assert backend.list_workers() == []


def test_failed_startup_leaves_no_liveness_entry():
    backend = MemoryBackend(clock=time.time)

    class Broken(Doubler):
        def load_model(self):
            raise RuntimeError("no weights")

    worker = Worker(Broken(), backend, poll_interval=0.01)
    with pytest.raises(RuntimeError):
        worker.run()
    assert backend.list_workers() == []


def test_an_outage_while_publishing_liveness_does_not_stop_the_worker():
    class Flaky(MemoryBackend):
        def worker_alive(self, worker_id, model_loaded, ttl):
            raise QueueUnavailable("redis down")

    backend = Flaky()
    worker = Worker(Doubler(), backend, poll_interval=0.01)
    job_id = backend.enqueue({"data": {"n": 3}})
    t = run_in_thread(worker)
    try:
        assert wait_for(lambda: backend.get_job(job_id).status == "succeeded")
    finally:
        worker.stop()
        t.join(5)

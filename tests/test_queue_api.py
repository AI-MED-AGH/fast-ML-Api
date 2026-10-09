"""API and controller integration for queue mode."""

from datetime import datetime

import uvicorn
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

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


# -- same model input as /predict (sync and queue mode agree) ---------------


class Inner(BaseModel):
    n: int


class NestedRequest(BaseModel):
    data: Inner


class NestedDoubler(MLController):
    model_name = "nested-doubler"
    request_model = NestedRequest
    seen = []

    @preprocessing
    def preprocess(self, data):
        type(self).seen.append(data)
        return data.n

    @prediction
    def infer(self, n):
        return n * 2


class WhenRequest(BaseModel):
    when: datetime
    n: int


class WhenYear(MLController):
    model_name = "when-year"
    request_model = WhenRequest
    seen = []

    @preprocessing
    def preprocess(self, data):
        type(self).seen.append(data)
        return data["when"].year + data["n"]

    @prediction
    def infer(self, x):
        return x


def _run_job(controller_cls, body):
    backend = MemoryBackend()
    api = controller_cls()
    api.use_queue(backend)
    worker_ctrl = controller_cls()
    worker_ctrl._initialize()
    worker = Worker(worker_ctrl, backend, poll_interval=0.01)
    with TestClient(api.app) as client:
        r = client.post("/jobs", json=body)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        assert worker.run_once() is True
        return client.get(f"/jobs/{job_id}").json()


def _run_sync(controller_cls, body):
    with TestClient(controller_cls().app) as client:
        r = client.post("/predict", json=body)
        assert r.status_code == 200, r.text
        return r.json()["prediction"]


def test_nested_data_model_reaches_preprocess_as_the_model_instance():
    NestedDoubler.seen = []
    job = _run_job(NestedDoubler, {"data": {"n": 3}})
    assert job["status"] == "succeeded", job
    assert job["result"] == {"result": 6}
    assert isinstance(NestedDoubler.seen[-1], Inner)
    # and it matches what sync /predict hands preprocess
    assert _run_sync(NestedDoubler, {"data": {"n": 3}}) == {"result": 6}
    assert isinstance(NestedDoubler.seen[-1], Inner)


def test_request_model_without_data_gets_real_python_types_in_the_worker():
    WhenYear.seen = []
    body = {"when": "2024-05-01T12:00:00", "n": 1}
    job = _run_job(WhenYear, body)
    assert job["status"] == "succeeded", job
    assert job["result"] == {"result": 2025}
    assert isinstance(WhenYear.seen[-1]["when"], datetime)
    assert _run_sync(WhenYear, body) == {"result": 2025}
    assert WhenYear.seen[0] == WhenYear.seen[1]

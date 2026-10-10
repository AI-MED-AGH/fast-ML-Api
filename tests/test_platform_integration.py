"""What an orchestrating platform needs from fastmlapi: roles from the environment, honest validation errors."""

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from fastmlapi import MLController, postprocessing, preprocessing
from fastmlapi.queue.memory_backend import MemoryBackend


class Scorer(MLController):
    model_name = "scorer"

    def load_model(self):
        return lambda xs: sum(xs)

    @preprocessing
    def preprocess(self, data):
        return data["values"]

    @postprocessing
    def postprocess(self, out):
        return {"score": out}


class Typed(Scorer):
    class Request(BaseModel):
        values: list[float] = Field(..., min_length=1)

    request_model = Request


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("FASTMLAPI_ROLE", "FASTMLAPI_QUEUE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def spy(monkeypatch):
    """Replaces the things run() would start for real."""
    calls = {}

    def fake_uvicorn_run(app, **kw):
        calls["app"] = app
        calls["uvicorn"] = kw

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr(MLController, "run_worker",
                        lambda self, redis_url=None, **kw: calls.setdefault("worker", redis_url))
    monkeypatch.setattr(MLController, "_redis_backend",
                        lambda self, url: calls.setdefault("backend_url", url) and MemoryBackend())
    return calls


# -- 1. the role and the queue URL come from the environment ------------------------

def test_role_worker_runs_a_worker_with_the_url_from_the_environment(spy, monkeypatch):
    monkeypatch.setenv("FASTMLAPI_ROLE", "worker")
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://m_scorer:pw@queue-redis:6379/0")
    Scorer().run()
    assert spy["worker"] == "redis://m_scorer:pw@queue-redis:6379/0"
    assert "uvicorn" not in spy                           # a worker serves no HTTP


def test_role_api_serves_jobs_instead_of_predict(spy, monkeypatch):
    monkeypatch.setenv("FASTMLAPI_ROLE", "api")
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://queue:6379/0")
    Scorer().run()
    assert spy["backend_url"] == "redis://queue:6379/0"
    paths = {r.path for r in spy["app"].routes}
    assert "/jobs" in paths and "/predict" not in paths


def test_role_sync_and_unset_ignore_the_queue_url(spy, monkeypatch):
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://queue:6379/0")
    Scorer().run()                                        # no role
    monkeypatch.setenv("FASTMLAPI_ROLE", "sync")
    Scorer().run()
    assert "backend_url" not in spy and "worker" not in spy
    assert "/predict" in {r.path for r in spy["app"].routes}


def test_an_explicit_queue_url_argument_wins_over_the_environment(spy, monkeypatch):
    monkeypatch.setenv("FASTMLAPI_ROLE", "api")
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://from-env:6379/0")
    Scorer().run(queue_url="redis://explicit:6379/0")
    assert spy["backend_url"] == "redis://explicit:6379/0"


def test_run_worker_without_arguments_uses_the_environment(spy, monkeypatch):
    monkeypatch.undo()                                    # use the real run_worker, but a fake Worker
    seen = {}

    class FakeWorker:
        def __init__(self, controller, backend, **kw):
            seen["backend"] = backend

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr("fastmlapi.queue.worker.Worker", FakeWorker)
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://queue:6379/0")
    monkeypatch.setattr(MLController, "_redis_backend", lambda self, url: seen.setdefault("url", url) and MemoryBackend())
    Scorer().run_worker()
    assert seen["url"] == "redis://queue:6379/0" and seen["ran"] is True


@pytest.mark.parametrize("role", ["master", "WORKER ", "api,worker", "1"])
def test_unknown_roles_are_rejected_with_the_valid_ones_listed(spy, monkeypatch, role):
    monkeypatch.setenv("FASTMLAPI_ROLE", role)
    with pytest.raises(ValueError, match="sync.*api.*worker"):
        Scorer().run()


@pytest.mark.parametrize("role", ["api", "worker"])
def test_queue_roles_without_a_queue_url_fail_loudly(spy, monkeypatch, role):
    monkeypatch.setenv("FASTMLAPI_ROLE", role)
    with pytest.raises(ValueError, match="FASTMLAPI_QUEUE_URL"):
        Scorer().run()


def test_the_app_property_honours_the_api_role_for_uvicorn_main_app(spy, monkeypatch):
    monkeypatch.setenv("FASTMLAPI_ROLE", "api")
    monkeypatch.setenv("FASTMLAPI_QUEUE_URL", "redis://queue:6379/0")
    paths = {r.path for r in Scorer().app.routes}
    assert "/jobs" in paths and "/predict" not in paths


# -- 4. a client's mistake is a 422, not a 500 ---------------------------------------

@pytest.fixture
def typed_client():
    with TestClient(Typed().app) as c:
        yield c


def test_predict_with_invalid_input_is_422_and_says_what_is_wrong(typed_client):
    r = typed_client.post("/predict", json={"values": "not a list"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error_type"] == "ValidationError"
    assert detail["errors"][0]["loc"] == ["values"] and detail["errors"][0]["msg"]


def test_the_422_does_not_echo_the_submitted_data_back(typed_client):
    r = typed_client.post("/predict", json={"values": "patient-name-12345"})
    assert r.status_code == 422 and "patient-name-12345" not in r.text


def test_predict_with_a_missing_field_or_empty_body_is_422(typed_client):
    assert typed_client.post("/predict", json={}).status_code == 422
    assert typed_client.post("/predict", json={"values": []}).status_code == 422


def test_predict_with_malformed_json_is_422(typed_client):
    r = typed_client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["detail"]["error_type"] == "ValidationError"


def test_the_default_request_shape_requires_data():
    with TestClient(Scorer().app) as c:
        assert c.post("/predict", json={"values": [1]}).status_code == 422
        assert c.post("/predict", json={"data": {"values": [1, 2]}}).json()["prediction"] == {"score": 3}


def test_a_bug_in_the_model_code_is_still_a_500(typed_client):
    class Bug(Typed):
        @preprocessing
        def preprocess(self, data):
            raise KeyError("oops")

    with TestClient(Bug().app) as c:
        r = c.post("/predict", json={"values": [1.0]})
    assert r.status_code == 500 and r.json()["detail"]["error_type"] == "KeyError"


# -- queue health shows whether workers are ready -----------------------------------

def test_queue_health_reports_workers_and_how_many_have_loaded_the_model():
    backend = MemoryBackend()
    controller = Scorer()
    controller.use_queue(backend)
    with TestClient(controller.app) as c:
        h = c.get("/health").json()
        assert h["queue_workers"] == 0 and h["queue_workers_ready"] == 0
        backend.worker_alive("w1", model_loaded=False, ttl=60)
        backend.worker_alive("w2", model_loaded=True, ttl=60)
        h = c.get("/health").json()
        assert h["queue_workers"] == 2 and h["queue_workers_ready"] == 1

"""A cap on queued jobs, and control over how much of an exception reaches the caller."""

import pytest
from fastapi.testclient import TestClient

from fastmlapi import MLController, postprocessing, prediction, preprocessing
from fastmlapi.queue.memory_backend import MemoryBackend
from fastmlapi.queue.worker import Worker

SECRET_PATH = "/srv/weights/model.pkl"


class Leaky(MLController):
    model_name = "leaky"

    def load_model(self):
        return None

    @preprocessing
    def preprocess(self, data):
        return data["n"]

    @prediction
    def infer(self, n):
        if n == 0:
            raise FileNotFoundError(f"cannot open {SECRET_PATH}")
        return n

    @postprocessing
    def postprocess(self, out):
        return {"n": out}


def queue_client(monkeypatch, *, max_pending=None, expose=None):
    if max_pending is not None:
        monkeypatch.setenv("FASTMLAPI_MAX_PENDING", str(max_pending))
    if expose is not None:
        monkeypatch.setenv("FASTMLAPI_EXPOSE_ERRORS", expose)
    backend = MemoryBackend()
    api = Leaky()
    api.use_queue(backend)
    return api, backend


def post_job(client, n=1):
    return client.post("/jobs", json={"data": {"n": n}})


# ── queued jobs are capped ──


def test_submissions_are_refused_once_the_queue_holds_the_maximum(monkeypatch):
    api, backend = queue_client(monkeypatch, max_pending=2)
    with TestClient(api.app) as c:
        assert post_job(c).status_code == 202
        assert post_job(c).status_code == 202
        full = post_job(c)
    assert full.status_code == 429
    assert full.json()["detail"]["error_type"] == "QueueFull"
    assert int(full.headers["Retry-After"]) >= 1
    assert backend.pending_count() == 2  # the refused job was not stored


def test_room_returns_as_workers_take_jobs(monkeypatch):
    api, backend = queue_client(monkeypatch, max_pending=1)
    worker_ctrl = Leaky()
    worker_ctrl._initialize()
    worker = Worker(worker_ctrl, backend, poll_interval=0.01)
    with TestClient(api.app) as c:
        assert post_job(c).status_code == 202
        assert post_job(c).status_code == 429
        worker.run_once()
        assert post_job(c).status_code == 202


def test_zero_means_no_cap(monkeypatch):
    api, backend = queue_client(monkeypatch, max_pending=0)
    with TestClient(api.app) as c:
        assert all(post_job(c).status_code == 202 for _ in range(20))


def test_the_default_cap_is_generous_but_finite(monkeypatch):
    monkeypatch.delenv("FASTMLAPI_MAX_PENDING", raising=False)
    api, _ = queue_client(monkeypatch)
    assert api.max_pending == 1000


@pytest.mark.parametrize("bad", ["lots", "-5", "1.5"])
def test_a_bad_cap_in_the_environment_stops_the_server_from_starting(monkeypatch, bad):
    api, _ = queue_client(monkeypatch, max_pending=bad)
    with pytest.raises(ValueError, match="FASTMLAPI_MAX_PENDING"):
        with TestClient(api.app):
            pass


# ── how much of an exception callers see ──


def run_failing_job(monkeypatch, expose):
    api, backend = queue_client(monkeypatch, expose=expose)
    worker_ctrl = Leaky()
    worker_ctrl._initialize()
    worker = Worker(worker_ctrl, backend, poll_interval=0.01)
    with TestClient(api.app) as c:
        job_id = post_job(c, 0).json()["job_id"]
        worker.run_once()
        return c.get(f"/jobs/{job_id}").json()


def test_by_default_a_library_user_sees_the_whole_error(monkeypatch):
    monkeypatch.delenv("FASTMLAPI_EXPOSE_ERRORS", raising=False)
    body = run_failing_job(monkeypatch, expose=None)
    assert SECRET_PATH in body["error"] and body["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize("off", ["false", "0", "no", "FALSE"])
def test_the_platform_turns_the_message_off_and_keeps_the_kind_of_failure(monkeypatch, off):
    body = run_failing_job(monkeypatch, expose=off)
    assert body["status"] == "failed"
    assert SECRET_PATH not in str(body)
    assert body["error"] == "The job failed"
    assert body["error_type"] == "FileNotFoundError"


def test_a_500_from_predict_hides_the_message_too(monkeypatch):
    monkeypatch.setenv("FASTMLAPI_EXPOSE_ERRORS", "false")
    with TestClient(Leaky().app) as c:
        r = c.post("/predict", json={"data": {"n": 0}})
    assert r.status_code == 500
    assert SECRET_PATH not in r.text
    assert r.json()["detail"] == {"error": "Prediction failed", "error_type": "FileNotFoundError"}


def test_a_500_from_predict_shows_the_message_when_asked(monkeypatch):
    monkeypatch.setenv("FASTMLAPI_EXPOSE_ERRORS", "true")
    with TestClient(Leaky().app) as c:
        r = c.post("/predict", json={"data": {"n": 0}})
    assert SECRET_PATH in r.json()["detail"]["error"]


def test_an_unexpected_error_outside_predict_is_also_hidden(monkeypatch):
    monkeypatch.setenv("FASTMLAPI_EXPOSE_ERRORS", "false")

    class Broken(Leaky):
        def health_check(self):
            raise RuntimeError(f"disk {SECRET_PATH} unreadable")

    with TestClient(Broken().app, raise_server_exceptions=False) as c:
        r = c.get("/health")
    assert r.status_code == 500 and SECRET_PATH not in r.text

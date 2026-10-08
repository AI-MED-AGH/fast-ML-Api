"""
Tests for FastMLAPI.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import fastmlapi
from fastmlapi import MLController, prediction, preprocessing, postprocessing


class MockClassifier(MLController):
    """Mock classifier for testing."""
    
    model_name = "test-classifier"
    model_version = "1.0.0"
    
    def load_model(self):
        return lambda x: np.array([1])
    
    @preprocessing
    def preprocess(self, data: dict) -> np.ndarray:
        return np.array(data.get("features", [0])).reshape(1, -1)
    
    @postprocessing
    def postprocess(self, prediction: np.ndarray) -> dict:
        return {"class": int(prediction[0])}


@pytest.fixture
def client():
    """Create a test client with lifespan context."""
    controller = MockClassifier()
    # Use context manager to trigger lifespan events
    with TestClient(controller.app) as client:
        yield client


class TestMLController:
    """Tests for MLController."""
    
    def test_info_endpoint(self, client):
        """Test the info endpoint returns model info."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-classifier"
        assert data["version"] == "1.0.0"
        assert data["endpoint"] == "/predict"
    
    def test_health_endpoint(self, client):
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "test-classifier"
    
    def test_predict_endpoint(self, client):
        """Test the predict endpoint."""
        response = client.post(
            "/predict",
            json={"data": {"features": [1, 2, 3]}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert data["prediction"]["class"] == 1
    
    def test_predict_with_metadata(self, client):
        """Test that prediction includes metadata."""
        response = client.post(
            "/predict",
            json={"data": {"features": [1, 2, 3]}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["model_name"] == "test-classifier"
        assert data["metadata"]["model_version"] == "1.0.0"
    
    def test_controller_initialization(self):
        """Test controller initializes correctly."""
        controller = MockClassifier()
        controller.setup = MagicMock()

        controller._initialize()
        assert controller.is_loaded is True
        controller.setup.assert_called()

    def test_controller_not_loaded_error(self):
        """Test accessing model before loading raises error."""
        controller = MockClassifier()
        with pytest.raises(RuntimeError):
            _ = controller.model
    
    def test_health_check(self):
        """Test health check method."""
        controller = MockClassifier()
        controller._initialize()
        health = controller.health_check()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True


class TestDecorators:
    """Tests for preprocessing/postprocessing decorators."""
    
    def test_preprocessing_discovered(self):
        """Test that preprocessing function is discovered."""
        controller = MockClassifier()
        assert controller._preprocessing_fn is not None
    
    def test_postprocessing_discovered(self):
        """Test that postprocessing function is discovered."""
        controller = MockClassifier()
        assert controller._postprocessing_fn is not None


class Typed(BaseModel):
    features: List[float] = Field(..., description="Feature vector")


class TypedClassifier(MockClassifier):
    request_model = Typed
    request_example = {"features": [1.0, 2.0]}
    response_example = {"success": True}


class TestHealthSemantics:
    def test_health_503_before_load(self):
        # No lifespan: the model has not been loaded yet.
        client = TestClient(MockClassifier().app)
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["model_loaded"] is False
        assert set(body) == {"status", "model_loaded", "model_name", "version"}

    def test_health_503_for_prediction_only_before_startup(self):
        class PredOnly(MLController):
            @prediction
            def run(self, data):
                return data

        client = TestClient(PredOnly().app)
        assert client.get("/health").status_code == 503
        with TestClient(PredOnly().app) as ready:
            assert ready.get("/health").status_code == 200

    def test_health_200_after_load(self, client):
        assert client.get("/health").status_code == 200

    def test_livez_ok_even_when_not_ready(self, client):
        not_ready = TestClient(MockClassifier().app)
        assert not_ready.get("/livez").status_code == 200
        assert client.get("/livez").json() == {"status": "alive"}


class TestSchema:
    def test_schema_default_models(self, client):
        data = client.get("/schema").json()
        assert data["model_name"] == "test-classifier"
        assert data["model_version"] == "1.0.0"
        assert "data" in data["request_schema"]["properties"]
        assert "prediction" in data["response_schema"]["properties"]
        assert data["example"] == {"request": None, "response": None}

    def test_schema_typed_request_model(self):
        with TestClient(TypedClassifier().app) as c:
            data = c.get("/schema").json()
        assert "features" in data["request_schema"]["properties"]
        assert data["example"]["request"] == {"features": [1.0, 2.0]}
        assert data["example"]["response"] == {"success": True}

    def test_warning_without_request_model(self, caplog):
        with caplog.at_level(logging.WARNING, logger="fastmlapi.controller"):
            with TestClient(MockClassifier().app):
                pass
        assert any("request_model is not set" in r.message for r in caplog.records)

    def test_no_warning_with_request_model(self, caplog):
        with caplog.at_level(logging.WARNING, logger="fastmlapi.controller"):
            with TestClient(TypedClassifier().app):
                pass
        assert not any("request_model is not set" in r.message for r in caplog.records)


class TestInfo:
    def test_info_new_fields(self, client):
        data = client.get("/info").json()
        assert data["library_version"] == fastmlapi.__version__
        assert isinstance(data["gpu_available"], bool)
        assert data["endpoint"] == "/predict"


class TestCacheDir:
    def test_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FASTMLAPI_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        c = MockClassifier()
        assert isinstance(c.cache_dir, Path)
        assert c.cache_dir == Path("./.cache/fastmlapi")
        c._initialize()
        assert (tmp_path / ".cache" / "fastmlapi").is_dir()

    def test_env_and_available_in_setup_and_load(self, monkeypatch, tmp_path):
        target = tmp_path / "nested" / "cache"
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(target))
        seen = {}

        class C(MockClassifier):
            def setup(self):
                seen["setup"] = self.cache_dir.is_dir()

            def load_model(self):
                seen["load"] = self.cache_dir.is_dir()
                return super().load_model()

        c = C()
        assert c.cache_dir == target
        c._initialize()
        assert seen == {"setup": True, "load": True}


class _FakeCuda:
    def __init__(self, available):
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def set_per_process_memory_fraction(self, f):
        self.calls.append(f)


def _fake_torch(available):
    return SimpleNamespace(cuda=_FakeCuda(available))


class TestGpuFraction:
    def test_noop_without_torch(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(tmp_path))
        monkeypatch.setitem(sys.modules, "torch", None)  # makes import fail
        c = MockClassifier()
        c.gpu_memory_fraction = 0.5
        with caplog.at_level(logging.INFO, logger="fastmlapi.controller"):
            c._initialize()
        assert c.is_loaded
        assert any("torch is not installed" in r.message for r in caplog.records)
        assert c.gpu_available is False

    def test_noop_without_cuda(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(tmp_path))
        torch = _fake_torch(False)
        monkeypatch.setitem(sys.modules, "torch", torch)
        c = MockClassifier()
        c.gpu_memory_fraction = 0.5
        c._initialize()
        assert torch.cuda.calls == []

    def test_applied_before_load_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(tmp_path))
        torch = _fake_torch(True)
        monkeypatch.setitem(sys.modules, "torch", torch)
        order = []

        class C(MockClassifier):
            gpu_memory_fraction = 0.25

            def load_model(self):
                order.append(("load", list(torch.cuda.calls)))
                return super().load_model()

        C()._initialize()
        assert order == [("load", [0.25])]

    def test_env_overrides_attribute(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("FASTMLAPI_GPU_MEMORY_FRACTION", "0.75")
        torch = _fake_torch(True)
        monkeypatch.setitem(sys.modules, "torch", torch)
        c = MockClassifier()
        c.gpu_memory_fraction = 0.25
        c._initialize()
        assert torch.cuda.calls == [0.75]

    @pytest.mark.parametrize("bad", ["0", "1.5", "-1", "abc"])
    def test_invalid_fraction_raises(self, monkeypatch, tmp_path, bad):
        monkeypatch.setenv("FASTMLAPI_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("FASTMLAPI_GPU_MEMORY_FRACTION", bad)
        with pytest.raises(ValueError):
            MockClassifier()._initialize()


class TestRunEnv:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(kw))
        return calls

    def test_defaults(self, monkeypatch, captured):
        monkeypatch.delenv("FASTMLAPI_HOST", raising=False)
        monkeypatch.delenv("FASTMLAPI_PORT", raising=False)
        MockClassifier().run()
        assert (captured["host"], captured["port"]) == ("0.0.0.0", 8000)

    def test_env(self, monkeypatch, captured):
        monkeypatch.setenv("FASTMLAPI_HOST", "127.0.0.1")
        monkeypatch.setenv("FASTMLAPI_PORT", "9123")
        MockClassifier().run()
        assert (captured["host"], captured["port"]) == ("127.0.0.1", 9123)

    def test_explicit_args_win(self, monkeypatch, captured):
        monkeypatch.setenv("FASTMLAPI_HOST", "127.0.0.1")
        monkeypatch.setenv("FASTMLAPI_PORT", "9123")
        MockClassifier().run(host="localhost", port=1234)
        assert (captured["host"], captured["port"]) == ("localhost", 1234)

    def test_invalid_port(self, monkeypatch, captured):
        monkeypatch.setenv("FASTMLAPI_PORT", "nope")
        with pytest.raises(ValueError):
            MockClassifier().run()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_graceful_shutdown(tmp_path):
    """SIGTERM must run the lifespan shutdown (uvicorn graceful stop)."""
    script = tmp_path / "srv.py"
    script.write_text(
        "import logging, sys\n"
        "logging.basicConfig(level=logging.INFO, stream=sys.stderr)\n"
        "from fastmlapi import MLController\n"
        "class S(MLController):\n"
        "    model_name = 'sig'\n"
        "    def load_model(self):\n"
        "        return lambda x: x\n"
        "S().run()\n"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, FASTMLAPI_HOST="127.0.0.1", FASTMLAPI_PORT=str(port),
               FASTMLAPI_CACHE_DIR=str(tmp_path / "cache"))
    proc = subprocess.Popen([sys.executable, str(script)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(url, timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "Shutting down sig" in out

"""
Tests for FastMLAPI.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from fastmlapi import MLController, preprocessing, postprocessing


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
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-classifier"
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
        controller.initialize()
        assert controller.is_loaded is True
    
    def test_controller_not_loaded_error(self):
        """Test accessing model before loading raises error."""
        controller = MockClassifier()
        with pytest.raises(RuntimeError):
            _ = controller.model
    
    def test_health_check(self):
        """Test health check method."""
        controller = MockClassifier()
        controller.initialize()
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

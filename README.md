# FastMLAPI

A FastAPI-based ML model serving library for easy deployment. Create production-ready ML APIs with minimal boilerplate.

## Features

- 🚀 **Simple API**: Extend `MLController` and implement a few methods
- 🔄 **Automatic `/predict` endpoint**: Generated automatically with proper request/response handling
- 🎯 **Flexible prediction**: Use `load_model()` for standard models or `@prediction` decorator for custom logic
- 🔧 **Data pipelines**: `@preprocessing` and `@postprocessing` decorators for clean data flow
- �️ **Custom endpoints**: Add additional routes with `@route` decorator
- �📊 **Health checks**: Built-in `/health` endpoint
- 📝 **Auto documentation**: Swagger/OpenAPI docs out of the box
- 🎨 **Customizable**: Custom request/response Pydantic models supported

## Installation

```bash
pip install fastmlapi
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

### Option 1: Using `load_model()` (Recommended for sklearn, XGBoost, etc.)

Best for models that have a standard `.predict()` method:

```python
from fastmlapi import MLController, preprocessing, postprocessing
import joblib
import numpy as np

class MyClassifier(MLController):
    model_name = "my-classifier"
    model_version = "1.0.0"
    
    def load_model(self):
        """Load and return your model. Called once at startup."""
        return joblib.load("model.pkl")
    
    @preprocessing
    def preprocess(self, data: dict) -> np.ndarray:
        """Transform input data for the model."""
        features = data.get("features", [])
        return np.array(features).reshape(1, -1)
    
    @postprocessing  
    def postprocess(self, prediction: np.ndarray) -> dict:
        """Format model output for API response."""
        return {
            "class": int(prediction[0]),
            "label": "positive" if prediction[0] == 1 else "negative"
        }

if __name__ == "__main__":
    MyClassifier().run()
```

### Option 2: Using `@prediction` Decorator (Recommended for custom inference)

Best for PyTorch, TensorFlow, or any custom prediction logic:

```python
from fastmlapi import MLController, prediction, postprocessing
import torch

class PyTorchModel(MLController):
    model_name = "pytorch-classifier"
    model_version = "1.0.0"
    
    def load_model(self):
        """Load PyTorch model."""
        model = torch.load("model.pt")
        model.eval()
        return model
    
    @prediction
    def run_inference(self, data: dict) -> torch.Tensor:
        """Custom prediction logic - replaces the default predict behavior."""
        with torch.no_grad():
            tensor = torch.tensor(data["features"], dtype=torch.float32)
            return self.model(tensor)
    
    @postprocessing
    def postprocess(self, output: torch.Tensor) -> dict:
        """Convert tensor to JSON-serializable format."""
        probabilities = torch.softmax(output, dim=-1)
        return {
            "class": int(torch.argmax(probabilities)),
            "confidence": float(probabilities.max())
        }

if __name__ == "__main__":
    PyTorchModel().run()
```

### Option 3: No Model Needed (External APIs, rule-based systems)

When you don't need to load a model at all:

```python
from fastmlapi import MLController, prediction
import requests

class ExternalAPIController(MLController):
    model_name = "external-predictor"
    
    # No load_model() needed!
    
    @prediction
    def call_external_service(self, data: dict) -> dict:
        """Call an external ML service."""
        response = requests.post(
            "https://api.example.com/predict",
            json=data,
            timeout=30
        )
        return response.json()

if __name__ == "__main__":
    ExternalAPIController().run()
```

## How It Works

### Prediction Pipeline

```
Request → preprocess() → predict_raw() → postprocess() → Response
                              ↑
                    Uses @prediction method
                    OR model.predict()
```

1. **`preprocess(data)`**: Transform raw input into model-ready format
2. **`predict_raw(preprocessed_data)`**: Run the actual prediction
   - If `@prediction` decorator is used → calls your decorated method
   - Otherwise → calls `self.model.predict()`
3. **`postprocess(prediction)`**: Format output for the API response

### Decorators

| Decorator | Purpose | Required? |
|-----------|---------|-----------|
| `@preprocessing` | Mark a method as the preprocessing step | No (defaults to pass-through) |
| `@postprocessing` | Mark a method as the postprocessing step | No (defaults to `{"result": prediction}`) |
| `@prediction` | Mark a method as the custom prediction function | No (uses `model.predict()` by default) |
| `@route` | Add a custom API endpoint | No |

## Running the Server

### Direct execution

```python
if __name__ == "__main__":
    MyClassifier().run(host="0.0.0.0", port=8000)
```

### With Uvicorn (for development with auto-reload)

```python
# main.py
classifier = MyClassifier()
app = classifier.app
```

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

Once running, your API provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Run predictions |
| `/health` | GET | Health check status |
| `/` | GET | API info |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"features": [1.0, 2.0, 3.0]}}'
```

### Example Response

```json
{
  "success": true,
  "prediction": {
    "class": 1,
    "label": "positive"
  },
  "metadata": {
    "model_name": "my-classifier",
    "model_version": "1.0.0"
  }
}
```

## Advanced Usage

### Custom Endpoints with `@route` Decorator

Add additional API endpoints beyond `/predict` using the `@route` decorator:

```python
from fastmlapi import MLController, preprocessing, postprocessing, route
from fastapi import Request
from typing import List

class MyController(MLController):
    model_name = "my-model"
    
    def load_model(self):
        return joblib.load("model.pkl")
    
    @preprocessing
    def preprocess(self, data: dict):
        return np.array(data["features"]).reshape(1, -1)
    
    @postprocessing
    def postprocess(self, prediction):
        return {"class": int(prediction[0])}
    
    # Custom endpoint for batch predictions
    @route("/batch", methods=["POST"], tags=["Batch"], summary="Batch prediction")
    async def batch_predict(self, request: Request):
        """Process multiple predictions in one request."""
        data = await request.json()
        results = []
        for item in data["items"]:
            result = await self.predict(item)
            results.append(result)
        return {"results": results, "count": len(results)}
    
    # Custom endpoint for model info
    @route("/model-info", methods=["GET"], tags=["Info"])
    async def model_info(self):
        """Get detailed model information."""
        return {
            "name": self.model_name,
            "version": self.model_version,
            "features": ["feature1", "feature2", "feature3"],
            "classes": ["negative", "positive"]
        }
    
    # Custom endpoint with path parameters
    @route("/predict/{category}", methods=["POST"], tags=["Prediction"])
    async def predict_by_category(self, request: Request, category: str):
        """Run prediction for a specific category."""
        data = await request.json()
        data["category"] = category
        return await self.predict(data)

if __name__ == "__main__":
    MyController().run()
```

#### `@route` Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | (required) | URL path (e.g., `/batch`, `/analyze`) |
| `methods` | list | `["GET"]` | HTTP methods (`["GET"]`, `["POST"]`, etc.) |
| `response_model` | BaseModel | `None` | Pydantic model for response validation |
| `tags` | list | `["Custom"]` | OpenAPI tags for documentation |
| `summary` | str | `None` | Short description for docs |
| `description` | str | `None` | Detailed description for docs |

### Alternative: Using `add_route()` Method

You can also add routes programmatically in `__init__`:

```python
class MyController(MLController):
    def __init__(self):
        super().__init__()
        self.add_route("/status", self.get_status, methods=["GET"])
        self.add_route("/batch", self.batch_predict, methods=["POST"])
    
    async def get_status(self):
        return {"status": "ready", "model_loaded": self.is_loaded}
    
    async def batch_predict(self, request: Request):
        data = await request.json()
        return {"results": [await self.predict(item) for item in data["items"]]}
```

### Custom Request/Response Models

Define your own Pydantic models for type-safe requests and responses:

```python
from pydantic import BaseModel, Field
from typing import List

class ImageRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze")
    threshold: float = Field(0.5, ge=0, le=1, description="Detection confidence threshold")

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class ImageResponse(BaseModel):
    objects: List[DetectedObject]
    count: int

class ObjectDetector(MLController):
    model_name = "object-detector"
    request_model = ImageRequest    # Custom request schema
    response_model = ImageResponse  # Custom response schema
    
    def load_model(self):
        return load_yolo_model()
    
    @preprocessing
    def preprocess(self, data: dict):
        # data contains: {"image_url": "...", "threshold": 0.5}
        image = download_image(data["image_url"])
        self.threshold = data["threshold"]
        return image
    
    @prediction
    def detect(self, image):
        detections = self.model(image)
        return [d for d in detections if d.confidence >= self.threshold]
    
    @postprocessing
    def postprocess(self, detections) -> dict:
        return {
            "objects": [
                {"label": d.label, "confidence": d.conf, "bbox": d.bbox}
                for d in detections
            ],
            "count": len(detections)
        }
```

### Accessing the Model in @prediction Methods

When using `@prediction`, you can still access the loaded model via `self.model`:

```python
class HybridController(MLController):
    def load_model(self):
        return {"encoder": load_encoder(), "classifier": load_classifier()}
    
    @prediction
    def predict(self, data: dict):
        # Access multiple models
        encoded = self.model["encoder"].transform(data["text"])
        return self.model["classifier"].predict(encoded)
```

### TensorFlow/Keras Example

```python
from fastmlapi import MLController, preprocessing, postprocessing
import tensorflow as tf
import numpy as np

class KerasClassifier(MLController):
    model_name = "keras-classifier"
    
    def load_model(self):
        return tf.keras.models.load_model("model.h5")
    
    @preprocessing
    def preprocess(self, data: dict) -> np.ndarray:
        return np.array(data["features"]).reshape(1, -1)
    
    @postprocessing
    def postprocess(self, prediction: np.ndarray) -> dict:
        class_idx = int(np.argmax(prediction[0]))
        return {
            "class": class_idx,
            "probabilities": prediction[0].tolist()
        }
```

## Configuration Reference

### MLController Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"ml-model"` | Name of your model |
| `model_version` | str | `"1.0.0"` | Model version string |
| `title` | str | `"FastMLAPI"` | API title (shown in docs) |
| `description` | str | `"ML Model Serving API"` | API description |
| `api_version` | str | `"1.0.0"` | API version |
| `request_model` | BaseModel | `PredictionRequest` | Custom Pydantic request model |
| `response_model` | BaseModel | `PredictionResponse` | Custom Pydantic response model |
| `enable_health` | bool | `True` | Enable `/health` endpoint |
| `enable_docs` | bool | `True` | Enable Swagger/OpenAPI docs |

### run() Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | `"0.0.0.0"` | Host to bind to |
| `port` | int | `8000` | Port to bind to |
| `reload` | bool | `False` | Enable auto-reload (dev only) |
| `**uvicorn_kwargs` | dict | `{}` | Additional Uvicorn options |

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/fastmlapi.git
cd fastmlapi

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run example
python examples/simple_classifier.py
```

## License

MIT License

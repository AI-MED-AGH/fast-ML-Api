# FastMLAPI

A FastAPI-based ML worker node library for easy model deployment. Create production-ready ML APIs with minimal boilerplate.

## Features

- 🚀 **Simple API**: Extend `MLController` and implement a few methods
- 🔄 **Automatic `/predict` endpoint**: Generated automatically with proper request/response handling
- 🎯 **Preprocessing/Postprocessing**: Decorators for clean data pipeline
- 📊 **Health checks**: Built-in `/health` endpoint
- 📝 **Auto documentation**: Swagger/OpenAPI docs out of the box
- 🔧 **Customizable**: Custom request/response models supported

## Installation

```bash
pip install fastmlapi
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from fastmlapi import MLController, preprocessing, postprocessing
import numpy as np

class MyClassifier(MLController):
    model_name = "my-classifier"
    model_version = "1.0.0"
    
    def load_model(self):
        # Load your model here (sklearn, pytorch, tensorflow, etc.)
        return lambda x: np.array([1 if sum(x[0]) > 0 else 0])
    
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

# Run the server
if __name__ == "__main__":
    MyClassifier().run()
```

Or with uvicorn:

```python
# main.py
classifier = MyClassifier()
app = classifier.app
```

```bash
uvicorn main:app --reload
```

## API Endpoints

Once running, your API will have:

- `POST /predict` - Run predictions
- `GET /health` - Health check
- `GET /` - API info
- `GET /docs` - Swagger documentation

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

### Custom Request/Response Models

```python
from pydantic import BaseModel
from typing import List

class ImageRequest(BaseModel):
    image_url: str
    threshold: float = 0.5

class ImageResponse(BaseModel):
    objects: List[dict]
    count: int

class ObjectDetector(MLController):
    model_name = "object-detector"
    request_model = ImageRequest
    response_model = ImageResponse
    
    def load_model(self):
        return load_yolo_model()
    
    @preprocessing
    def preprocess(self, data: dict):
        image = download_image(data["image_url"])
        return image
    
    @postprocessing
    def postprocess(self, detections) -> dict:
        return {
            "objects": detections,
            "count": len(detections)
        }
```

### Custom Prediction Logic

Override `predict_raw` for models without a standard `predict()` method:

```python
class PyTorchController(MLController):
    def load_model(self):
        model = torch.load("model.pt")
        model.eval()
        return model
    
    def predict_raw(self, preprocessed_data):
        with torch.no_grad():
            return self.model(preprocessed_data)
```

## Configuration

### MLController Options

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "ml-model" | Name of the model |
| `model_version` | str | "1.0.0" | Model version |
| `title` | str | "FastMLAPI" | API title for docs |
| `description` | str | "ML Model Serving API" | API description |
| `api_version` | str | "1.0.0" | API version |
| `request_model` | BaseModel | PredictionRequest | Custom request schema |
| `response_model` | BaseModel | PredictionResponse | Custom response schema |
| `enable_health` | bool | True | Enable /health endpoint |
| `enable_docs` | bool | True | Enable Swagger/OpenAPI docs |

### run() Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "0.0.0.0" | Host to bind to |
| `port` | int | 8000 | Port to bind to |
| `reload` | bool | False | Enable auto-reload |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run example
python examples/simple_classifier.py
```

## License

MIT License

# FastMLAPI

A FastAPI-based ML model serving library for easy deployment. Create production-ready ML APIs with minimal boilerplate.

## Features

- 🚀 **Simple API**: Extend `MLController` and implement a few methods
- 🔄 **Automatic `/predict` endpoint**: Generated automatically with proper request/response handling
- 🎯 **Flexible prediction**: Use `load_model()` for standard models or `@prediction` decorator for custom logic
- 🔧 **Data pipelines**: `@preprocessing` and `@postprocessing` decorators for clean data flow
- �️ **Custom endpoints**: Add additional routes with `@route` decorator
- 🧰 **setup() hook**: Override `setup()` for extra initialization and route registration
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
| `/health` | GET | Readiness: 200 when the model is loaded, 503 otherwise |
| `/livez` | GET | Liveness: 200 whenever the process is up |
| `/schema` | GET | JSON Schema of request/response plus optional examples |
| `/info` | GET | Name, version, library version, GPU availability |
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

## Health, Schema and Deployment

These features are generic and opt-in; they are what you need when the service runs in a
container behind an orchestrator or gateway, but they are useful anywhere.
A complete container example lives in [`examples/containerized/`](examples/containerized/).

### Readiness and liveness

| Endpoint | Meaning | Status |
|----------|---------|--------|
| `GET /health` | Readiness. Same `HealthResponse` body as always. | `200` when the model is loaded, `503` otherwise |
| `GET /livez` | Liveness. | `200` whenever the process is up |

Plain HTTP probes only look at the status code, so `/health` returns `503` (with the
`"status": "not_ready"` body) until the model is loaded.

**Model loading can be slow.** `setup()` and `load_model()` run during application startup,
before the server accepts connections, so while a large model loads the port is not yet open
(connection refused, not `503`). This is normal. Configure the orchestrator with a
*startup probe* (for example Kubernetes `startupProbe` on `/health` with a generous
`failureThreshold * periodSeconds`) so the container is not killed while loading, and use
`/livez` for the liveness probe and `/health` for the readiness probe.

### Schema endpoint

`GET /schema` returns the JSON Schema of the request and response so clients can discover the
expected input without reading your code:

```json
{
  "model_name": "my-classifier",
  "model_version": "1.0.0",
  "request_schema": {"...": "JSON Schema of request_model"},
  "response_schema": {"...": "JSON Schema of response_model"},
  "example": {"request": {"features": [5.1, 3.5, 1.4, 0.2]}, "response": null}
}
```

The schemas come from `request_model` / `response_model`, falling back to the default
`PredictionRequest` / `PredictionResponse`. Set `request_example` / `response_example`
(dict or `None`) to fill `example`.

If `request_model` is not set, a **warning is logged at startup**: the default schema is just
`data: Any`, which tells clients nothing. Define a typed model instead:

```python
from typing import List
from pydantic import BaseModel, Field
from fastmlapi import MLController

class IrisRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4,
                                  description="sepal/petal length and width in cm")

class IrisClassifier(MLController):
    model_name = "iris"
    request_model = IrisRequest
    request_example = {"features": [5.1, 3.5, 1.4, 0.2]}
```

### Cache directory

`self.cache_dir` (a `pathlib.Path`) is read from `FASTMLAPI_CACHE_DIR`
(default `./.cache/fastmlapi`). It is created if missing before `setup()` runs, so it can be
used in both `setup()` and `load_model()`. Mount a persistent volume there to keep downloads
across restarts:

```python
import os

class HFModel(MLController):
    def load_model(self):
        os.environ.setdefault("HF_HOME", str(self.cache_dir / "hf"))
        from transformers import pipeline
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
```

**Weights baked into the image vs. downloaded at startup:**

- *Baked in*: fast, reproducible starts and no network needed at runtime, but large images,
  slow pulls and a rebuild for every weight change.
- *Downloaded into a persistent cache*: small images and weights updated independently of the
  code, but the first start is slow, needs network access and credentials, and the cache
  volume must be persistent (otherwise every restart re-downloads).

### GPU memory cap (best effort)

Set `gpu_memory_fraction = 0.5` on the class, or `FASTMLAPI_GPU_MEMORY_FRACTION=0.5` (the
environment wins). If `torch` is installed and CUDA is available, fastmlapi calls
`torch.cuda.set_per_process_memory_fraction(...)` before `setup()` and `load_model()`.
Without torch or CUDA nothing happens (an info message is logged). Values outside `(0, 1]`
raise `ValueError` at startup. This only limits PyTorch's caching allocator; other frameworks
(TensorFlow, JAX, ONNX Runtime, ...) need their own settings.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FASTMLAPI_HOST` | `0.0.0.0` | Default bind host for `run()` |
| `FASTMLAPI_PORT` | `8000` | Default bind port for `run()` |
| `FASTMLAPI_CACHE_DIR` | `./.cache/fastmlapi` | Persistent cache directory (`self.cache_dir`) |
| `FASTMLAPI_GPU_MEMORY_FRACTION` | unset | Overrides `gpu_memory_fraction` |

Arguments passed explicitly to `run(host=..., port=...)` take priority over the environment.

### Graceful shutdown

`run()` uses Uvicorn, which handles `SIGTERM` (what Docker and Kubernetes send on stop) by
finishing in-flight requests and running application shutdown before the process exits.
Pass e.g. `timeout_graceful_shutdown=20` to `run()` to bound the wait, and keep the
orchestrator's grace period (`docker stop -t`, Kubernetes `terminationGracePeriodSeconds`)
longer than that. Note that Uvicorn re-raises the signal after shutting down, so the exit
status is typically 143 (`128 + SIGTERM`) rather than 0. In a Dockerfile use the exec form
of `CMD` so Python is PID 1 and receives the signal.

## Advanced Usage

### Initialization with `setup()`

Use `setup()` to perform additional initialization (load auxiliary data, warm caches) or register routes
after the FastAPI app is configured and before the model is loaded.

```python
from fastmlapi import MLController

def load_thresholds(path: str) -> dict:
    """Example helper; return a mapping of labels to threshold values."""
    return {"default": 0.5}

class ThresholdController(MLController):
    model_name = "threshold-model"
    
    def setup(self):
        self.thresholds = load_thresholds("thresholds.json")
        self.add_route("/status", self.status, methods=["GET"])
    
    async def status(self):
        return {"ready": self.is_loaded, "thresholds": self.thresholds}
```

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

You can also add routes programmatically in `setup()`:

```python
class MyController(MLController):
    def setup(self):
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

## Long-running jobs (queue mode)

For slow or heavy inference, move the model out of the API process. The API accepts a job and returns immediately; worker processes (same or other machines) take jobs from Redis, run your pipeline and store the result. Clients submit, then poll.

```bash
pip install fastmlapi[queue]
```

Write one `MLController` subclass and start it in either role:

```python
import sys

if __name__ == "__main__":
    controller = MyModel()
    if sys.argv[1:] == ["worker"]:
        controller.run_worker(redis_url="redis://localhost:6379/0")
    else:
        controller.run(queue_url="redis://localhost:6379/0")
```

See [`examples/queue_worker.py`](examples/queue_worker.py) for a complete example. Start as many workers as you need; they scale horizontally against the same Redis.

To serve the API with `uvicorn main:app`, enable queue mode before taking `.app`, with the same Redis URL the workers use:

```python
# main.py
controller = MyModel()
controller.use_queue("redis://localhost:6379/0")
app = controller.app
```

Pass the URL string, not a `RedisBackend` you built yourself: the URL form uses the key prefix `fastmlapi:{model_name}:`, the same as `run_worker(redis_url=...)`. A `RedisBackend(url)` defaults to the prefix `fastmlapi:`, which workers do not read.

The raw JSON request body is queued (after `POST /jobs` has validated it); the worker validates it with `request_model` exactly as `/predict` does, so `preprocess()` receives the same input in both modes.

### Endpoints

| Endpoint | Behaviour |
|----------|-----------|
| `POST /jobs` | Validates the body against `request_model` (422 if invalid), enqueues it and returns `202 {"job_id": "..."}`. 503 if Redis is unreachable or returns an error. |
| `GET /jobs/{id}` | Returns `status` (`queued`, `running`, `succeeded` or `failed`) plus `result` (succeeded) or `error` / `error_type` (failed), and `attempts`, `created_at`, `started_at`, `finished_at`. 404 if unknown or expired. |
| `GET /health` | 200 when the queue backend is reachable, 503 otherwise; includes `queue_connected` and `queue_depth`. `model_loaded` is omitted. It does not tell you whether any worker is alive. |

`/predict` is not available in queue mode, and the API process does not load the model (`setup()` still runs). Workers run `setup()` and `load_model()` at startup and exit non-zero if they fail.

### Tuning

Override these class attributes on your controller:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `result_ttl` | `3600` | Seconds a finished job (result or error) is kept |
| `visibility_timeout` | `30` | Seconds without a heartbeat before a running job is considered lost and requeued |
| `max_attempts` | `3` | Times a job may be claimed; after that a lost job is marked `failed` |
| `heartbeat_interval` | `5` | Seconds between worker heartbeats |

### Caveats

- **At-least-once delivery.** A job can run twice if a worker dies after finishing but before saving the result. Make inference idempotent where that matters.
- **No hard per-job timeout.** Python cannot safely interrupt a running model, so a stuck job keeps its worker busy. Run time does not matter as long as heartbeats continue.
- **Heartbeats and the GIL.** A long CPU-bound call that never releases the GIL can delay heartbeats; if jobs get requeued while still running, raise `visibility_timeout`.
- **Results must be JSON-serializable.** Otherwise the job fails with a serialization error.
- **Results expire.** Finished jobs are deleted after `result_ttl`.
- **Crash recovery is done by workers, not the API.** Workers requeue jobs of dead workers when they sweep. The sweep runs between jobs (at the start of each claim attempt, at most every `sweep_interval`, default `max(visibility_timeout / 2, 1)` seconds). If every worker is busy with a long job, a dead worker's job is recovered once one of them becomes free; if all workers are dead, it is recovered when a worker starts.
- **No authentication on `/jobs`.** Anyone who can reach the API can submit jobs and poll them. Job ids are random (uuid4) and hard to guess, but are not secrets; put the API behind your own auth if needed.
- **Configure Redis to keep jobs.** Use persistence and `maxmemory-policy noeviction`. If Redis is flushed or evicts keys, queued jobs are lost and show as 404.
- **Redis Cluster is not supported.** Use a single Redis instance (with replicas/Sentinel if you like).
- **One queue per `model_name`.** Queue keys are named after `model_name`, so controllers that keep the default `"ml-model"` share a queue. Give each model a unique `model_name`.
- **Redis timeouts.** `RedisBackend` uses 5 s socket and connect timeouts by default (`socket_timeout=` / `socket_connect_timeout=`, or set them in the URL query string).
- **Worker ids must be unique per process.** Completion, failure and heartbeats are fenced by worker id, so a worker that lost its job cannot overwrite the new owner. The default id (`hostname-pid-random`) is unique per process; if you pass a custom `worker_id`, keep it unique.

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
| `request_example` | dict \| None | `None` | Example request body, shown by `/schema` |
| `response_example` | dict \| None | `None` | Example response body, shown by `/schema` |
| `gpu_memory_fraction` | float \| None | `None` | Optional GPU memory cap, `0 < x <= 1` (see below) |
| `enable_health` | bool | `True` | Enable `/health` and `/livez` endpoints |
| `enable_docs` | bool | `True` | Enable Swagger/OpenAPI docs |

### run() Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | `$FASTMLAPI_HOST`, else `"0.0.0.0"` | Host to bind to |
| `port` | int | `$FASTMLAPI_PORT`, else `8000` | Port to bind to |
| `reload` | bool | `False` | Enable auto-reload (dev only) |
| `queue_url` | str \| None | `None` | Redis URL; enables queue mode (see above) |
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

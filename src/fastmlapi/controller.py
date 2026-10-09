"""
MLController - Unified ML model controller with built-in FastAPI server.
"""

from abc import ABC
from typing import Any, Dict, Optional, Type, Callable, Union
import importlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ._version import __version__
from .queue.backend import QueueBackend, QueueUnavailable
from .types import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
)


logger = logging.getLogger(__name__)

ENV_HOST = "FASTMLAPI_HOST"
ENV_PORT = "FASTMLAPI_PORT"
ENV_CACHE_DIR = "FASTMLAPI_CACHE_DIR"
ENV_GPU_MEMORY_FRACTION = "FASTMLAPI_GPU_MEMORY_FRACTION"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_CACHE_DIR = "./.cache/fastmlapi"


class MLController(ABC):
    """
    Unified ML model controller with built-in FastAPI server.
    
    Users should extend this class and implement prediction logic in one of these ways:
    
    Option 1: Traditional approach with load_model (recommended for sklearn, etc.)
        - load_model(): Load and return the ML model
        - preprocess(): Prepare input data for the model (use @preprocessing decorator)
        - postprocess(): Format model output for API response (use @postprocessing decorator)
    
    Option 2: Custom prediction function (recommended for complex inference)
        - Use @prediction decorator on a method that handles the full prediction
        - preprocess(): Optional preprocessing (use @preprocessing decorator)
        - postprocess(): Optional postprocessing (use @postprocessing decorator)
        - load_model() becomes optional
    
    Example with load_model:
        from fastmlapi import MLController, preprocessing, postprocessing
        import numpy as np
        
        class MyClassifier(MLController):
            model_name = "my-classifier"
            model_version = "1.0.0"
            
            def load_model(self):
                return joblib.load("model.pkl")
            
            @preprocessing
            def preprocess(self, data: dict) -> np.ndarray:
                return np.array(data["features"]).reshape(1, -1)
            
            @postprocessing
            def postprocess(self, prediction: np.ndarray) -> dict:
                return {"class": int(prediction[0])}
    
    Example with @prediction decorator:
        from fastmlapi import MLController, prediction, postprocessing
        import torch
        
        class PyTorchModel(MLController):
            model_name = "pytorch-model"
            
            def load_model(self):
                model = torch.load("model.pt")
                model.eval()
                return model
            
            @prediction
            def run_inference(self, data: dict) -> torch.Tensor:
                with torch.no_grad():
                    tensor = torch.tensor(data["features"])
                    return self.model(tensor)
            
            @postprocessing
            def postprocess(self, output: torch.Tensor) -> dict:
                return {"result": output.tolist()}
    
    Example without load_model (external service, custom logic):
        from fastmlapi import MLController, prediction
        
        class ExternalAPIController(MLController):
            model_name = "external-api"
            
            def load_model(self):
                return None  # No model needed
            
            @prediction
            def call_external(self, data: dict) -> dict:
                response = requests.post("https://api.example.com/predict", json=data)
                return response.json()
        
        # Run the server
        if __name__ == "__main__":
            MyClassifier().run()
    """
    
    # Class attributes that can be overridden
    model_name: str = "ml-model"
    model_version: str = "1.0.0"
    
    # API configuration
    title: str = "FastMLAPI"
    description: str = "ML Model Serving API"
    api_version: str = "1.0.0"
    
    # Custom request/response models (optional)
    request_model: Optional[Type[BaseModel]] = None
    response_model: Optional[Type[BaseModel]] = None
    
    # Optional examples, exposed by GET /schema
    request_example: Optional[Dict[str, Any]] = None
    response_example: Optional[Dict[str, Any]] = None
    
    # Optional per-process GPU memory cap (0 < x <= 1), best effort, torch only.
    # Overridden by the FASTMLAPI_GPU_MEMORY_FRACTION environment variable.
    gpu_memory_fraction: Optional[float] = None
    
    # Enable/disable features
    enable_health: bool = True
    enable_docs: bool = True

    # Job queue tuning (only used in queue mode / by workers)
    result_ttl: float = 3600.0          # seconds a finished job's result is kept
    visibility_timeout: float = 30.0    # seconds without a heartbeat before a job is requeued
    max_attempts: int = 3               # claims before a job is failed as "worker lost"
    heartbeat_interval: float = 5.0     # seconds between worker heartbeats
    
    def __init__(self):
        self._model = None
        self._ready = False
        self._gpu_available: Optional[bool] = None
        # Persistent cache directory (FASTMLAPI_CACHE_DIR). The path is known
        # from construction; the directory itself is created before setup().
        self.cache_dir: Path = Path(
            os.environ.get(ENV_CACHE_DIR) or DEFAULT_CACHE_DIR
        ).expanduser()
        self._preprocessing_fn: Optional[Callable] = None
        self._postprocessing_fn: Optional[Callable] = None
        self._prediction_fn: Optional[Callable] = None
        self._custom_routes: list = []
        self._app: Optional[FastAPI] = None
        self._queue_backend: Optional[QueueBackend] = None
        self._discover_processors()
    
    def _discover_processors(self) -> None:
        """Discover methods marked with @preprocessing, @postprocessing, @prediction, and @route decorators."""
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue
            try:
                # Get from class first to check if it's a property
                class_attr = getattr(self.__class__, name, None)
                if isinstance(class_attr, property):
                    continue
                
                # Now get the bound method from instance
                method = getattr(self, name, None)
                if method is None or not callable(method):
                    continue
                    
                if hasattr(method, '_is_preprocessing') and method._is_preprocessing:
                    self._preprocessing_fn = method
                    logger.debug(f"Discovered preprocessing function: {name}")
                if hasattr(method, '_is_postprocessing') and method._is_postprocessing:
                    self._postprocessing_fn = method
                    logger.debug(f"Discovered postprocessing function: {name}")
                if hasattr(method, '_is_prediction') and method._is_prediction:
                    self._prediction_fn = method
                    logger.debug(f"Discovered prediction function: {name}")
                if hasattr(method, '_is_route') and method._is_route:
                    config = getattr(method, '_route_config', {})
                    self._custom_routes.append({
                        "endpoint": method,
                        **config,
                    })
                    logger.debug(f"Discovered custom route: {name} -> {config.get('path')}")
            except Exception:
                continue
    
    def setup(self):
        """
        Override this method to initialize the model or perform other setup tasks.
        
        This method is run after FastAPI self.app is configured, so it can be used to add custom endpoints.
        """
        pass
    
    def load_model(self) -> Any:
        """
        Load and return the ML model.
        
        This method is called once during server startup.
        
        Override this method to load your model. If you're using the @prediction
        decorator with custom logic that doesn't need a model, you can either:
        - Not override this method (returns None)
        - Return None explicitly
        
        Returns:
            The loaded model object, or None if not needed.
        
        Examples:
            # sklearn
            def load_model(self):
                return joblib.load("model.pkl")
            
            # PyTorch
            def load_model(self):
                model = torch.load("model.pt")
                model.eval()
                return model
            
            # TensorFlow/Keras
            def load_model(self):
                return tf.keras.models.load_model("model.h5")
            
            # External API (no model needed)
            def load_model(self):
                return None
        """
        return None
    
    def _initialize(self) -> None:
        """
        Do NOT override this method! You can override init instead, to do perform setup tasks.
        
        Initialize the controller and load the model.
        """
        logger.info("Initializing controller")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._apply_gpu_memory_fraction()
        self.setup()
        logger.info(f"Loading model: {self.model_name} v{self.model_version}")
        self._model = self.load_model()
        self._ready = True
        logger.info(f"Model loaded successfully: {self.model_name}")
    
    def _resolve_gpu_memory_fraction(self) -> Optional[float]:
        """Return the configured GPU memory fraction (env overrides class attribute)."""
        raw: Any = os.environ.get(ENV_GPU_MEMORY_FRACTION) or self.gpu_memory_fraction
        if raw is None:
            return None
        try:
            fraction = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid GPU memory fraction {raw!r}: expected a number in (0, 1]")
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid GPU memory fraction {fraction}: must satisfy 0 < x <= 1")
        return fraction
    
    def _apply_gpu_memory_fraction(self) -> None:
        """Best-effort GPU memory cap via torch. No-op if torch or CUDA is unavailable."""
        fraction = self._resolve_gpu_memory_fraction()
        if fraction is None:
            return
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            logger.info("GPU memory fraction %s requested but torch is not installed; ignoring", fraction)
            return
        if not torch.cuda.is_available():
            logger.info("GPU memory fraction %s requested but CUDA is not available; ignoring", fraction)
            return
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info("Set per-process GPU memory fraction to %s", fraction)
    
    @property
    def gpu_available(self) -> bool:
        """Whether a CUDA GPU is visible to torch (False if torch is not installed)."""
        if self._gpu_available is None:
            try:
                torch = importlib.import_module("torch")
                self._gpu_available = bool(torch.cuda.is_available())
            except Exception:
                self._gpu_available = False
        return self._gpu_available
    
    @property
    def model(self) -> Any:
        """
        Get the loaded model instance.
        
        Note: If using @prediction decorator without a model, this may return None.
        """
        if self._model is None and self._prediction_fn is None:
            raise RuntimeError(
                "Model not loaded. Either implement load_model() to return a model, "
                "or use @prediction decorator for custom prediction logic."
            )
        return self._model
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded or prediction function is available."""
        return self._model is not None or self._prediction_fn is not None
    
    def add_route(
        self,
        path: str,
        endpoint: Callable,
        methods: Optional[list] = None,
        response_model: Optional[Type[BaseModel]] = None,
        tags: Optional[list] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a custom route to the API.
        
        This method allows you to add additional endpoints beyond /predict.
        Routes must be added before accessing the `app` property or calling `run()`.
        
        Args:
            path: The URL path for the endpoint (e.g., "/analyze", "/batch")
            endpoint: The async function to handle requests
            methods: HTTP methods (default: ["GET"])
            response_model: Optional Pydantic model for response validation
            tags: OpenAPI tags for documentation
            summary: Short summary for OpenAPI docs
            description: Detailed description for OpenAPI docs
            **kwargs: Additional arguments passed to FastAPI's add_api_route
        
        Example:
            class MyController(MLController):
                def __init__(self):
                    super().__init__()
                    self.add_route("/batch", self.batch_predict, methods=["POST"])
                
                async def batch_predict(self, request: Request):
                    data = await request.json()
                    results = [await self.predict(item) for item in data["items"]]
                    return {"results": results}
        """
        if self._app is not None:
            raise RuntimeError(
                "Cannot add routes after the app has been created. "
                "Add routes in __init__ before accessing .app or calling .run()"
            )
        
        self._custom_routes.append({
            "path": path,
            "endpoint": endpoint,
            "methods": methods or ["GET"],
            "response_model": response_model,
            "tags": tags or ["Custom"],
            "summary": summary,
            "description": description,
            **kwargs,
        })
    
    def use_queue(self, backend: Union[str, QueueBackend]) -> None:
        """Run in queue mode: `/jobs` replaces `/predict` and this process
        does not load the model (start workers with `run_worker()`).

        `backend` is a Redis URL (keys prefixed `fastmlapi:{model_name}:`,
        the same keys `run_worker(redis_url=...)` uses) or a QueueBackend.
        Must be called before the app is created.
        """
        if self._app is not None:
            raise RuntimeError(
                "Cannot enable the job queue after the app has been created. "
                "Call use_queue() (or run(queue_url=...)) before accessing .app"
            )
        if isinstance(backend, str):
            backend = self._redis_backend(backend)
        self._queue_backend = backend

    def _redis_backend(self, url: str) -> QueueBackend:
        from .queue.redis_backend import RedisBackend

        return RedisBackend(
            url, prefix=f"fastmlapi:{self.model_name}:", result_ttl=self.result_ttl
        )

    def _initialize_api_only(self) -> None:
        """Queue-mode API startup: no model, workers load it."""
        logger.info("Initializing controller (queue mode, model not loaded)")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.setup()
        self._ready = True

    def run_worker(
        self,
        redis_url: Optional[str] = None,
        *,
        backend: Optional[QueueBackend] = None,
        **worker_kwargs: Any,
    ) -> None:
        """Run a queue worker in this process: load the model, then process
        jobs until SIGINT/SIGTERM. Exits non-zero if startup fails."""
        if backend is None:
            if redis_url is None:
                raise ValueError("run_worker() needs redis_url (or backend=)")
            backend = self._redis_backend(redis_url)
        from .queue.worker import Worker

        print(f"Starting worker for {self.model_name} v{self.model_version}...")
        try:
            Worker(self, backend, **worker_kwargs).run()
        except Exception:
            logger.exception("Worker stopped: failed to start or fatal error")
            sys.exit(1)

    def _request_to_input(self, validated: BaseModel) -> Any:
        """The model input for a validated request (shared by /predict and
        queue workers): `.data` if the request model has it, else the whole
        request as a dict."""
        if hasattr(validated, 'data'):
            return getattr(validated, 'data')
        # If custom request model, pass the whole request
        return validated.model_dump()

    def preprocess(self, data: Any) -> Any:
        """
        Default preprocessing - returns data as-is.
        
        Override this method with @preprocessing decorator to customize.
        """
        if self._preprocessing_fn:
            return self._preprocessing_fn(data)
        return data
    
    def postprocess(self, prediction: Any) -> Dict[str, Any]:
        """
        Default postprocessing - wraps prediction in a dict.
        
        Override this method with @postprocessing decorator to customize.
        """
        if self._postprocessing_fn:
            return self._postprocessing_fn(prediction)
        
        # Default: try to make it JSON serializable
        if hasattr(prediction, 'tolist'):
            return {"result": prediction.tolist()}
        return {"result": prediction}
    
    def predict_raw(self, preprocessed_data: Any) -> Any:
        """
        Run prediction on preprocessed data.
        
        This method is called after preprocessing and before postprocessing.
        
        You can customize prediction in several ways:
        1. Use the @prediction decorator on a custom method (recommended)
        2. Override this method directly
        3. Return a model with a predict() method from load_model()
        4. Return a callable from load_model()
        
        If using @prediction decorator, load_model() becomes optional.
        """
        # Use custom prediction function if decorated with @prediction
        if self._prediction_fn:
            return self._prediction_fn(preprocessed_data)
        
        # Otherwise use the loaded model
        if hasattr(self.model, 'predict'):
            return self.model.predict(preprocessed_data)
        elif callable(self.model):
            return self.model(preprocessed_data)
        else:
            raise NotImplementedError(
                "Model doesn't have a predict() method and is not callable. "
                "Use @prediction decorator or override predict_raw() to implement custom prediction logic."
            )
    
    async def predict(self, data: Any) -> Dict[str, Any]:
        """
        Full prediction pipeline: preprocess -> predict -> postprocess.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed prediction result as a dictionary
        """
        # Preprocess
        preprocessed = self.preprocess(data)
        
        # Predict
        raw_prediction = self.predict_raw(preprocessed)
        
        # Postprocess
        result = self.postprocess(raw_prediction)
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """Return health check information."""
        backend = self._queue_backend
        if backend is not None:
            try:
                connected = backend.ping()
                depth = backend.pending_count() if connected else None
            except QueueUnavailable:
                connected, depth = False, None
            return {
                "status": "healthy" if connected else "not_ready",
                "model_name": self.model_name,
                "version": self.model_version,
                "queue_connected": connected,
                "queue_depth": depth,
            }
        ready = self._ready and self.is_loaded
        return {
            "status": "healthy" if ready else "not_ready",
            "model_loaded": ready,
            "model_name": self.model_name,
            "version": self.model_version,
        }
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        controller = self
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: Initialize controller
            logger.info(f"Initializing {controller.model_name}...")
            if controller.request_model is None:
                logger.warning(
                    f"{controller.model_name}: request_model is not set, so the request schema "
                    "is the generic `data: Any`. Set request_model to a Pydantic model so "
                    "clients can discover the expected input via GET /schema."
                )
            try:
                if controller._queue_backend is not None:
                    controller._initialize_api_only()
                else:
                    controller._initialize()
            except Exception as e:
                logger.error(f"Failed to initialize {controller.model_name}: {e}")
                raise
            logger.info(f"{controller.model_name} initialized successfully")
            yield
            # Shutdown
            logger.info(f"Shutting down {controller.model_name}")
        
        app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.api_version,
            lifespan=lifespan,
            docs_url="/docs" if self.enable_docs else None,
            redoc_url="/redoc" if self.enable_docs else None,
        )
        
        # Global exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            logger.exception(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=str(exc),
                    error_type=type(exc).__name__,
                ).model_dump(),
            )
        
        # Health endpoint
        if self.enable_health:
            @app.get(
                "/health",
                response_model=HealthResponse,
                response_model_exclude_none=True,
                responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
                tags=["Health"],
                summary="Health Check",
            )
            async def health(response: Response):
                """Readiness: 200 once the model is loaded, 503 otherwise."""
                result = await run_in_threadpool(controller.health_check)
                if result.get("status") != "healthy":
                    response.status_code = 503
                return result
            
            @app.get("/livez", tags=["Health"], summary="Liveness Check")
            async def livez():
                """Liveness: 200 whenever the process is up, regardless of model state."""
                return {"status": "alive"}
        
        # Root endpoint
        @app.get("/info", tags=["Info"])
        async def info():
            """API information endpoint."""
            return {
                "name": controller.model_name,
                "version": controller.model_version,
                "endpoint": "/jobs" if controller._queue_backend is not None else "/predict",
                "library_version": __version__,
                "gpu_available": controller.gpu_available,
            }
        
        RequestModel: Type[BaseModel] = self.request_model or PredictionRequest
        ResponseModel: Type[BaseModel] = self.response_model or PredictionResponse
        
        # Schema endpoint
        @app.get("/schema", tags=["Info"], summary="Input/Output Schema")
        async def schema():
            """JSON Schema of the request and response, plus optional examples."""
            return {
                "model_name": controller.model_name,
                "model_version": controller.model_version,
                "request_schema": RequestModel.model_json_schema(),
                "response_schema": ResponseModel.model_json_schema(),
                "example": {
                    "request": controller.request_example,
                    "response": controller.response_example,
                },
            }
        
        if self._queue_backend is not None:
            from .queue.api import register_queue_routes

            register_queue_routes(app, self, self._queue_backend, RequestModel)
        else:
            self._register_predict_route(app, RequestModel, ResponseModel)
        
        # Register custom routes
        for route_config in self._custom_routes:
            endpoint = route_config.pop("endpoint")
            path = route_config.pop("path")
            methods = route_config.pop("methods", ["GET"])
            
            app.add_api_route(
                path,
                endpoint,
                methods=methods,
                **route_config,
            )
            logger.debug(f"Registered custom route: {methods} {path}")
        
        return app
    
    def _register_predict_route(
        self,
        app: FastAPI,
        RequestModel: Type[BaseModel],
        ResponseModel: Type[BaseModel],
    ) -> None:
        # Predict endpoint
        
        async def predict_endpoint(request: Request) -> Any:
            """Run prediction on the input data."""
            try:
                # Parse request body using the configured model
                body = await request.json()
                validated_request = RequestModel.model_validate(body)
                
                # Extract data from request
                data = self._request_to_input(validated_request)
                
                # Run prediction pipeline
                result = await self.predict(data)
                
                # Build response
                if self.response_model:
                    return result
                
                return PredictionResponse(
                    success=True,
                    prediction=result,
                    metadata={
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                    },
                )
                
            except Exception as e:
                logger.exception(f"Prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
        
        # Register the predict endpoint with proper OpenAPI schema
        app.add_api_route(
            "/predict",
            predict_endpoint,
            methods=["POST"],
            response_model=ResponseModel,
            tags=["Prediction"],
            summary=f"Predict using {self.model_name}",
            description=f"Run prediction using the {self.model_name} model (v{self.model_version})",
            openapi_extra={
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": RequestModel.model_json_schema()
                        }
                    },
                    "required": True,
                }
            },
        )
    
    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application instance."""
        if self._app is None:
            self._app = self._create_app()
        return self._app
    
    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reload: bool = False,
        queue_url: Optional[str] = None,
        **uvicorn_kwargs,
    ) -> None:
        """
        Run the FastAPI server with Uvicorn.
        
        Args:
            host: Host to bind to (default: $FASTMLAPI_HOST, else 0.0.0.0)
            port: Port to bind to (default: $FASTMLAPI_PORT, else 8000)
            reload: Enable auto-reload
            queue_url: Redis URL; enables queue mode (see use_queue)
            **uvicorn_kwargs: Additional arguments passed to uvicorn.run
        """
        import uvicorn
        
        if queue_url is not None:
            self.use_queue(queue_url)
        
        if host is None:
            host = os.environ.get(ENV_HOST) or DEFAULT_HOST
        if port is None:
            raw_port = os.environ.get(ENV_PORT)
            try:
                port = int(raw_port) if raw_port else DEFAULT_PORT
            except ValueError:
                raise ValueError(f"Invalid {ENV_PORT}={raw_port!r}: expected an integer")
        
        print(f"Starting {self.model_name} v{self.model_version}...")
        print(f"API docs available at: http://{host}:{port}/docs")
        if self._queue_backend is not None:
            print(f"Jobs endpoint: POST http://{host}:{port}/jobs")
        else:
            print(f"Predict endpoint: POST http://{host}:{port}/predict")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            **uvicorn_kwargs,
        )

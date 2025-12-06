"""
MLController - Unified ML model controller with built-in FastAPI server.
"""

from abc import ABC
from typing import Any, Dict, Optional, Type, Callable
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .types import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
)


logger = logging.getLogger(__name__)


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
    
    # Enable/disable features
    enable_health: bool = True
    enable_docs: bool = True
    
    def __init__(self):
        self._model = None
        self._preprocessing_fn: Optional[Callable] = None
        self._postprocessing_fn: Optional[Callable] = None
        self._prediction_fn: Optional[Callable] = None
        self._custom_routes: list = []
        self._app: Optional[FastAPI] = None
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
    
    def initialize(self) -> None:
        """Initialize the controller and load the model."""
        logger.info(f"Loading model: {self.model_name} v{self.model_version}")
        self._model = self.load_model()
        logger.info(f"Model loaded successfully: {self.model_name}")
    
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
        methods: list = None,
        response_model: Optional[Type[BaseModel]] = None,
        tags: Optional[list] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
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
        return {
            "status": "healthy" if self.is_loaded else "not_ready",
            "model_loaded": self.is_loaded,
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
            try:
                controller.initialize()
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
                tags=["Health"],
                summary="Health Check",
            )
            async def health():
                """Check the health status of the model."""
                return controller.health_check()
        
        # Root endpoint
        @app.get("/", tags=["Info"])
        async def root():
            """API information endpoint."""
            return {
                "name": controller.model_name,
                "version": controller.model_version,
                "endpoint": "/predict",
            }
        
        # Predict endpoint
        RequestModel: Type[BaseModel] = self.request_model or PredictionRequest
        ResponseModel: Type[BaseModel] = self.response_model or PredictionResponse
        
        async def predict_endpoint(request: Request) -> Any:
            """Run prediction on the input data."""
            try:
                # Parse request body using the configured model
                body = await request.json()
                validated_request = RequestModel.model_validate(body)
                
                # Extract data from request
                if hasattr(validated_request, 'data'):
                    data = getattr(validated_request, 'data')
                else:
                    # If custom request model, pass the whole request
                    data = validated_request.model_dump()
                
                # Run prediction pipeline
                result = await controller.predict(data)
                
                # Build response
                if controller.response_model:
                    return result
                
                return PredictionResponse(
                    success=True,
                    prediction=result,
                    metadata={
                        "model_name": controller.model_name,
                        "model_version": controller.model_version,
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
    
    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application instance."""
        if self._app is None:
            self._app = self._create_app()
        return self._app
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        **uvicorn_kwargs,
    ) -> None:
        """
        Run the FastAPI server with Uvicorn.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload
            **uvicorn_kwargs: Additional arguments passed to uvicorn.run
        """
        import uvicorn
        
        print(f"Starting {self.model_name} v{self.model_version}...")
        print(f"API docs available at: http://{host}:{port}/docs")
        print(f"Predict endpoint: POST http://{host}:{port}/predict")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            **uvicorn_kwargs,
        )

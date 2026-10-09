# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Queue mode for long-running inference: `run(queue_url=...)` serves `POST /jobs` and `GET /jobs/{id}` and does not load the model; `run_worker(redis_url=...)` runs a worker. Redis-backed (`pip install fastmlapi[queue]`), with heartbeats, crash recovery and graceful shutdown.
- `fastmlapi.queue` package: `QueueBackend`, `MemoryBackend`, `RedisBackend`, `Worker`
- `result_ttl`, `visibility_timeout`, `max_attempts`, `heartbeat_interval` class attributes
- `HealthResponse.queue_connected` / `queue_depth` (queue mode only)
- `use_queue()` accepts a Redis URL (same key prefix as `run_worker(redis_url=...)`), for `uvicorn main:app` deployments
- `RedisBackend(socket_timeout=5.0, socket_connect_timeout=5.0)`; options set in the URL take precedence

### Changed

- `HealthResponse.model_loaded` is now optional (omitted in queue mode); sync-mode output is unchanged

## [0.2.0] - 2026-10-08

### Added

- `GET /livez` liveness endpoint (always 200 while the process is up; registered with `enable_health`)
- `GET /schema` endpoint returning `model_name`, `model_version`, `request_schema`, `response_schema`, `example`
- Optional `request_example` / `response_example` class attributes
- Startup warning when `request_model` is not set
- `self.cache_dir` (`pathlib.Path`) from `FASTMLAPI_CACHE_DIR` (default `./.cache/fastmlapi`), created before `setup()`
- Optional `gpu_memory_fraction` class attribute / `FASTMLAPI_GPU_MEMORY_FRACTION` (best effort, torch only, lazily imported; no-op without torch/CUDA)
- `run()` defaults from `FASTMLAPI_HOST` / `FASTMLAPI_PORT`
- `/info` gains `library_version` and `gpu_available`
- `examples/containerized/` (Dockerfile, minimal server, README, example CI workflow)
- Documentation for startup probes, graceful SIGTERM shutdown, cache directory and GPU cap

### Changed

- **Behaviour change:** `GET /health` now returns HTTP **503** (same body, `"status": "not_ready"`)
  while the model is not loaded, instead of 200. Once loaded it returns 200 as before. Probes that
  only read the status code now work; clients that treated a 200 with `not_ready` as "up" must
  check the status code. This applies to all controllers, including a controller that overrides
  `predict_raw()` and returns `None` from `load_model()` (it never reports ready, as before, but now
  with 503).
- `/health` reports not ready until startup (`setup()` + `load_model()`) has completed, also for
  `@prediction`-only controllers (previously they reported healthy before initialization).
- `run()` signature is now `host: Optional[str] = None, port: Optional[int] = None`; without
  arguments and environment variables the behaviour is unchanged (`0.0.0.0:8000`).
- `__version__` now lives in `fastmlapi/_version.py` (still exported as `fastmlapi.__version__`).

### Note

- The new built-in `/schema` and `/livez` routes take precedence over custom routes with the same path.

## [0.1.2] - 2024-12-06

### Added

- `@route` decorator for adding custom API endpoints
- `add_route()` method for programmatic route registration
- Custom endpoints are auto-discovered and registered with FastAPI
- Full OpenAPI documentation support for custom routes

## [0.1.1] - 2024-12-06

### Added

- `@prediction` decorator for custom prediction logic (PyTorch, TensorFlow, external APIs)
- `load_model()` is now optional when using `@prediction` decorator
- Improved README with comprehensive usage examples and documentation

### Changed

- `MLController` no longer requires `load_model()` to be overridden
- Better error messages when model is not loaded and no `@prediction` is defined

## [0.1.0] - 2024-12-06

### Added

- Initial release of FastMLAPI
- `MLController` base class for creating ML APIs
- `@preprocessing` decorator for data preprocessing pipelines
- `@postprocessing` decorator for formatting model outputs
- Built-in `/predict` endpoint with automatic request/response handling
- Built-in `/health` endpoint for health checks
- Built-in `/` endpoint for API information
- Automatic Swagger/OpenAPI documentation
- Support for custom Pydantic request/response models
- `PredictionRequest` and `PredictionResponse` default models
- Uvicorn server integration with configurable host/port
- Python 3.9+ support
- Full type hints and typing support

### Dependencies

- FastAPI >= 0.100.0
- Uvicorn >= 0.23.0
- Pydantic >= 2.0.0

[Unreleased]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/grzegorzprywatny/fastmlapi/releases/tag/v0.1.0

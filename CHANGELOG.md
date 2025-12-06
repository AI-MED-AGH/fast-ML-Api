# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/grzegorzprywatny/fastmlapi/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/grzegorzprywatny/fastmlapi/releases/tag/v0.1.0

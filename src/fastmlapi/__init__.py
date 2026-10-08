"""
FastMLAPI - A FastAPI-based ML worker node library for easy model deployment.
"""

from .controller import MLController
from .decorators import preprocessing, postprocessing, prediction, route
from ._version import __version__
from .types import PredictionRequest, PredictionResponse

__all__ = [
    "MLController",
    "preprocessing",
    "postprocessing",
    "prediction",
    "route",
    "PredictionRequest",
    "PredictionResponse",
]

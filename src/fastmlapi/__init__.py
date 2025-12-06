"""
FastMLAPI - A FastAPI-based ML worker node library for easy model deployment.
"""

from .controller import MLController
from .decorators import preprocessing, postprocessing
from .types import PredictionRequest, PredictionResponse

__version__ = "0.1.0"
__all__ = [
    "MLController",
    "preprocessing",
    "postprocessing",
    "PredictionRequest",
    "PredictionResponse",
]

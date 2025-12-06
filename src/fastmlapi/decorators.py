"""
Decorators for marking preprocessing and postprocessing methods.
"""

from functools import wraps
from typing import Callable, Any


def preprocessing(func: Callable) -> Callable:
    """
    Decorator to mark a method as a preprocessing function.
    
    The preprocessing function receives raw input data and should return
    data in the format expected by the model's predict method.
    
    Example:
        @preprocessing
        def preprocess(self, data: dict) -> np.ndarray:
            return np.array(data["features"])
    """
    func._is_preprocessing = True
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._is_preprocessing = True
    return wrapper


def postprocessing(func: Callable) -> Callable:
    """
    Decorator to mark a method as a postprocessing function.
    
    The postprocessing function receives raw model output and should return
    a JSON-serializable dictionary for the API response.
    
    Example:
        @postprocessing
        def postprocess(self, prediction: np.ndarray) -> dict:
            return {"class": int(prediction.argmax()), "probabilities": prediction.tolist()}
    """
    func._is_postprocessing = True
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._is_postprocessing = True
    return wrapper

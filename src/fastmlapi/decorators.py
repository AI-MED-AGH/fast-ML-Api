"""
Decorators for marking preprocessing and postprocessing methods.
"""

from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def preprocessing(func: F) -> F:
    """
    Decorator to mark a method as a preprocessing function.
    
    The preprocessing function receives raw input data and should return
    data in the format expected by the model's predict method.
    
    Example:
        @preprocessing
        def preprocess(self, data: dict) -> np.ndarray:
            return np.array(data["features"])
    """
    setattr(func, "_is_preprocessing", True)
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    
    setattr(wrapper, "_is_preprocessing", True)
    return cast(F, wrapper)


def postprocessing(func: F) -> F:
    """
    Decorator to mark a method as a postprocessing function.
    
    The postprocessing function receives raw model output and should return
    a JSON-serializable dictionary for the API response.
    
    Example:
        @postprocessing
        def postprocess(self, prediction: np.ndarray) -> dict:
            return {"class": int(prediction.argmax()), "probabilities": prediction.tolist()}
    """
    setattr(func, "_is_postprocessing", True)
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    
    setattr(wrapper, "_is_postprocessing", True)
    return cast(F, wrapper)

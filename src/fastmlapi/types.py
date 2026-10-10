"""
Type definitions for FastMLAPI.
"""

from typing import Any, Dict, TypeVar, Optional
from pydantic import BaseModel, Field

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PredictionRequest(BaseModel):
    """Base request model for predictions."""
    
    data: Any = Field(..., description="Input data for prediction")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata for the request"
    )

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    """Base response model for predictions."""
    
    success: bool = Field(default=True, description="Whether the prediction was successful")
    prediction: Any = Field(..., description="The prediction result")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the response"
    )

    model_config = {"extra": "allow"}


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(default="healthy")
    model_loaded: Optional[bool] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    queue_connected: Optional[bool] = Field(default=None)
    queue_depth: Optional[int] = Field(default=None)
    queue_workers: Optional[int] = Field(default=None, description="Live workers (queue mode)")
    queue_workers_ready: Optional[int] = Field(default=None, description="Live workers that have loaded the model")


class JobSubmitResponse(BaseModel):
    """Returned by POST /jobs."""

    job_id: str


class JobStatusResponse(BaseModel):
    """Returned by GET /jobs/{job_id}."""

    job_id: str
    status: str = Field(description="queued, running, succeeded or failed")
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attempts: int = 0
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

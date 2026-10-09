"""HTTP routes for queue mode."""

import logging
from typing import Any, Dict, Type, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..types import JobStatusResponse, JobSubmitResponse
from .backend import QueueBackend, QueueUnavailable


logger = logging.getLogger(__name__)

_ERROR_DETAIL = {
    "description": "Error",
    "content": {
        "application/json": {
            "example": {"detail": {"error": "...", "error_type": "..."}}
        }
    },
}
_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    503: {**_ERROR_DETAIL, "description": "Job queue unavailable"},
}


def _unavailable(exc: QueueUnavailable) -> HTTPException:
    # The backend message can name the Redis host/URL: log it, don't return it.
    logger.warning("Job queue unavailable: %s", exc)
    return HTTPException(
        status_code=503,
        detail={"error": "Job queue is unavailable", "error_type": "QueueUnavailable"},
    )


def register_queue_routes(
    app: FastAPI,
    controller: Any,
    backend: QueueBackend,
    request_model: Type[BaseModel],
) -> None:
    @app.post(
        "/jobs",
        status_code=202,
        response_model=JobSubmitResponse,
        tags=["Jobs"],
        summary=f"Submit a {controller.model_name} job",
        responses=_RESPONSES,
    )
    async def submit_job(body: request_model):  # type: ignore[valid-type]
        # The whole validated body is stored; the worker re-validates it so
        # preprocess() gets exactly what /predict would give it.
        payload = body.model_dump(mode="json")  # type: ignore[attr-defined]
        try:
            job_id = await run_in_threadpool(backend.enqueue, payload)
        except QueueUnavailable as exc:
            raise _unavailable(exc) from exc
        return JobSubmitResponse(job_id=job_id)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["Jobs"],
        summary="Get a job's status and result",
        responses={
            404: {**_ERROR_DETAIL, "description": "Job not found or expired"},
            **_RESPONSES,
        },
    )
    async def get_job(job_id: str):
        try:
            job = await run_in_threadpool(backend.get_job, job_id)
        except QueueUnavailable as exc:
            raise _unavailable(exc) from exc
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Job {job_id} not found or expired",
                    "error_type": "JobNotFound",
                },
            )
        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            result=job.result,
            error=job.error,
            error_type=job.error_type,
            attempts=job.attempts,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

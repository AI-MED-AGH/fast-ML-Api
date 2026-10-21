"""HTTP routes for queue mode."""

import logging
from typing import Any, Dict, Type, Union

from fastapi import FastAPI, HTTPException, Request
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
_SUBMIT_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    **_RESPONSES,
    429: {**_ERROR_DETAIL, "description": "Too many jobs are waiting; retry later"},
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
    *,
    max_pending: int = 0,
    expose_errors: bool = True,
) -> None:
    @app.post(
        "/jobs",
        status_code=202,
        response_model=JobSubmitResponse,
        tags=["Jobs"],
        summary=f"Submit a {controller.model_name} job",
        responses=_SUBMIT_RESPONSES,
    )
    async def submit_job(request: Request, body: request_model):  # type: ignore[valid-type]
        # `body` makes FastAPI validate (422) and document the schema. Store
        # the raw JSON, not a dump of `body`: the worker validates it exactly
        # as /predict does (aliases, validators and strict types all agree).
        payload = await request.json()
        try:
            # Checked and enqueued separately, so a burst of simultaneous submissions can overshoot by a few.
            if max_pending > 0 and await run_in_threadpool(backend.pending_count) >= max_pending:
                raise HTTPException(
                    status_code=429,
                    detail={"error": "The job queue is full; retry later", "error_type": "QueueFull"},
                    headers={"Retry-After": "5"},
                )
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
            error=job.error if expose_errors or job.error is None else "The job failed",
            error_type=job.error_type,
            attempts=job.attempts,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

"""HTTP routes for queue mode."""

from typing import Any, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..types import JobStatusResponse, JobSubmitResponse
from .backend import QueueBackend, QueueUnavailable


def _unavailable(exc: QueueUnavailable) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"error": str(exc), "error_type": "QueueUnavailable"},
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
    )
    async def submit_job(body: request_model):  # type: ignore[valid-type]
        if hasattr(body, "data"):
            data = body.data  # type: ignore[attr-defined]
        else:
            data = body.model_dump(mode="json")  # type: ignore[attr-defined]
        try:
            job_id = await run_in_threadpool(backend.enqueue, data)
        except QueueUnavailable as exc:
            raise _unavailable(exc)
        return JobSubmitResponse(job_id=job_id)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["Jobs"],
        summary="Get a job's status and result",
    )
    async def get_job(job_id: str):
        try:
            job = await run_in_threadpool(backend.get_job, job_id)
        except QueueUnavailable as exc:
            raise _unavailable(exc)
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

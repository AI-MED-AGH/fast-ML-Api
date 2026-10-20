# Redis-backed job queue and workers

Date: 2026-10-08
Status: Draft, awaiting review

## Goal

Support long-running (seconds to minutes) and heavy inference by moving it out of the API process. The API accepts a job and returns immediately. Worker processes, on the same or other machines, consume jobs from Redis, run the model, and store results. Clients submit, then poll.

Success criteria:
- A user writes the same `MLController` subclass as today and starts it as an API or as a worker.
- API processes in queue mode never load the model.
- Workers scale horizontally by starting more processes pointed at the same Redis.
- A worker crash does not lose a job.
- Existing synchronous `/predict` behaviour is unchanged.

## Non-goals (v1)

Priorities, scheduled jobs, webhooks, request batching, job cancellation, auth on job endpoints, hard-killing a running model on timeout. Webhooks and cancellation are the likeliest follow-ups; the backend interface leaves room for them.

## Public API

One class, two roles:

```python
if __name__ == "__main__":
    import sys
    ctrl = MyModel()
    if sys.argv[1:] == ["worker"]:
        ctrl.run_worker(redis_url="redis://...")
    else:
        ctrl.run(queue_url="redis://...")
```

`run()` without `queue_url` behaves exactly as today (sync `/predict`, model loaded in-process).

For programmatic use, `use_queue(backend)` enables queue mode with any `QueueBackend` (e.g. `MemoryBackend` in tests); it must be called before the app is created. `run(queue_url=...)` is shorthand for it with a `RedisBackend`.

### Endpoints in queue mode

| Endpoint | Behaviour |
|---|---|
| `POST /jobs` | Validate body with `request_model` (422 from FastAPI body validation; note that today's `/predict` returns 500 for an invalid body, which is not changed), enqueue, return `202 {job_id}`. `503` if the backend is unreachable. |
| `GET /jobs/{id}` | `{status: queued\|running\|succeeded\|failed, result?, error?, error_type?}`. `404` if unknown or expired. |
| `GET /health` | 200 when the backend is reachable and 503 otherwise, with `queue_connected` and `queue_depth` (`pending` length). `model_loaded` is omitted. |
| `GET /info` | Unchanged. |

`/predict` is not registered in queue mode, so clients get a clear 404 rather than a silently different contract.

### Class attributes (all overridable)

`result_ttl = 3600`, `visibility_timeout = 30`, `max_attempts = 3`, `heartbeat_interval = 5`.

## Components

New package `src/fastmlapi/queue/`, so `controller.py` does not grow further:

- `backend.py`: `QueueBackend` ABC with `enqueue`, `claim`, `heartbeat`, `complete`, `fail`, `get_job`, `requeue_stale`.
- `redis_backend.py`: `RedisBackend`.
- `memory_backend.py`: `MemoryBackend`, thread-safe, single process, for local dev and tests.
- `worker.py`: worker loop, heartbeat thread, signal handling.

`MLController` gains `run_worker()` and the queue-mode branch in `_create_app()`. In queue mode the lifespan skips `load_model()`; `setup()` still runs so users can register routes. Workers run the same `_initialize()` as today.

`redis` is an optional dependency: `pip install fastmlapi[queue]`.

## Redis data model

Keys prefixed `fastmlapi:{model_name}:`.

- `pending`: list of job ids.
- `processing:{worker_id}`: list of job ids claimed by that worker.
- `job:{id}`: hash with `status`, `payload` (the raw JSON request body, stored once `POST /jobs` has validated it; the worker validates it again with `request_model` and extracts the model input the same way `/predict` does), `result` or `error`/`error_type`, `attempts`, `created_at`, `started_at`, `finished_at`, `heartbeat`.

### Lifecycle

1. API: in one pipeline, `HSET job:{id}` (status `queued`) and `LPUSH pending id`.
2. Worker: claim with a non-blocking atomic Lua script: `RPOP pending`, `LPUSH processing:{worker_id}`, set `status=running`, `attempts += 1` and the heartbeat, all in one step. Workers poll every 0.5 s when the queue is empty (no `BLMOVE`, so no Redis >= 6.2 requirement).
3. Worker: a background thread refreshes `heartbeat` every `heartbeat_interval` seconds, independent of the inference thread.
4. Success: write `result` as JSON, `status=succeeded`, remove from `processing`, set TTL `result_ttl` on `job:{id}`.
5. Exception: record `error`/`error_type`, `status=failed`, remove from `processing`, set TTL. No retry, since model errors are normally deterministic.
6. Non-JSON-serializable result: job fails with an explicit serialization error.

### Crash recovery

Workers (not the API) periodically run `requeue_stale`, between jobs, at the start of a claim attempt and at most every `sweep_interval` (default `max(visibility_timeout/2, 1)` s). Jobs of dead workers are therefore recovered as soon as any worker is between jobs, or starts: any job in a `processing:*` list whose heartbeat is older than `visibility_timeout` returns to `pending`. After `max_attempts` it is marked `failed` with a "worker lost" error. Runtime length does not matter, only heartbeat freshness, so a 20-minute job on a live worker is safe.

Completion, failure and heartbeat are fenced by worker id: a worker that lost its job (because it was requeued and claimed elsewhere) cannot overwrite the new owner's state.

Delivery is at-least-once: a worker that dies after finishing but before acking can cause a second run. This is documented.

## Worker behaviour

- Startup failure in `setup()` or `load_model()`: exit non-zero so the orchestrator restarts it. Never consume jobs with a broken model.
- Redis outage: retry with exponential backoff; an in-flight job finishes and its result write is retried.
- `SIGTERM`/`SIGINT`: finish the current job, then exit. A hard kill is covered by crash recovery.
- One job at a time per process. Concurrency comes from more worker processes.
- No hard `job_timeout` in v1. Python cannot safely interrupt a running model; this is documented as a limitation.

## Testing

- `MemoryBackend` unit tests: enqueue/claim/complete/fail, stale requeue, max-attempts exhaustion, TTL expiry.
- API tests with `httpx` + `MemoryBackend`: 202 on submit, 422 on bad body, 404 on unknown id, full submit-run-poll cycle with an in-process worker thread, `/predict` absent in queue mode, model not loaded in API process.
- Worker tests: model exception becomes `failed`, non-serializable result becomes `failed`, startup failure exits non-zero, graceful shutdown finishes the current job.
- Redis backend tests: run against a real Redis when `REDIS_URL` is set, skipped otherwise; CI gets a Redis service container. Includes a simulated worker death (stop heartbeat, verify requeue and attempt counting) and atomic-claim check with two concurrent claimers.
- Regression: all existing tests pass unchanged, confirming sync `/predict` is untouched.

## Docs

README section "Long-running jobs": enabling queue mode, running API and worker, endpoint contract, tuning attributes, at-least-once and no-hard-timeout caveats. CHANGELOG entry. An example under `examples/`.

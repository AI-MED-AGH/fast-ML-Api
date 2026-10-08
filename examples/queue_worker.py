"""
Queue mode example: one file, started as an API or as a worker.

    pip install fastmlapi[queue]
    docker run --rm -p 6379:6379 redis:7

    python examples/queue_worker.py            # API on :8000 (no model loaded)
    python examples/queue_worker.py worker     # worker (start as many as you like)

    curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
         -d '{"data": {"n": 21}}'               # -> {"job_id": "..."}
    curl localhost:8000/jobs/<job_id>           # queued -> running -> succeeded
"""

import os
import sys
import time

from fastmlapi import MLController, postprocessing, prediction, preprocessing

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class SlowDoubler(MLController):
    model_name = "slow-doubler"

    @preprocessing
    def preprocess(self, data: dict) -> int:
        return data["n"]

    @prediction
    def infer(self, n: int) -> int:
        time.sleep(10)  # stand in for heavy inference
        return n * 2

    @postprocessing
    def postprocess(self, out: int) -> dict:
        return {"doubled": out}


if __name__ == "__main__":
    controller = SlowDoubler()
    if sys.argv[1:] == ["worker"]:
        controller.run_worker(redis_url=REDIS_URL)
    else:
        controller.run(queue_url=REDIS_URL)

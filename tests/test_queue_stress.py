"""Concurrency stress for the Lua-free Redis backend (opt in with REDIS_URL)."""
import os
import random
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")


def test_many_workers_and_an_aggressive_sweeper_finish_every_job_exactly_once():
    from fastmlapi.queue.redis_backend import RedisBackend

    prefix = f"fastmlapi-test:{uuid.uuid4().hex}:"
    backend = RedisBackend(os.environ["REDIS_URL"], prefix=prefix, result_ttl=300)
    jobs = {backend.enqueue({"n": i}): i for i in range(150)}
    accepted = {}           # job id -> how many complete() calls were accepted (must end at exactly 1)
    lock = threading.Lock()
    stop = threading.Event()
    errors = []

    def worker(name):
        b = RedisBackend(os.environ["REDIS_URL"], prefix=prefix, result_ttl=300)
        try:
            while not stop.is_set():
                job = b.claim(name)
                if job is None:
                    time.sleep(0.005)
                    continue
                time.sleep(random.uniform(0.0, 0.06))              # "inference"; no heartbeat, so the sweeper may steal it
                if b.complete(name, job.id, {"by": name, "n": job.payload["n"]}):
                    with lock:
                        accepted[job.id] = accepted.get(job.id, 0) + 1
        except Exception as exc:                                    # pragma: no cover
            errors.append(exc)

    def sweeper():
        b = RedisBackend(os.environ["REDIS_URL"], prefix=prefix, result_ttl=300)
        try:
            while not stop.is_set():
                b.requeue_stale(0.02, 1000)                         # steals any job running for more than 20 ms
                time.sleep(0.01)
        except Exception as exc:                                    # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
    threads += [threading.Thread(target=sweeper) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            with lock:
                if len(accepted) == len(jobs):
                    break
            time.sleep(0.05)
    finally:
        stop.set()
        for t in threads:
            t.join(10)

    assert not errors, errors
    assert set(accepted) == set(jobs), "some jobs never finished"
    assert all(count == 1 for count in accepted.values()), "a job was completed more than once"
    for job_id, n in jobs.items():
        job = backend.get_job(job_id)
        assert job.status == "succeeded" and job.result["n"] == n   # right result, belonging to the right job
        assert job.payload is None
    assert backend.pending_count() == 0
    for key in list(backend._r.scan_iter(match=prefix + "*")):
        backend._r.delete(key)

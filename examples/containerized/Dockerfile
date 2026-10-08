# Example Dockerfile for a fastmlapi service. Build from the repository root:
#   docker build -f examples/containerized/Dockerfile -t fastmlapi-example .
FROM python:3.11-slim

# --- GPU variant: swap the base image above for a CUDA runtime image and install
# --- python yourself, e.g.:
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
#     && rm -rf /var/lib/apt/lists/*
# Then install a CUDA build of torch (or your framework) in requirements. Run the
# container with GPU access (e.g. `docker run --gpus all`).

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTMLAPI_CACHE_DIR=/cache

# Non-root user; the cache directory is a volume so downloads persist across restarts.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /cache && chown app:app /cache
VOLUME /cache

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY examples/containerized/ml_server.py ./ml_server.py

USER app
EXPOSE 8000

# Readiness check. Model loading can be slow: give it a long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Exec form so Python is PID 1 and receives SIGTERM for graceful shutdown.
CMD ["python", "ml_server.py"]

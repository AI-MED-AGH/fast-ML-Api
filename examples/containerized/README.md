# Containerized fastmlapi example

A generic example of running a fastmlapi service in a container behind an orchestrator or
gateway. Nothing here is specific to a platform or registry.

```bash
docker build -f examples/containerized/Dockerfile -t fastmlapi-example .
docker run --rm -p 8000:8000 -v fastmlapi-cache:/cache fastmlapi-example
curl localhost:8000/schema
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"values": [1, 2, 3]}'
```

## Container contract

| Item | Value |
|------|-------|
| Port | `8000` (override with `FASTMLAPI_PORT`; bind host with `FASTMLAPI_HOST`) |
| `GET /health` | Readiness. `200` once the model is loaded, `503` before |
| `GET /livez` | Liveness. `200` whenever the process is up |
| `POST /predict` | Inference. Body follows the schema from `/schema` |
| `GET /schema` | JSON Schema of request/response plus examples |
| `FASTMLAPI_CACHE_DIR` | Directory for downloaded weights; mount a persistent volume (`/cache` here) |
| Shutdown | `SIGTERM` triggers a graceful Uvicorn shutdown (exit status 143) |

## Probes

Model loading happens before the port opens, so a slow load looks like "connection refused".
Use a startup probe that tolerates it, then liveness and readiness probes:

```yaml
startupProbe:   { httpGet: { path: /health, port: 8000 }, periodSeconds: 10, failureThreshold: 60 }
livenessProbe:  { httpGet: { path: /livez,  port: 8000 }, periodSeconds: 15 }
readinessProbe: { httpGet: { path: /health, port: 8000 }, periodSeconds: 10 }
```

## Weights: bake in or download?

Baking weights into the image gives fast, offline, reproducible starts at the price of large
images. Downloading into `FASTMLAPI_CACHE_DIR` keeps images small and lets weights change
independently, but the first start is slow and needs network access; the volume must be
persistent. See the main README for the Hugging Face (`HF_HOME`) pattern.

## GPU

Use the commented CUDA base image in the `Dockerfile`, install a CUDA build of your
framework, and run with GPU access. `gpu_memory_fraction` / `FASTMLAPI_GPU_MEMORY_FRACTION`
optionally caps PyTorch memory (best effort).

## CI example

`build-and-push.yml.example` is an **example only**: copy it to `.github/workflows/`, rename it
to `.yml`, and adjust the registry and image name for your setup.

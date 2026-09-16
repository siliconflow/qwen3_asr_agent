# Qwen3-ASR Agent

OpenAI-compatible ASR proxy sidecar for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) served by vLLM.

This project extracts the proxy sidecar program out of the `Qwen3-ASR` GPU Function template
(`siliconflow/ready-to-use/qwen3-asr`) into a standalone, version-controlled repository, mirroring
the `siliconflow/vllm_agent` / `siliconflow/sglang_agent` pattern. The template consumes the
prebuilt image published from this repository; the sidecar source no longer lives inline inside
the template's `command:` block.

## Components

| File | Purpose |
|------|---------|
| `app.py` | FastAPI sidecar (port `80` by default, overridable via `PORT`) proxying `/v1/audio/transcriptions` to vLLM (`127.0.0.1:8080`). Cleans the Qwen3-ASR output format (`language X<asr_text>...` → plain text), repairs repetition hallucinations, applies the multi-language strategy, converts audio to 16 kHz mono WAV, exposes Prometheus metrics, and drains in-flight requests on SIGTERM. |
| `Dockerfile` | `python:3.10-slim` image; no build-time GPU/torch dependency. |
| `requirements.txt` | Runtime dependencies (fastapi / uvicorn / httpx / prometheus-client / av / python-multipart). |
| `VERSION` | Semantic version used as the image tag suffix. |
| `test_proxy.py` | Local smoke test against a running proxy + vLLM. |
| `build-image.sh` | Local build + push helper (personal namespace by default). |

## Endpoints

| Path | Purpose |
|------|---------|
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription (multipart `file` upload). |
| `GET /health` | Liveness — proxy process only. |
| `GET /ready` | Readiness — cascades to vLLM `/ping` with a cooldown window. |
| `GET /metrics` | Prometheus metrics. |
| `ANY /{path}` | Catch-all pass-through to vLLM. |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_HOST` | `127.0.0.1` | vLLM host (pod-shared network namespace). |
| `VLLM_PORT` | `8080` | vLLM port. |
| `PORT` | `80` | Sidecar listen port; must match the template's `entrypointPort`. |
| `CLEAN_OUTPUT` | `true` | Strip `language X<asr_text>` prefixes from the transcript. |
| `ENABLE_REPETITION_FIX` | `true` | Enable repetition repair. |
| `REPETITION_THRESHOLD` | `20` | Repetition count that triggers repair. |
| `MULTILANG_STRATEGY` | `major` | `first` \| `major` \| `mixed` — how `language` is picked for multi-language audio. |
| `MAX_FILE_SIZE_MB` | `500` | Max request body size. |
| `READY_INFERENCE_COOLDOWN` | `60` | Seconds between readiness cascade probes to vLLM. |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | SIGTERM drain timeout; keep below the pod's `terminationGracePeriodSeconds`. |

## Quick start

```bash
pip install -r requirements.txt
python app.py            # proxy on :80, expects vLLM on 127.0.0.1:8080
python test_proxy.py     # smoke test (needs a reachable proxy + vLLM)
```

## Docker

```bash
docker build -t qwen3-asr-proxy .

# Local build + push (bumps nothing; fails if the tag already exists)
./build-image.sh

# Build for the shared registry namespace used by the templates (CI does this)
```

## Image publishing

`.github/workflows/docker-build.yml` builds and pushes on push to `main` /
`feat/sf-gpu-function-adaptation` and on manual dispatch:

```
hub.6scloud.com/d1r7umcsfi9c73b4drdg/qwen3-asr-proxy:{YYYYMMDD}-v{VERSION}
```

Tag rule: `{YYYYMMDD}-v{SEMVER}` from `VERSION` (no GPU suffix — the image is pure Python and
architecture-independent). The tag format and image name are intentionally kept identical to the
image previously built from `Qwen3-ASR/proxy/`, so already-published tags such as
`20260914-v1.0.0` remain valid and the template needs no image rename.

## Template wiring

`siliconflow/ready-to-use/qwen3-asr/` in the templates repository carries this repository as a git
submodule at `qwen3_asr_agent/`, and runs it as a sidecar container next to the vLLM main
container:

```
Gateway :80 -> [qwen3-asr-proxy :80] -> [vLLM :8080]
```

The submodule gitlink records the exact sidecar revision the pinned image tag was built from.

## Relationship to the Qwen3-ASR fork

`app.py` is the same program as `Qwen3-ASR/proxy/app.py` (branch `feat/sf-gpu-function-adaptation`).
Once this repository is the published source of the sidecar image, the copy inside the fork should
be retired (or turned into a pointer README) to avoid two drifting sources.

# Qwen3-ASR API Proxy (sidecar)
# Lightweight proxy to clean vLLM transcription responses.
# Runs in the same pod as vLLM via sidecarContainers; talks to vLLM over
# 127.0.0.1:8080 (pod-level network namespace).

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
# - av (PyAV): mp3/m4a/aac/wma/opus/webm -> WAV (16kHz mono s16) conversion,
#   ships bundled ffmpeg libs via manylinux wheels (no system ffmpeg needed)
# - python-multipart: fastapi form parsing + multipart.body rewrite
#   (app.py uses `from multipart.multipart import parse_options_header`)
# - prometheus-client: /metrics endpoint
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.32.0 \
    httpx==0.28.0 \
    "av>=10.0,<18" \
    python-multipart==0.0.20 \
    prometheus-client>=0.20.0

# Copy proxy application
COPY app.py .

# Environment defaults
ENV VLLM_HOST=127.0.0.1
ENV VLLM_PORT=8080
ENV CLEAN_OUTPUT=true
ENV PORT=80

EXPOSE 80

# Run proxy
CMD ["python", "app.py"]

#!/usr/bin/env python3
"""
Qwen3-ASR API Proxy

Cleans vLLM transcription responses by parsing the Qwen3-ASR output format.
Transforms: "language Cantonese<asr_text>转录文本" -> "转录文本"

Features:
- Output format cleaning (removes language tags and <asr_text>)
- Intelligent repetition detection (fixes model hallucinations, preserves normal speech)
- Configurable via environment variables
"""

import io
import os
import json
import logging
import signal
import asyncio
import time
from typing import Optional, Tuple
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = os.environ.get("VLLM_PORT", "8080")
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
CLEAN_OUTPUT = os.environ.get("CLEAN_OUTPUT", "true").lower() == "true"
ENABLE_REPETITION_FIX = os.environ.get("ENABLE_REPETITION_FIX", "true").lower() == "true"
REPETITION_THRESHOLD = int(os.environ.get("REPETITION_THRESHOLD", "20"))
MULTILANG_STRATEGY = os.environ.get("MULTILANG_STRATEGY", "major").lower()  # "first" | "major" | "mixed"
MAX_FILE_SIZE_MB = float(os.environ.get("MAX_FILE_SIZE_MB", "500"))  # Max request body size in MB
READY_INFERENCE_COOLDOWN = int(os.environ.get("READY_INFERENCE_COOLDOWN", "60"))  # 级联检查冷却时间（秒）
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))  # 优雅退出等待超时（秒）

# Readiness probe state
_last_ready_check_time = 0.0
_last_ready_check_result = False

# Graceful shutdown state
_is_shutting_down = False
_active_requests = 0
_shutdown_event = asyncio.Event()

# Prometheus metrics
if _PROMETHEUS_AVAILABLE:
    ASR_REQUESTS_TOTAL = Counter(
        "asr_requests_total", "Total ASR requests", ["endpoint", "status"]
    )
    ASR_REQUEST_DURATION = Histogram(
        "asr_request_duration_seconds", "ASR request duration", ["endpoint"]
    )
    ASR_ACTIVE_REQUESTS = Gauge(
        "asr_active_requests", "Current active requests"
    )
    ASR_VLLM_UP = Gauge(
        "asr_vllm_up", "vLLM backend reachable (1=up, 0=down)"
    )

# ASR output parsing
ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    """Normalize language name: first letter uppercase, rest lowercase."""
    if language is None:
        return ""
    s = str(language).strip()
    if not s:
        return ""
    return s[:1].upper() + s[1:].lower()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Detect and fix abnormal repetitions in text.

    Only triggers when repetitions exceed threshold (default 20).
    This preserves normal speech patterns while removing model hallucinations.

    Args:
        text: Input text
        threshold: Minimum repetitions to trigger fix (default 20)

    Returns:
        Text with excessive repetitions compressed
    """
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break

                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
    """
    Parse Qwen3-ASR raw output into (language, text).

    Handles multiple occurrences of "language xxx<asr_text>" pattern with
    improved multilingual support.

    Args:
        raw: Raw decoded string from model.
        user_language: Optional forced language name.

    Returns:
        Tuple of (language, cleaned_text)
    """
    if raw is None:
        return "", ""

    s = str(raw).strip()
    if not s:
        return "", ""

    # Apply repetition fix if enabled
    if ENABLE_REPETITION_FIX:
        s = detect_and_fix_repetitions(s, threshold=REPETITION_THRESHOLD)

    # If user forced language, treat output as plain text
    if user_language:
        return user_language, s

    # Check for ASR text tag
    if ASR_TEXT_TAG not in s:
        return "", s

    # Extract ALL language tags for better multilingual handling
    import re
    from collections import Counter

    lang_pattern = re.compile(r'language\s+(\w+)<asr_text>')
    all_langs = lang_pattern.findall(s)

    # Determine final language based on strategy
    final_lang = ""

    if not all_langs:
        # No language tags found
        final_lang = ""
    elif len(all_langs) == 1:
        # Single language occurrence
        final_lang = normalize_language_name(all_langs[0])
    else:
        # Multiple language tags detected
        # Check for "none" language
        if any(lang.lower() == "none" for lang in all_langs):
            final_lang = ""
        elif MULTILANG_STRATEGY == "first":
            # Legacy: return first language
            final_lang = normalize_language_name(all_langs[0])
        elif MULTILANG_STRATEGY == "major":
            # Return most frequent language (default)
            lang_counts = Counter(normalize_language_name(lang) for lang in all_langs)
            major_lang, major_count = lang_counts.most_common(1)[0]
            final_lang = major_lang
        elif MULTILANG_STRATEGY == "mixed":
            # Return "Mixed" if multiple distinct languages
            unique_langs = set(normalize_language_name(lang) for lang in all_langs)
            if len(unique_langs) > 1:
                final_lang = "Mixed"
            else:
                final_lang = normalize_language_name(all_langs[0])
        else:
            # Default to first
            final_lang = normalize_language_name(all_langs[0])

    # Remove ALL "language xxx<asr_text>" patterns from the text
    cleaned_text = re.sub(r'language\s+\w+<asr_text>', '', s).strip()

    return final_lang, cleaned_text


def clean_transcription_response(data: dict) -> dict:
    """
    Clean transcription response data.
    """
    if not isinstance(data, dict):
        return data

    # Handle standard transcription response
    if "text" in data:
        raw_text = data.get("text", "")
        lang, clean_text = parse_asr_output(raw_text)

        # Update response
        result = dict(data)
        result["text"] = clean_text
        # Always set language, override any existing value
        if lang:
            result["language"] = lang
        return result

    return data


def clean_json_response(data: dict) -> dict:
    """
    Recursively clean JSON response.
    """
    if isinstance(data, dict):
        # Check if this is a transcription response
        if "text" in data and isinstance(data.get("text"), str):
            return clean_transcription_response(data)

        # Recursively process nested dicts
        return {k: clean_json_response(v) for k, v in data.items()}

    elif isinstance(data, list):
        return [clean_json_response(item) for item in data]

    return data


# Formats that soundfile cannot decode but ffmpeg/av can
_NEEDS_CONVERSION = {".mp3", ".m4a", ".aac", ".wma", ".opus", ".webm"}


def convert_to_wav(data: bytes, filename: str) -> bytes:
    """Convert audio bytes to 16kHz mono 16-bit PCM WAV using PyAV.

    ASR models expect 16kHz mono audio. This keeps file sizes reasonable
    and ensures compatibility with vLLM's audio endpoint.

    File size calculation:
    16kHz * 16-bit * 1 channel = 32,000 bytes/second = ~31.25 KB/s
    A 3-minute phone call: ~5.6 MB (vs ~60 MB for uncompressed 44.1kHz stereo)
    """
    import av
    in_buf = io.BytesIO(data)
    out_buf = io.BytesIO()

    # Create resampler: convert to 16kHz mono 16-bit PCM
    resampler = av.AudioResampler(
        format="s16",      # 16-bit PCM
        layout="mono",     # Single channel
        rate=16000         # 16kHz sample rate
    )

    with av.open(in_buf, "r") as in_container:
        in_stream = in_container.streams.audio[0]
        logger.info(f"Input audio: {in_stream.sample_rate}Hz, {in_stream.channels} channels, {in_stream.layout}")

        with av.open(out_buf, "w", format="wav") as out_container:
            # Add PCM stream with explicit 16kHz mono settings
            # Try layout parameter first (PyAV >= 10.0)
            try:
                out_stream = out_container.add_stream("pcm_s16le", rate=16000, layout="mono")
                logger.info(f"Output stream created with layout='mono'")
            except (TypeError, ValueError) as e:
                # Fallback for older PyAV versions
                logger.warning(f"layout parameter not supported: {e}, using fallback")
                out_stream = out_container.add_stream("pcm_s16le", rate=16000)
                out_stream.options = {'channels': '1'}

            for frame in in_container.decode(in_stream):
                # Resample to 16kHz mono 16-bit PCM
                for resampled_frame in resampler.resample(frame):
                    if resampled_frame is not None:
                        for packet in out_stream.encode(resampled_frame):
                            out_container.mux(packet)

            # Flush resampler
            for resampled_frame in resampler.resample(None):
                if resampled_frame is not None:
                    for packet in out_stream.encode(resampled_frame):
                        out_container.mux(packet)

            # Flush encoder
            for packet in out_stream.encode(None):
                out_container.mux(packet)

    result = out_buf.getvalue()
    logger.info(f"Converted WAV: {len(result)} bytes ({len(result) / 1024 / 1024:.2f} MB)")
    return result


# HTTP client for proxying
client: Optional[httpx.AsyncClient] = None


def handle_sigterm(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _is_shutting_down
    _is_shutting_down = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown (timeout={GRACEFUL_SHUTDOWN_TIMEOUT}s)")
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(wait_for_requests_complete())


async def wait_for_requests_complete():
    """Wait for active requests to finish or timeout."""
    waited = 0
    while _active_requests > 0 and waited < GRACEFUL_SHUTDOWN_TIMEOUT:
        logger.info(f"Graceful shutdown: waiting for {_active_requests} active request(s)...")
        await asyncio.sleep(1)
        waited += 1
    if _active_requests > 0:
        logger.warning(f"Graceful shutdown: timeout reached with {_active_requests} request(s) still active")
    else:
        logger.info("Graceful shutdown: all requests completed")
    _shutdown_event.set()


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    logger.info(f"Proxy started, forwarding to {VLLM_URL}")
    logger.info(f"Clean output: {CLEAN_OUTPUT}")
    logger.info(f"Repetition fix: {ENABLE_REPETITION_FIX} (threshold={REPETITION_THRESHOLD})")
    logger.info(f"Multilang strategy: {MULTILANG_STRATEGY}")
    logger.info(f"Graceful shutdown timeout: {GRACEFUL_SHUTDOWN_TIMEOUT}s")
    yield
    # Graceful shutdown: wait for requests to complete
    if _active_requests > 0:
        logger.info(f"Shutting down, waiting for {_active_requests} active request(s)...")
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout reached")
    if client is not None:
        await client.aclose()
    logger.info("Proxy stopped")


app = FastAPI(title="Qwen3-ASR Proxy", lifespan=lifespan)


@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """Reject new requests during shutdown and track active requests."""
    if _is_shutting_down and request.url.path not in ("/health", "/ready", "/metrics"):
        return JSONResponse(
            content={"error": "Service is shutting down"},
            status_code=503,
        )
    global _active_requests
    _active_requests += 1
    if _PROMETHEUS_AVAILABLE:
        ASR_ACTIVE_REQUESTS.set(_active_requests)
    start = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        _active_requests -= 1
        if _PROMETHEUS_AVAILABLE:
            ASR_ACTIVE_REQUESTS.set(_active_requests)
            duration = time.time() - start
            ASR_REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
            ASR_REQUESTS_TOTAL.labels(endpoint=request.url.path, status=str(status_code)).inc()


@app.get("/health")
async def health():
    """Liveness check - returns proxy status only."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "prometheus-client not installed"},
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/ready")
async def ready():
    """Readiness check - verifies proxy + vLLM chain (with cooldown)."""
    global _last_ready_check_time, _last_ready_check_result
    errors = []
    current_time = time.time()

    # 1. Proxy self-check
    if client is None:
        errors.append("proxy_client_not_initialized")

    # 2. Cascaded vLLM check (with cooldown to avoid frequent pings)
    vllm_ready = False
    if client is not None:
        if (current_time - _last_ready_check_time) < READY_INFERENCE_COOLDOWN and _last_ready_check_result:
            # Within cooldown and last check passed; skip ping
            vllm_ready = True
            logger.debug("/ready: skipping vLLM ping, within cooldown")
        else:
            try:
                resp = await client.get(f"{VLLM_URL}/ping", timeout=5.0)
                if resp.status_code == 200:
                    vllm_ready = True
                    _last_ready_check_time = current_time
                    _last_ready_check_result = True
                else:
                    errors.append(f"vllm_not_ready: status {resp.status_code}")
                    _last_ready_check_result = False
            except Exception as e:
                errors.append(f"vllm_unreachable: {str(e)}")
                _last_ready_check_result = False

    if _PROMETHEUS_AVAILABLE:
        ASR_VLLM_UP.set(1 if vllm_ready else 0)

    if errors:
        logger.warning(f"/ready check failed: {errors}")
        return Response(
            content=json.dumps({"status": "not_ready", "errors": errors}),
            media_type="application/json",
            status_code=503,
        )

    resp = {"status": "ready", "vllm": "up"}
    if vllm_ready and (current_time - _last_ready_check_time) <= READY_INFERENCE_COOLDOWN:
        resp["inference_test"] = {"performed": False, "reason": "cooldown"}
    return resp


async def maybe_convert_audio(body: bytes, headers: dict) -> tuple:
    """
    Parse multipart body, convert any unsupported audio format to WAV in-place.
    Operates at raw bytes level to preserve boundary format exactly.
    Returns (new_body, new_headers). No-op if format is already supported.

    Key fix: Only modify multipart data when conversion is needed.
    Otherwise, forward the original body unchanged to preserve multipart structure.
    """
    from multipart.multipart import parse_options_header

    content_type = headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return body, headers

    _, params = parse_options_header(content_type)
    boundary = params.get(b"boundary")
    if not boundary:
        return body, headers

    delim = b"--" + boundary

    # First pass: check if any conversion is needed
    converted = False
    parts = body.split(delim)

    for part in parts[1:-1]:
        if not part.startswith(b"\r\n"):
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_bytes = part[2:header_end]
        fname = ""
        for line in header_bytes.split(b"\r\n"):
            if b"Content-Disposition" in line and b'name="file"' in line:
                for token in line.split(b";"):
                    token = token.strip()
                    if token.startswith(b"filename="):
                        fname = token[9:].strip().strip(b'"').decode(errors="replace")
        ext = os.path.splitext(fname)[1].lower()
        if ext in _NEEDS_CONVERSION:
            converted = True
            break

    # If no conversion needed, forward original body unchanged
    if not converted:
        new_headers = dict(headers)
        new_headers.pop("content-length", None)
        return body, new_headers

    # Second pass: do the conversion
    new_parts = [parts[0]]
    for part in parts[1:-1]:
        if not part.startswith(b"\r\n"):
            new_parts.append(part)
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            new_parts.append(part)
            continue
        header_bytes = part[2:header_end]
        payload = part[header_end + 4:]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
            trailing = b"\r\n"
        else:
            trailing = b""

        fname = ""
        for line in header_bytes.split(b"\r\n"):
            if b"Content-Disposition" in line and b'name="file"' in line:
                for token in line.split(b";"):
                    token = token.strip()
                    if token.startswith(b"filename="):
                        fname = token[9:].strip().strip(b'"').decode(errors="replace")
        ext = os.path.splitext(fname)[1].lower()
        if ext in _NEEDS_CONVERSION:
            try:
                wav_data = convert_to_wav(payload, fname)
                new_fname = os.path.splitext(fname)[0] + ".wav"
                new_header_bytes = header_bytes.replace(
                    f'filename="{fname}"'.encode(),
                    f'filename="{new_fname}"'.encode(),
                )
                new_header_lines = []
                ct_replaced = False
                for line in new_header_bytes.split(b"\r\n"):
                    if line.lower().startswith(b"content-type:"):
                        new_header_lines.append(b"Content-Type: audio/wav")
                        ct_replaced = True
                    else:
                        new_header_lines.append(line)
                if not ct_replaced:
                    new_header_lines.append(b"Content-Type: audio/wav")
                new_header_bytes = b"\r\n".join(new_header_lines)
                part = b"\r\n" + new_header_bytes + b"\r\n\r\n" + wav_data + trailing
                converted = True
                logger.info(f"Converted {fname} -> {new_fname} ({len(payload)} -> {len(wav_data)} bytes)")
            except Exception as e:
                logger.warning(f"Audio conversion failed for {fname}: {e}, forwarding as-is")
                part = b"\r\n" + header_bytes + b"\r\n\r\n" + payload + trailing
        else:
            part = b"\r\n" + header_bytes + b"\r\n\r\n" + payload + trailing
        new_parts.append(part)

    # Preserve the trailing boundary marker (--boundary--)
    if len(parts) > 1:
        new_parts.append(parts[-1])

    new_body = delim.join(new_parts)
    new_headers = dict(headers)
    new_headers.pop("content-length", None)
    return new_body, new_headers


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_request(request: Request, path: str):
    """
    Proxy all requests to vLLM backend with response cleaning.
    """
    # Build target URL
    target_url = f"{VLLM_URL}/{path}"

    # Forward query parameters
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Get request body
    body = await request.body()

    # Forward headers (skip host and transfer-encoding)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("transfer-encoding", None)  # Must remove - httpx will set Content-Length

    # Convert unsupported audio formats for transcription endpoints
    if path.rstrip("/") in ("v1/audio/transcriptions", "v1/audio/translations"):
        # Check request body size
        size_mb = len(body) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return Response(
                content=json.dumps({"error": f"Request body too large: {size_mb:.2f}MB (max {MAX_FILE_SIZE_MB}MB)"}),
                media_type="application/json",
                status_code=413,
            )
        body, headers = await maybe_convert_audio(body, headers)

    # Make request to vLLM
    try:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        return Response(content=f"Proxy error: {e}", status_code=502)

    # Check if we should clean the response
    content_type = response.headers.get("content-type", "")

    # Headers safe to forward (exclude hop-by-hop and content-length which may be stale)
    _HOP_BY_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive",
                   "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"}

    def safe_headers(h):
        return {k: v for k, v in h.items() if k.lower() not in _HOP_BY_HOP}

    # Handle streaming responses
    if "text/event-stream" in content_type or "application/x-ndjson" in content_type:
        return StreamingResponse(
            clean_streaming_response(response),
            media_type=content_type,
            headers=safe_headers(response.headers),
        )

    # Handle JSON responses
    if "application/json" in content_type and CLEAN_OUTPUT:
        try:
            data = response.json()
            cleaned = clean_json_response(data)
            import json
            return Response(
                content=json.dumps(cleaned, ensure_ascii=False),
                media_type="application/json",
                status_code=response.status_code,
            )
        except Exception as e:
            logger.warning(f"Failed to parse/clean JSON: {e}")

    # Return response as-is
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=safe_headers(response.headers),
    )


async def clean_streaming_response(response: httpx.Response):
    """
    Clean streaming (SSE) response.
    """
    import json

    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk

        # Process complete SSE events
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            if not event.strip():
                continue

            # Parse SSE event
            lines = event.strip().split("\n")
            event_data = {}
            for line in lines:
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        event_data["data"] = "[DONE]"
                    else:
                        try:
                            event_data["data"] = json.loads(data_str)
                        except json.JSONDecodeError:
                            event_data["data"] = data_str

            # Clean if it's a transcription response
            if isinstance(event_data.get("data"), dict):
                event_data["data"] = clean_json_response(event_data["data"])

            # Rebuild SSE event
            if event_data.get("data") == "[DONE]":
                yield "data: [DONE]\n\n"
            else:
                yield f"data: {json.dumps(event_data['data'], ensure_ascii=False)}\n\n"

    # Handle remaining buffer
    if buffer.strip():
        yield buffer


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)

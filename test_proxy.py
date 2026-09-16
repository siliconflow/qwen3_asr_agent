#!/usr/bin/env python3
"""
Local test script for the Qwen3-ASR proxy sidecar.

Usage:
  1. Start vLLM server locally:
     qwen-asr-serve Qwen/Qwen3-ASR-1.7B --port 8080

  2. Run the proxy:
     python app.py

  3. Run this test:
     python test_proxy.py
"""

import requests
import json

PROXY_URL = "http://127.0.0.1:80"
VLLM_URL = "http://127.0.0.1:8080"


def test_health():
    """Test health endpoint."""
    resp = requests.get(f"{PROXY_URL}/health")
    print(f"Health check: {resp.json()}")


def test_transcription():
    """Test transcription endpoint with a sample audio."""

    # Use a sample audio URL or local file
    audio_url = "https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B/resolve/master/examples/cantonese.wav"

    # Test via proxy
    print("\n=== Testing via Proxy ===")
    resp = requests.post(
        f"{PROXY_URL}/v1/audio/transcriptions",
        data={"model": "Qwen/Qwen3-ASR-1.7B"},
        files={"file": ("audio.wav", requests.get(audio_url).content)},
    )
    proxy_result = resp.json()
    print(f"Proxy response: {json.dumps(proxy_result, indent=2, ensure_ascii=False)}")

    # Test direct vLLM (for comparison)
    print("\n=== Testing Direct vLLM ===")
    resp = requests.post(
        f"{VLLM_URL}/v1/audio/transcriptions",
        data={"model": "Qwen/Qwen3-ASR-1.7B"},
        files={"file": ("audio.wav", requests.get(audio_url).content)},
    )
    vllm_result = resp.json()
    print(f"vLLM response: {json.dumps(vllm_result, indent=2, ensure_ascii=False)}")

    # Compare text fields
    print("\n=== Comparison ===")
    print(f"vLLM text:   {vllm_result.get('text')}")
    print(f"Proxy text:  {proxy_result.get('text')}")


if __name__ == "__main__":
    test_health()
    test_transcription()

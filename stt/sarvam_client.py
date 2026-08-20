"""Sarvam AI speech-to-text client.

Chosen over ElevenLabs specifically for Indic-language coverage, since the indexed
corpus (MSMARCO-XI) is Indian-language text. Requires SARVAM_API_KEY in the
environment; see https://docs.sarvam.ai for account setup.

Network calls are wrapped with bounded retries (tenacity) and a timeout; on repeated
failure the harness gets a `used_fallback=True` result with an empty transcript so it
can short-circuit to "please try again" rather than crash or silently send an empty
string into retrieval.
"""
from __future__ import annotations

import os
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from harness.schemas import STTResult

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saarika:v2")
REQUEST_TIMEOUT_S = float(os.environ.get("SARVAM_TIMEOUT_S", "15"))


class SarvamSTTError(Exception):
    pass


class SarvamSTTClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TransportError, SarvamSTTError)),
        reraise=True,
    )
    def _call_api(self, audio_bytes: bytes, filename: str, language_code: str | None) -> dict:
        if not self.api_key:
            raise SarvamSTTError("SARVAM_API_KEY not set")

        headers = {"api-subscription-key": self.api_key}
        data = {"model": DEFAULT_MODEL}
        if language_code:
            data["language_code"] = language_code

        files = {"file": (filename, audio_bytes, "audio/wav")}
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            resp = client.post(SARVAM_STT_URL, headers=headers, data=data, files=files)
        if resp.status_code >= 500:
            raise SarvamSTTError(f"Sarvam server error {resp.status_code}")
        if resp.status_code != 200:
            raise SarvamSTTError(f"Sarvam API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", language_code: str | None = None) -> STTResult:
        t0 = time.perf_counter()
        try:
            payload = self._call_api(audio_bytes, filename, language_code)
            transcript = payload.get("transcript", "") or ""
            detected_lang = payload.get("language_code") or language_code
            return STTResult(
                transcript=transcript.strip(),
                language=detected_lang,
                confidence=payload.get("confidence"),
                provider="sarvam",
                latency_ms=(time.perf_counter() - t0) * 1000,
                used_fallback=False,
            )
        except Exception:  # noqa: BLE001 - deliberate: any failure -> structured fallback
            return STTResult(
                transcript="",
                language=language_code,
                provider="sarvam",
                latency_ms=(time.perf_counter() - t0) * 1000,
                used_fallback=True,
            )

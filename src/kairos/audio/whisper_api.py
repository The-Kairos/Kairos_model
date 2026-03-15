"""Whisper API client: lazy singleton for Azure OpenAI Whisper transcription."""

import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from openai import AzureOpenAI


def _get_whisper_client():
    key = os.environ.get("WHISPER_API_KEY")
    endpoint = os.environ.get("WHISPER_API_ENDPOINT")
    api_version = os.environ.get("WHISPER_API_VERSION", "2024-12-01-preview")
    if key and endpoint:
        base_endpoint = endpoint.split("/openai")[0]
        return AzureOpenAI(
            api_key=key, azure_endpoint=base_endpoint, api_version=api_version
        )
    return None


_whisper_client = None


def _ensure_whisper_client():
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = _get_whisper_client()
    return _whisper_client


def transcribe_via_api(
    audio_np: np.ndarray, sr: int, language: str | None = None, client=None
) -> list:
    if client is None:
        client = _ensure_whisper_client()
    if client is None:
        raise ValueError(
            "Whisper API credentials not found in environment"
            " (WHISPER_API_KEY, WHISPER_API_ENDPOINT)."
        )
    deployment = os.environ.get("WHISPER_API_DEPLOYMENT")
    tmp_dir = Path(__file__).resolve().parent.parent / "tmp_whisper"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    os.close(fd)
    try:
        wav.write(tmp_path, sr, (audio_np * 32767).astype(np.int16))
        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=deployment,
                file=audio_file,
                language=language,
                response_format="verbose_json",
            )
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_path)
    segments = getattr(response, "segments", [])
    if segments and not isinstance(segments[0], dict):
        segments = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": str(s.text),
                "avg_logprob": getattr(s, "avg_logprob", 0),
                "no_speech_prob": getattr(s, "no_speech_prob", 0),
            }
            for s in segments
        ]
    return segments

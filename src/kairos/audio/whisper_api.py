"""Whisper API client: lazy singleton for Azure OpenAI Whisper transcription.

Manages an :class:`~openai.AzureOpenAI` client whose credentials are read
from environment variables (``WHISPER_API_KEY``, ``WHISPER_API_ENDPOINT``,
``WHISPER_API_VERSION``, ``WHISPER_API_DEPLOYMENT``).  The client is created
on first use and reused for subsequent calls.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from openai import AzureOpenAI


def _get_whisper_client() -> AzureOpenAI | None:
    """Build an Azure OpenAI Whisper client from environment variables.

    Reads ``WHISPER_API_KEY``, ``WHISPER_API_ENDPOINT``, and
    ``WHISPER_API_VERSION`` (default ``"2024-12-01-preview"``) from the
    process environment.  If both the key and endpoint are present a new
    :class:`~openai.AzureOpenAI` instance is returned; otherwise
    ``None``.

    Returns:
        A configured :class:`~openai.AzureOpenAI` client, or ``None``
        when the required environment variables are not set.
    """
    key: str | None = os.environ.get("WHISPER_API_KEY")
    endpoint: str | None = os.environ.get("WHISPER_API_ENDPOINT")
    api_version: str = os.environ.get("WHISPER_API_VERSION", "2024-12-01-preview")
    if key and endpoint:
        base_endpoint: str = endpoint.split("/openai")[0]
        return AzureOpenAI(
            api_key=key, azure_endpoint=base_endpoint, api_version=api_version
        )
    return None


_whisper_client: AzureOpenAI | None = None


def _ensure_whisper_client() -> AzureOpenAI | None:
    """Ensure the module-level singleton Whisper client exists.

    Creates the client via :func:`_get_whisper_client` on the first call
    and caches it in the module-global ``_whisper_client``.  Subsequent
    calls return the cached instance.

    Returns:
        The cached :class:`~openai.AzureOpenAI` client, or ``None`` when
        credentials are unavailable.
    """
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = _get_whisper_client()
    return _whisper_client


def transcribe_via_api(
    audio_np: np.ndarray,
    sr: int,
    language: str | None = None,
    client: AzureOpenAI | None = None,
) -> list[dict]:
    """Transcribe audio via the Azure OpenAI Whisper API.

    Writes the audio array to a temporary ``.wav`` file, sends it to the
    Whisper deployment configured by ``WHISPER_API_DEPLOYMENT``, and
    returns the resulting segments as a list of dicts.

    Args:
        audio_np: 1-D float NumPy array of audio samples in the range
            ``[-1.0, 1.0]``.
        sr: Sampling rate in Hz.
        language: Optional ISO-639-1 language hint passed to the API
            (e.g. ``"en"``).  When ``None`` the API auto-detects.
        client: Pre-built :class:`~openai.AzureOpenAI` client.  If
            ``None``, the module singleton is used (created on demand).

    Returns:
        List of segment dicts, each containing ``"start"``, ``"end"``,
        ``"text"``, ``"avg_logprob"``, and ``"no_speech_prob"`` keys.

    Raises:
        ValueError: If no client could be obtained (missing credentials).
    """
    if client is None:
        client = _ensure_whisper_client()
    if client is None:
        raise ValueError(
            "Whisper API credentials not found in environment"
            " (WHISPER_API_KEY, WHISPER_API_ENDPOINT)."
        )
    deployment: str | None = os.environ.get("WHISPER_API_DEPLOYMENT")
    tmp_dir: Path = Path(__file__).resolve().parent.parent / "tmp_whisper"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd: int
    tmp_path: str
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

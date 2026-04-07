from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from systems.base import AskResult, BaseSystem


def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


class GoogleGeminiSystem(BaseSystem):
    """Video QA via Google GenAI (Gemini) using uploaded video files."""

    def __init__(
        self,
        name: str = "google_gemini",
        model: str = "gemini-2.5-flash",
        vertexai: bool = True,
        env_api_key: str = "GEMINI_API_KEY",
        poll_interval_sec: float = 2.0,
        file_active_timeout_sec: float = 3600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._model = model
        self._vertexai = vertexai
        self._env_api_key = env_api_key
        self._poll_interval = poll_interval_sec
        self._file_timeout = file_active_timeout_sec
        self._client = None
        self._file_handles: dict[str, Any] = {}

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = os.environ.get(self._env_api_key)
        if not key:
            raise RuntimeError(f"Set {self._env_api_key} for Google Gemini.")
        from google import genai

        self._client = genai.Client(vertexai=self._vertexai, api_key=key)
        return self._client

    def _wait_file_active(self, file_name: str) -> None:
        client = self._ensure_client()
        deadline = time.monotonic() + self._file_timeout
        while time.monotonic() < deadline:
            info = client.files.get(name=file_name)
            state = getattr(info, "state", None) or _get_by_path(info, "state")
            state_name = getattr(state, "name", None) or str(state)
            if state_name and "ACTIVE" in state_name.upper():
                return
            if state_name and ("FAILED" in state_name.upper() or "ERROR" in state_name.upper()):
                raise RuntimeError(f"Gemini file processing failed: {state_name!r}")
            time.sleep(self._poll_interval)
        raise TimeoutError(f"Gemini file {file_name} did not become ACTIVE in time.")

    def upload_video(self, video_path: Path, video_id: str) -> str:
        client = self._ensure_client()
        path = video_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        uploaded = client.files.upload(file=str(path))
        name = getattr(uploaded, "name", None)
        if name is None and isinstance(uploaded, dict):
            name = uploaded.get("name")
        if not name:
            raise RuntimeError("Upload response missing file name.")
        self._wait_file_active(name)
        self._file_handles[video_id] = uploaded
        return name

    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        from google.genai import types

        client = self._ensure_client()
        t0 = time.perf_counter()
        video_part = self._file_handles.get(video_id)
        if video_part is None:
            video_part = client.files.get(name=remote_video_id)
        response = client.models.generate_content(
            model=self._model,
            contents=[video_part, question],
            config=types.GenerateContentConfig(temperature=0),
        )
        text = getattr(response, "text", None) or ""
        if not text and hasattr(response, "candidates"):
            parts = []
            for c in response.candidates or []:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        if getattr(p, "text", None):
                            parts.append(p.text)
            text = "\n".join(parts)
        return AskResult(
            text=text.strip(),
            latency_sec=time.perf_counter() - t0,
            raw={"model": self._model},
            metadata={"summaries_generated": 0},
        )

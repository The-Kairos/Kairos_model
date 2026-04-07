from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from systems.base import AskResult, BaseSystem


class TwelveLabsSystem(BaseSystem):
    """
    Twelve Labs: index a video task, poll until ready, then open-ended generation.
    Docs: https://docs.twelvelabs.io/
    Requires TWELVE_LABS_API_KEY and TWELVE_LABS_INDEX_ID (create an index in the dashboard).
    """

    def __init__(
        self,
        name: str = "twelve_labs",
        env_api_key: str = "TWELVE_LABS_API_KEY",
        base_url: str = "https://api.twelvelabs.io/v1.3",
        index_id_env: str = "TWELVE_LABS_INDEX_ID",
        poll_interval_sec: float = 5.0,
        task_timeout_sec: float = 7200.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._env_api_key = env_api_key
        self._base = base_url.rstrip("/")
        self._index_id_env = index_id_env
        self._poll_interval = poll_interval_sec
        self._task_timeout = task_timeout_sec
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self._env_api_key)
        if not key:
            raise RuntimeError(f"Set {self._env_api_key} for Twelve Labs.")
        return {"x-api-key": key}

    def _index_id(self) -> str:
        iid = os.environ.get(self._index_id_env)
        if not iid:
            raise RuntimeError(f"Set {self._index_id_env} to your Twelve Labs index id.")
        return iid

    def upload_video(self, video_path: Path, video_id: str) -> str:
        path = video_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        url = f"{self._base}/tasks"
        data = {"index_id": self._index_id(), "language": "en"}
        with open(path, "rb") as f:
            files = {"video_file": (path.name, f, "application/octet-stream")}
            r = self._session.post(url, headers=self._headers(), data=data, files=files, timeout=600)
        r.raise_for_status()
        body = r.json()
        tid = body.get("_id") or body.get("id") or body.get("task_id")
        if not tid:
            raise RuntimeError(f"Unexpected task create response: {body}")
        self._wait_task_ready(tid)
        vid = self._video_id_from_task(tid)
        return vid

    def _wait_task_ready(self, task_id: str) -> None:
        url = f"{self._base}/tasks/{task_id}"
        deadline = time.monotonic() + self._task_timeout
        while time.monotonic() < deadline:
            r = self._session.get(url, headers=self._headers(), timeout=120)
            r.raise_for_status()
            body = r.json()
            status = (body.get("status") or "").lower()
            if status in ("ready", "completed", "success"):
                return
            if status in ("failed", "error"):
                raise RuntimeError(f"Twelve Labs task {task_id} failed: {body}")
            time.sleep(self._poll_interval)
        raise TimeoutError(f"Twelve Labs task {task_id} did not finish.")

    def _video_id_from_task(self, task_id: str) -> str:
        url = f"{self._base}/tasks/{task_id}"
        r = self._session.get(url, headers=self._headers(), timeout=120)
        r.raise_for_status()
        body = r.json()
        vid = body.get("video_id") or (body.get("video") or {}).get("_id")
        if not vid:
            raise RuntimeError(f"Could not read video_id from task payload: {body}")
        return vid

    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        # Open-ended video QA: https://docs.twelvelabs.io/api-reference/analyze-videos/analyze
        url = f"{self._base}/analyze"
        payload = {
            "video_id": remote_video_id,
            "prompt": question,
            "temperature": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        r = self._session.post(url, headers={**self._headers(), "Content-Type": "application/json"}, json=payload, timeout=300)
        r.raise_for_status()
        body = r.json()
        text = (
            body.get("data")
            or body.get("generated_text")
            or body.get("text")
            or body.get("analysis")
            or ""
        )
        if isinstance(text, dict):
            text = text.get("text") or text.get("content") or str(text)
        return AskResult(
            text=str(text).strip(),
            latency_sec=time.perf_counter() - t0,
            raw=body,
            ranked_segment_ids=self._extract_ranked_ids(body),
            metadata={"summaries_generated": 0},
        )

    def _extract_ranked_ids(self, body: dict[str, Any]) -> list[str] | None:
        clips = body.get("clips") or body.get("segments")
        if not clips or not isinstance(clips, list):
            return None
        out: list[str] = []
        for c in clips:
            if isinstance(c, dict):
                cid = c.get("id") or c.get("segment_id") or c.get("_id")
                if cid:
                    out.append(str(cid))
        return out or None

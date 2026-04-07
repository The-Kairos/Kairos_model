from __future__ import annotations

import time
from pathlib import Path

from systems.base import AskResult, BaseSystem


class MockSystem(BaseSystem):
    """Offline stub: echoes the question so you can test the pipeline without vendor APIs."""

    def __init__(self, name: str = "mock", echo_prefix: str = "[mock] ", **kwargs: object) -> None:
        super().__init__(name, echo_prefix=echo_prefix, **kwargs)
        self._echo_prefix = echo_prefix
        self._uploaded: dict[str, str] = {}

    def upload_video(self, video_path: Path, video_id: str) -> str:
        self._uploaded[video_id] = str(video_path.resolve())
        return f"mock_remote::{video_id}"

    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        t0 = time.perf_counter()
        text = f"{self._echo_prefix}{question}"
        return AskResult(
            text=text,
            latency_sec=time.perf_counter() - t0,
            raw={"remote_video_id": remote_video_id, "video_id": video_id},
            ranked_segment_ids=None,
            metadata={"summaries_generated": 0},
        )

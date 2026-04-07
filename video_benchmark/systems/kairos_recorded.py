from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from systems.base import AskResult, BaseSystem


def _bench_data() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


class KairosRecordedSystem(BaseSystem):
    """
    Replays answers from an import script output (no Kairos API).
    Matches on (video_id, normalized question text).
    """

    def __init__(self, name: str = "kairos_recorded", recording_file: str = "", **kwargs: Any) -> None:
        super().__init__(name, recording_file=recording_file, **kwargs)
        if not recording_file:
            raise ValueError("KairosRecordedSystem requires recording_file under data/")
        path = (_bench_data() / recording_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Kairos recording not found: {path}")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        self._by_key: dict[tuple[str, str], dict] = {}
        for r in rows:
            vid = r.get("video_id", "")
            q = r.get("question", "")
            self._by_key[(vid, self._norm(q))] = r

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.strip().split()).lower()

    def upload_video(self, video_path: Path, video_id: str) -> str:
        return f"kairos_recording::{video_id}"

    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        t0 = time.perf_counter()
        row = self._by_key.get((video_id, self._norm(question)))
        if not row:
            return AskResult(
                text="",
                latency_sec=time.perf_counter() - t0,
                raw={"error": "no_recording_match", "video_id": video_id},
                metadata={},
            )
        return AskResult(
            text=str(row.get("response", "")).strip(),
            latency_sec=float(row.get("latency_sec") or time.perf_counter() - t0),
            raw={"kairos": row.get("kairos"), "question_id": row.get("question_id")},
            ranked_segment_ids=row.get("ranked_segment_ids"),
            metadata={"summaries_generated": 0},
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AskResult:
    """Unified response from any video-QA backend."""

    text: str
    latency_sec: float
    raw: dict[str, Any] = field(default_factory=dict)
    # Ordered list of segment/chunk ids returned by retrieval (optional, for recall@K)
    ranked_segment_ids: list[str] | None = None
    # Free-form counters the adapter can expose (e.g. summaries_generated)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSystem(ABC):
    """Fair-comparison interface: same videos, same questions across vendors."""

    name: str

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self.params = params

    @abstractmethod
    def upload_video(self, video_path: Path, video_id: str) -> str:
        """Upload or register a video; return an opaque handle for subsequent asks."""

    @abstractmethod
    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        """Answer one natural-language question about the given video."""

    def close(self) -> None:
        """Release connections or remote resources if needed."""
        return None

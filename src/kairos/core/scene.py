"""Typed Scene dataclass replacing raw ``dict`` scene representations.

Every pipeline stage that previously passed ``dict`` objects with keys
like ``start_seconds``, ``frame_captions``, etc. should migrate to
:class:`Scene`.  The class provides a :meth:`to_dict` serializer and
a :meth:`from_dict` constructor for backward-compatible checkpoint I/O.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Scene:
    """A single detected scene with all enriched pipeline data."""

    # ---- Identity / timing (set by scene detection) ----------------------
    scene_index: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    duration_seconds: float = 0.0
    start_timecode: str = "00:00:00.000"
    end_timecode: str = "00:00:00.000"

    # ---- Video clip (set by save_clips) ----------------------------------
    clip_path: Optional[str] = None

    # ---- Frame sampling (set by sample_frames / sample_fps) ---------------
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    frame_paths: Optional[list[str]] = None

    # ---- YOLO-specific frames (set by sample_fps for YOLO) ---------------
    yolo_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    yolo_frame_paths: Optional[list[str]] = None

    # ---- BLIP captioning (set by caption_frames) -------------------------
    frame_captions: list[str] = field(default_factory=list)

    # ---- YOLO detection (set by detect_object_yolo) ----------------------
    yolo_detections: Any = field(default_factory=list)

    # ---- Audio (set by AST / Whisper) ------------------------------------
    audio_natural: str = ""
    audio_speech: str = ""

    # ---- LLM enrichment (set by describe_scenes) -------------------------
    llm_scene_description: str = ""

    # ---- Extra fields for forward compatibility --------------------------
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- Serialization ---------------------------------------------------

    #: Keys that hold large non-serializable data (numpy arrays, etc.)
    _TRANSIENT_KEYS: set[str] = frozenset(
        {
            "frames",
            "yolo_frames",
            "frame_paths",
            "yolo_frame_paths",
            "frame_indices",
            "frame_timestamps",
            "sample_fps",
            "motion_bullets",
            "yolo_tracks",
            "yolo_track_summaries",
        }
    )

    def to_dict(self, *, include_transient: bool = False) -> dict:
        """Convert to a plain ``dict`` suitable for JSON checkpoint serialization.

        By default, transient keys (frames, yolo_frames, etc.) are omitted.
        """
        d: dict[str, Any] = {}
        for k in (
            "scene_index",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "start_timecode",
            "end_timecode",
            "clip_path",
            "frame_captions",
            "yolo_detections",
            "audio_natural",
            "audio_speech",
            "llm_scene_description",
        ):
            val = getattr(self, k)
            if val is not None and val != "" and val != [] and val != {}:
                d[k] = val
            elif k in (
                "scene_index",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_timecode",
                "end_timecode",
            ):
                d[k] = val  # always include timing keys
        if include_transient:
            if self.frames:
                d["frames"] = self.frames
            if self.frame_paths:
                d["frame_paths"] = self.frame_paths
            if self.yolo_frames:
                d["yolo_frames"] = self.yolo_frames
            if self.yolo_frame_paths:
                d["yolo_frame_paths"] = self.yolo_frame_paths
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        """Reconstruct a :class:`Scene` from a plain ``dict`` (e.g. from checkpoint)."""
        known_keys = {
            "scene_index",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "start_timecode",
            "end_timecode",
            "clip_path",
            "frames",
            "frame_paths",
            "yolo_frames",
            "yolo_frame_paths",
            "frame_captions",
            "yolo_detections",
            "audio_natural",
            "audio_speech",
            "llm_scene_description",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in known_keys:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)

    def deepcopy(self) -> Scene:
        """Return a deep copy (including frame arrays)."""
        return copy.deepcopy(self)

    def shallow_copy(self, *, share_frames: bool = True) -> Scene:
        """Return a shallow copy.

        If *share_frames* is ``True`` (the default), frame arrays are shared
        (not duplicated) which is much cheaper.
        """
        new = Scene(
            **{k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
        )
        new.extra = dict(self.extra)
        if not share_frames:
            new.frames = [f.copy() for f in self.frames]
            new.yolo_frames = [f.copy() for f in self.yolo_frames]
        return new

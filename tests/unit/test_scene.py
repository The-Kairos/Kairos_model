"""Tests for the typed Scene dataclass.

Validates construction with defaults and kwargs, serialisation via
``to_dict`` (including transient-field handling), round-trip
dict ↔ Scene conversion, and both shallow and deep copy behaviour.
"""

from typing import Any

import numpy as np

from kairos.core.scene import Scene


class TestSceneConstruction:
    """Tests for Scene __init__ and default values."""

    def test_defaults(self) -> None:
        """Verify all default field values of a freshly constructed Scene."""
        s = Scene()
        assert s.scene_index == 0
        assert s.frames == []
        assert s.frame_captions == []
        assert s.audio_speech == ""

    def test_from_kwargs(self) -> None:
        """Verify construction with explicit keyword arguments."""
        s = Scene(
            scene_index=5,
            start_seconds=10.0,
            end_seconds=15.0,
            duration_seconds=5.0,
            audio_speech="hello",
        )
        assert s.scene_index == 5
        assert s.audio_speech == "hello"


class TestSceneSerialization:
    """Tests for Scene.to_dict and Scene.from_dict."""

    def test_to_dict_includes_timing(self) -> None:
        """Verify to_dict includes timing fields."""
        s = Scene(
            scene_index=1, start_seconds=0.0, end_seconds=5.0, duration_seconds=5.0
        )
        d = s.to_dict()
        assert d["scene_index"] == 1
        assert "start_seconds" in d

    def test_to_dict_omits_empty(self) -> None:
        """Verify to_dict omits empty/default optional fields."""
        s = Scene()
        d = s.to_dict()
        assert "frame_captions" not in d
        assert "llm_scene_description" not in d

    def test_to_dict_omits_transient_by_default(self) -> None:
        """Verify to_dict excludes transient frame data by default."""
        s = Scene(frames=[np.zeros((10, 10, 3), dtype=np.uint8)])
        d = s.to_dict()
        assert "frames" not in d

    def test_to_dict_includes_transient_when_asked(self) -> None:
        """Verify to_dict includes transient data when include_transient=True."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        d = s.to_dict(include_transient=True)
        assert "frames" in d
        assert len(d["frames"]) == 1

    def test_roundtrip(self) -> None:
        """Verify dict → Scene → dict round-trip preserves all fields."""
        s = Scene(
            scene_index=2,
            start_seconds=5.0,
            end_seconds=12.0,
            duration_seconds=7.0,
            start_timecode="00:00:05.000",
            end_timecode="00:00:12.000",
            frame_captions=["a crowd"],
            audio_speech="applause",
            llm_scene_description="The audience claps.",
        )
        d = s.to_dict()
        s2 = Scene.from_dict(d)
        assert s2.scene_index == s.scene_index
        assert s2.frame_captions == s.frame_captions
        assert s2.llm_scene_description == s.llm_scene_description

    def test_from_dict_extra_keys(self) -> None:
        """Verify from_dict stores unrecognised keys in the extra dict."""
        d: dict[str, Any] = {
            "scene_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "duration_seconds": 5.0,
            "custom_field": "hello",
        }
        s = Scene.from_dict(d)
        assert s.extra["custom_field"] == "hello"


class TestSceneCopy:
    """Tests for Scene shallow_copy and deepcopy."""

    def test_shallow_copy_shares_frames(self) -> None:
        """Verify shallow_copy with share_frames=True shares frame references."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        s2 = s.shallow_copy(share_frames=True)
        assert s2.frames[0] is s.frames[0]

    def test_shallow_copy_no_share(self) -> None:
        """Verify shallow_copy with share_frames=False copies frame data."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        s2 = s.shallow_copy(share_frames=False)
        assert s2.frames[0] is not s.frames[0]
        np.testing.assert_array_equal(s2.frames[0], s.frames[0])

    def test_deepcopy(self) -> None:
        """Verify deepcopy creates fully independent copies."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame], frame_captions=["test"])
        s2 = s.deepcopy()
        s2.frame_captions.append("added")
        assert len(s.frame_captions) == 1

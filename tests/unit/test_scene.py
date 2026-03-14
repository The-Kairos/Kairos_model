"""Tests for the typed Scene dataclass."""

import numpy as np

from kairos.core.scene import Scene


class TestSceneConstruction:
    def test_defaults(self):
        s = Scene()
        assert s.scene_index == 0
        assert s.frames == []
        assert s.frame_captions == []
        assert s.audio_speech == ""

    def test_from_kwargs(self):
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
    def test_to_dict_includes_timing(self):
        s = Scene(
            scene_index=1, start_seconds=0.0, end_seconds=5.0, duration_seconds=5.0
        )
        d = s.to_dict()
        assert d["scene_index"] == 1
        assert "start_seconds" in d

    def test_to_dict_omits_empty(self):
        s = Scene()
        d = s.to_dict()
        assert "frame_captions" not in d
        assert "llm_scene_description" not in d

    def test_to_dict_omits_transient_by_default(self):
        s = Scene(frames=[np.zeros((10, 10, 3), dtype=np.uint8)])
        d = s.to_dict()
        assert "frames" not in d

    def test_to_dict_includes_transient_when_asked(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        d = s.to_dict(include_transient=True)
        assert "frames" in d
        assert len(d["frames"]) == 1

    def test_roundtrip(self):
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

    def test_from_dict_extra_keys(self):
        d = {
            "scene_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "duration_seconds": 5.0,
            "custom_field": "hello",
        }
        s = Scene.from_dict(d)
        assert s.extra["custom_field"] == "hello"


class TestSceneCopy:
    def test_shallow_copy_shares_frames(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        s2 = s.shallow_copy(share_frames=True)
        assert s2.frames[0] is s.frames[0]

    def test_shallow_copy_no_share(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame])
        s2 = s.shallow_copy(share_frames=False)
        assert s2.frames[0] is not s.frames[0]
        np.testing.assert_array_equal(s2.frames[0], s.frames[0])

    def test_deepcopy(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        s = Scene(frames=[frame], frame_captions=["test"])
        s2 = s.deepcopy()
        s2.frame_captions.append("added")
        assert len(s.frame_captions) == 1

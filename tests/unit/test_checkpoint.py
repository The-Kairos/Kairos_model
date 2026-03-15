"""Tests for kairos.core.checkpoint.

Covers JSON reading, frame-data stripping, checkpoint serialization, and
the ``have_key`` helper for checking key presence across all scenes.
"""

import json
from pathlib import Path
from typing import Any

from kairos.core.checkpoint import clear_frames, have_key, read_json, save_checkpoint


def test_read_json_missing_file(tmp_path: Path) -> None:
    """Verify reading a nonexistent JSON file returns an empty dict."""
    result = read_json(tmp_path / "nonexistent.json")
    assert result == {}


def test_read_json_dict(tmp_path: Path) -> None:
    """Verify reading a JSON dict preserves all keys."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"scenes": [{"a": 1}], "other": "val"}))
    result = read_json(path)
    assert "scenes" in result
    assert result["other"] == "val"


def test_read_json_list_wraps(tmp_path: Path) -> None:
    """Verify reading a JSON list wraps it into a dict with a 'scenes' key."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"scene_index": 0}]))
    result = read_json(path)
    assert isinstance(result, dict)
    assert "scenes" in result
    assert len(result["scenes"]) == 1


def test_clear_frames() -> None:
    """Verify clear_frames removes transient frame data but keeps other keys."""
    scenes: list[dict[str, Any]] = [
        {
            "scene_index": 0,
            "frames": [[1, 2, 3]],
            "yolo_frames": [[4, 5]],
            "frame_captions": ["cap"],
        },
        {"scene_index": 1, "frames": [[6]], "frame_paths": ["/a.jpg"]},
    ]
    cleaned = clear_frames(scenes)
    for scene in cleaned:
        assert "frames" not in scene
        assert "yolo_frames" not in scene
        assert "frame_paths" not in scene
    assert cleaned[0]["frame_captions"] == ["cap"]
    assert cleaned[0]["scene_index"] == 0


def test_save_checkpoint_strips_frames(tmp_path: Path) -> None:
    """Verify save_checkpoint removes frame data before writing to disk."""
    path = tmp_path / "cp.json"
    checkpoint: dict[str, Any] = {
        "scenes": [
            {
                "scene_index": 0,
                "frames": [[1]],
                "yolo_frames": [[2]],
                "frame_captions": ["a"],
            },
        ],
        "steps": {},
    }
    result = save_checkpoint(checkpoint, path)
    assert "frames" not in result["scenes"][0]
    assert "yolo_frames" not in result["scenes"][0]
    assert result["scenes"][0]["frame_captions"] == ["a"]

    # Verify file was written
    with open(path) as f:
        data = json.load(f)
    assert "frames" not in data["scenes"][0]


def test_save_checkpoint_list_input(tmp_path: Path) -> None:
    """Verify save_checkpoint wraps a bare list into a dict with 'scenes'."""
    path = tmp_path / "cp.json"
    scenes: list[dict[str, Any]] = [{"scene_index": 0, "frames": [[1]]}]
    result = save_checkpoint(scenes, path)
    assert isinstance(result, dict)
    assert "scenes" in result


def test_have_key_true() -> None:
    """Verify have_key returns True when all scenes contain the key."""
    scenes: list[dict[str, Any]] = [{"a": 1}, {"a": 2}]
    assert have_key(scenes, "a") is True


def test_have_key_false() -> None:
    """Verify have_key returns False when not all scenes contain the key."""
    scenes: list[dict[str, Any]] = [{"a": 1}, {"b": 2}]
    assert have_key(scenes, "a") is False


def test_have_key_empty() -> None:
    """Verify have_key returns False for an empty scene list."""
    assert have_key([], "a") is False

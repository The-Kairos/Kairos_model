"""Tests for kairos.cli.catalog.

Covers video-length categorization, output directory construction,
catalog loading from JSON files, and video argument resolution
against the filesystem and blob index.
"""

import json
from pathlib import Path
from typing import Any

from kairos.cli.catalog import (
    categorize_length,
    load_video_catalog,
    make_output_dir,
    resolve_video_arg,
)


def test_categorize_length_short() -> None:
    """Verify a 300-second video is categorized as 'short'."""
    assert categorize_length(300) == "short"


def test_categorize_length_medium() -> None:
    """Verify a 1200-second video is categorized as 'medium'."""
    assert categorize_length(1200) == "medium"


def test_categorize_length_long() -> None:
    """Verify a 4000-second video is categorized as 'long'."""
    assert categorize_length(4000) == "long"


def test_categorize_length_extra() -> None:
    """Verify a 6000-second video is categorized as 'extra'."""
    assert categorize_length(6000) == "extra"


def test_categorize_length_boundaries() -> None:
    """Verify exact boundary values for length categories."""
    assert categorize_length(599) == "short"
    assert categorize_length(600) == "medium"
    assert categorize_length(1799) == "medium"
    assert categorize_length(1800) == "long"
    assert categorize_length(5399) == "long"
    assert categorize_length(5400) == "extra"


def test_make_output_dir() -> None:
    """Verify output directory is constructed from video path and base dir."""
    result = make_output_dir(Path("data/videos/test.mp4"), "processed")
    assert result == "processed/test.mp4"


def test_make_output_dir_strips_dots() -> None:
    """Verify leading dots are stripped from the video filename."""
    result = make_output_dir(Path(".hidden_video.mp4"), "out")
    assert result == "out/hidden_video.mp4"


def test_load_video_catalog_list(tmp_path: Path) -> None:
    """Verify loading a catalog from a JSON list of video entries."""
    catalog_path = tmp_path / "catalog.json"
    data = [{"blob": "a.mp4"}, {"blob": "b.mp4"}]
    catalog_path.write_text(json.dumps(data))
    result = load_video_catalog(catalog_path)
    assert len(result) == 2
    assert result[0]["blob"] == "a.mp4"


def test_load_video_catalog_dict_with_videos(tmp_path: Path) -> None:
    """Verify loading a catalog from a JSON dict containing a 'videos' key."""
    catalog_path = tmp_path / "catalog.json"
    data = {"videos": [{"blob": "a.mp4"}]}
    catalog_path.write_text(json.dumps(data))
    result = load_video_catalog(catalog_path)
    assert len(result) == 1


def test_load_video_catalog_missing(tmp_path: Path) -> None:
    """Verify loading a missing catalog file returns an empty list."""
    result = load_video_catalog(tmp_path / "missing.json")
    assert result == []


def test_resolve_video_arg_direct_path(tmp_path: Path) -> None:
    """Verify resolution when a direct file path is given."""
    video = tmp_path / "test.mp4"
    video.touch()
    result = resolve_video_arg(str(video), {}, tmp_path)
    assert result == video


def test_resolve_video_arg_in_videos_dir(tmp_path: Path) -> None:
    """Verify resolution when the video name exists in the videos directory."""
    video = tmp_path / "test.mp4"
    video.touch()
    result = resolve_video_arg("test.mp4", {}, tmp_path)
    assert result == video


def test_resolve_video_arg_blob_lookup(tmp_path: Path) -> None:
    """Verify resolution through blob index lookup."""
    video = tmp_path / "actual_file.mp4"
    video.touch()
    blob_index: dict[str, Any] = {"my_video": {"blob": "actual_file.mp4"}}
    result = resolve_video_arg("my_video", blob_index, tmp_path)
    assert result == video


def test_resolve_video_arg_not_found(tmp_path: Path) -> None:
    """Verify None is returned when the video cannot be found."""
    result = resolve_video_arg("nonexistent.mp4", {}, tmp_path)
    assert result is None

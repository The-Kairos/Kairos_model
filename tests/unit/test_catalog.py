"""Tests for kairos.cli.catalog."""

import json
from pathlib import Path

from kairos.cli.catalog import (
    load_video_catalog,
    categorize_length,
    make_output_dir,
    resolve_video_arg,
)


def test_categorize_length_short():
    assert categorize_length(300) == "short"


def test_categorize_length_medium():
    assert categorize_length(1200) == "medium"


def test_categorize_length_long():
    assert categorize_length(4000) == "long"


def test_categorize_length_extra():
    assert categorize_length(6000) == "extra"


def test_categorize_length_boundaries():
    assert categorize_length(599) == "short"
    assert categorize_length(600) == "medium"
    assert categorize_length(1799) == "medium"
    assert categorize_length(1800) == "long"
    assert categorize_length(5399) == "long"
    assert categorize_length(5400) == "extra"


def test_make_output_dir():
    result = make_output_dir(Path("data/videos/test.mp4"), "processed")
    assert result == "processed/test.mp4"


def test_make_output_dir_strips_dots():
    result = make_output_dir(Path(".hidden_video.mp4"), "out")
    assert result == "out/hidden_video.mp4"


def test_load_video_catalog_list(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    data = [{"blob": "a.mp4"}, {"blob": "b.mp4"}]
    catalog_path.write_text(json.dumps(data))
    result = load_video_catalog(catalog_path)
    assert len(result) == 2
    assert result[0]["blob"] == "a.mp4"


def test_load_video_catalog_dict_with_videos(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    data = {"videos": [{"blob": "a.mp4"}]}
    catalog_path.write_text(json.dumps(data))
    result = load_video_catalog(catalog_path)
    assert len(result) == 1


def test_load_video_catalog_missing(tmp_path):
    result = load_video_catalog(tmp_path / "missing.json")
    assert result == []


def test_resolve_video_arg_direct_path(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    result = resolve_video_arg(str(video), {}, tmp_path)
    assert result == video


def test_resolve_video_arg_in_videos_dir(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    result = resolve_video_arg("test.mp4", {}, tmp_path)
    assert result == video


def test_resolve_video_arg_blob_lookup(tmp_path):
    video = tmp_path / "actual_file.mp4"
    video.touch()
    blob_index = {"my_video": {"blob": "actual_file.mp4"}}
    result = resolve_video_arg("my_video", blob_index, tmp_path)
    assert result == video


def test_resolve_video_arg_not_found(tmp_path):
    result = resolve_video_arg("nonexistent.mp4", {}, tmp_path)
    assert result is None

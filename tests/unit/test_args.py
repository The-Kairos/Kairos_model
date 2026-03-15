"""Tests for kairos.cli.args.

Validates CLI argument parsing for the ``process`` and ``rag`` subcommands,
including flags, presets, redo options, and error handling for missing
subcommands.
"""

import sys

import pytest

from kairos.cli.args import parse_args


def test_process_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 'process --video foo.mp4' parses correctly."""
    monkeypatch.setattr(sys, "argv", ["main.py", "process", "--video", "foo.mp4"])
    args = parse_args()
    assert args.command == "process"
    assert args.video == ["foo.mp4"]


def test_rag_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 'rag --video foo.mp4' parses correctly."""
    monkeypatch.setattr(sys, "argv", ["main.py", "rag", "--video", "foo.mp4"])
    args = parse_args()
    assert args.command == "rag"
    assert args.video == "foo.mp4"


def test_no_subcommand_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that calling the CLI with no subcommand raises SystemExit."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        parse_args()


def test_process_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --preset flag is captured in the parsed args."""
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--preset", "fast"]
    )
    args = parse_args()
    assert args.preset == "fast"


def test_process_redo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --redo flag captures the redo stage list."""
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--redo", "scenes"]
    )
    args = parse_args()
    assert args.redo == [["scenes"]]


def test_process_multiple_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multiple --video flags are collected into a list."""
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--video", "a.mp4", "--video", "b.mp4"]
    )
    args = parse_args()
    assert args.video == ["a.mp4", "b.mp4"]


def test_process_all_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --all flag is set to True when provided."""
    monkeypatch.setattr(sys, "argv", ["main.py", "process", "--all"])
    args = parse_args()
    assert args.all is True


def test_process_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --filter flag captures the filter value."""
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--filter", "short"]
    )
    args = parse_args()
    assert args.filter == "short"

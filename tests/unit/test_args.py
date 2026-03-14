"""Tests for kairos.cli.args."""

import sys

import pytest

from kairos.cli.args import parse_args


def test_process_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "process", "--video", "foo.mp4"])
    args = parse_args()
    assert args.command == "process"
    assert args.video == ["foo.mp4"]


def test_rag_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "rag", "--video", "foo.mp4"])
    args = parse_args()
    assert args.command == "rag"
    assert args.video == "foo.mp4"


def test_no_subcommand_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        parse_args()


def test_process_preset(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--preset", "fast"]
    )
    args = parse_args()
    assert args.preset == "fast"


def test_process_redo(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--redo", "scenes"]
    )
    args = parse_args()
    assert args.redo == [["scenes"]]


def test_process_multiple_videos(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--video", "a.mp4", "--video", "b.mp4"]
    )
    args = parse_args()
    assert args.video == ["a.mp4", "b.mp4"]


def test_process_all_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "process", "--all"])
    args = parse_args()
    assert args.all is True


def test_process_filter(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "process", "--all", "--filter", "short"]
    )
    args = parse_args()
    assert args.filter == "short"

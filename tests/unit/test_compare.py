"""Tests for kairos.cli.compare helper functions.

Validates the ``format_yolo`` function that renders YOLO detection dicts
into human-readable per-frame strings.
"""

from typing import Any

from kairos.cli.compare import format_yolo


class TestFormatYolo:
    """Tests for the format_yolo detection formatter."""

    def test_normal_detections(self) -> None:
        """Verify formatting of a typical multi-frame detection dict."""
        yolo: dict[str, Any] = {
            "0": [
                {"label": "person", "confidence": 0.95432},
                {"label": "car", "confidence": 0.8},
            ],
            "1": [{"label": "dog"}],
        }
        result = format_yolo(yolo)
        assert "Frame 0:" in result
        assert "- person (0.95)" in result
        assert "- car (0.8)" in result
        assert "Frame 1:" in result
        assert "- dog" in result

    def test_empty_dict(self) -> None:
        """Verify an empty dict produces an empty string."""
        assert format_yolo({}) == ""

    def test_non_dict(self) -> None:
        """Verify non-dict inputs produce an empty string."""
        assert format_yolo("not a dict") == ""
        assert format_yolo(None) == ""
        assert format_yolo(42) == ""

    def test_empty_detections_skipped(self) -> None:
        """Verify frames with empty detection lists are omitted from output."""
        result = format_yolo({"0": [], "1": [{"label": "cat"}]})
        assert "Frame 0:" not in result
        assert "Frame 1:" in result
        assert "- cat" in result

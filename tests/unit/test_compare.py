"""Tests for kairos.cli.compare helper functions."""

from kairos.cli.compare import format_yolo


class TestFormatYolo:
    def test_normal_detections(self):
        yolo = {
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

    def test_empty_dict(self):
        assert format_yolo({}) == ""

    def test_non_dict(self):
        assert format_yolo("not a dict") == ""
        assert format_yolo(None) == ""
        assert format_yolo(42) == ""

    def test_empty_detections_skipped(self):
        result = format_yolo({"0": [], "1": [{"label": "cat"}]})
        assert "Frame 0:" not in result
        assert "Frame 1:" in result
        assert "- cat" in result

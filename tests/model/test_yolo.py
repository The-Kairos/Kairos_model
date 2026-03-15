"""Model tests for YOLO object detection.

Runs YOLOv8 inference on a real sample frame to verify detection output
structure, key types, and expected object labels.
"""

from pathlib import Path

import cv2
import pytest

pytestmark = pytest.mark.model


def test_yolo_detect_real_image(sample_frame_path: Path) -> None:
    """Verify YOLO returns a list of detections from a real image."""
    from ultralytics import YOLO

    from kairos.video.yolo_inference import run_yolo_on_frame

    model = YOLO("models/yolov8s.pt")
    frame = cv2.imread(str(sample_frame_path))
    assert frame is not None

    detections = run_yolo_on_frame(model, frame, conf=0.3)
    assert isinstance(detections, list)


def test_yolo_detection_keys(sample_frame_path: Path) -> None:
    """Verify detection dicts have label, confidence, bbox."""
    from ultralytics import YOLO

    from kairos.video.yolo_inference import run_yolo_on_frame

    model = YOLO("models/yolov8s.pt")
    frame = cv2.imread(str(sample_frame_path))
    detections = run_yolo_on_frame(model, frame, conf=0.3)

    if detections:
        det = detections[0]
        assert "label" in det
        assert "confidence" in det
        assert "bbox" in det
        assert isinstance(det["label"], str)
        assert isinstance(det["confidence"], float)
        assert len(det["bbox"]) == 4


def test_yolo_detects_person(sample_frame_path: Path) -> None:
    """Verify YOLO detects at least one 'person' in the sample frame."""
    from ultralytics import YOLO

    from kairos.video.yolo_inference import run_yolo_on_frame

    model = YOLO("models/yolov8s.pt")
    frame = cv2.imread(str(sample_frame_path))
    detections = run_yolo_on_frame(model, frame, conf=0.3)

    labels = [d["label"] for d in detections]
    assert "person" in labels, f"Expected 'person' in detections, got: {labels}"

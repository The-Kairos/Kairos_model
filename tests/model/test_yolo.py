"""Model tests for YOLO object detection."""

import pytest

import cv2
import numpy as np

pytestmark = pytest.mark.model


def test_yolo_detect_real_image(sample_frame_path):
    from ultralytics import YOLO
    from kairos.video.object_detection import run_yolo_on_frame

    model = YOLO("models/yolov8s.pt")
    frame = cv2.imread(str(sample_frame_path))
    assert frame is not None

    detections = run_yolo_on_frame(model, frame, conf=0.3)
    assert isinstance(detections, list)


def test_yolo_detection_keys(sample_frame_path):
    from ultralytics import YOLO
    from kairos.video.object_detection import run_yolo_on_frame

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


def test_yolo_detects_person(sample_frame_path):
    from ultralytics import YOLO
    from kairos.video.object_detection import run_yolo_on_frame

    model = YOLO("models/yolov8s.pt")
    frame = cv2.imread(str(sample_frame_path))
    detections = run_yolo_on_frame(model, frame, conf=0.3)

    labels = [d["label"] for d in detections]
    assert "person" in labels, f"Expected 'person' in detections, got: {labels}"

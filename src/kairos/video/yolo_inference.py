"""YOLOv8 inference: single-frame detection and multi-frame tracking."""

from typing import Any

import numpy as np


def run_yolo_on_frame(
    model: Any,
    frame: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
) -> list[dict]:
    """Run YOLOv8 on a single frame and return detections.

    Args:
        model: A loaded ``ultralytics.YOLO`` model instance.
        frame: The input BGR image as a NumPy array.
        conf: Minimum confidence threshold for detections.
        iou: IoU threshold used during non-maximum suppression.

    Returns:
        A list of detection dictionaries, each containing:

        - ``"label"``: The predicted class name.
        - ``"confidence"``: Detection confidence score.
        - ``"bbox"``: Bounding box as ``[x1, y1, x2, y2]``.
    """
    results = model.predict(frame, conf=conf, iou=iou, stream=True, verbose=False)
    detections: list[dict] = []
    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            detections.append(
                {
                    "label": model.names[cls],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
            )
    return detections


def run_yolo_track_on_frames(
    model: Any,
    frames: list[np.ndarray],
    conf: float = 0.25,
    iou: float = 0.45,
    tracker: str = "bytetrack.yaml",
) -> Any | None:
    """Run YOLOv8 tracking on a list of frames.

    Uses the model's built-in ``track`` method with persistence enabled
    so that track IDs are maintained across frames.

    Args:
        model: A loaded ``ultralytics.YOLO`` model instance.
        frames: A list of BGR image arrays to track across.
        conf: Minimum confidence threshold for detections.
        iou: IoU threshold used during non-maximum suppression.
        tracker: Tracker configuration file name (e.g.
            ``"bytetrack.yaml"``).

    Returns:
        A generator / iterable of YOLO result objects if tracking
        succeeds, or ``None`` if an exception occurs.
    """
    try:
        return model.track(
            frames,
            conf=conf,
            iou=iou,
            tracker=tracker,
            persist=True,
            stream=True,
            verbose=False,
        )
    except Exception:
        return None


def parse_yolo_results(results: Any, model: Any) -> dict[int, list[dict]]:
    """Parse raw YOLO result objects into a structured detection dictionary.

    Args:
        results: An iterable of YOLO result objects (as returned by
            ``model.track`` or ``model.predict``).
        model: The ``ultralytics.YOLO`` model instance (used to map
            class indices to label names via ``model.names``).

    Returns:
        A mapping from frame index (zero-based) to a list of detection
        dictionaries.  Each detection contains:

        - ``"label"``: The predicted class name.
        - ``"confidence"``: Detection confidence score.
        - ``"bbox"``: Bounding box as ``[x1, y1, x2, y2]``.
        - ``"track_id"``: Integer track ID if available, otherwise
          ``None``.
    """
    yolo_dict: dict[int, list[dict]] = {}
    for idx, r in enumerate(results):
        dets: list[dict] = []
        if not hasattr(r, "boxes"):
            yolo_dict[idx] = dets
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            track_id: int | None = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id[0])
                except Exception:
                    track_id = None
            dets.append(
                {
                    "label": model.names[cls],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                    "track_id": track_id,
                }
            )
        yolo_dict[idx] = dets
    return yolo_dict

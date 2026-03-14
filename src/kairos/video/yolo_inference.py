"""YOLOv8 inference: single-frame detection and multi-frame tracking."""

import numpy as np


def run_yolo_on_frame(
    model, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45
) -> list:
    """Run YOLOv8 on a single frame and return detections."""
    results = model.predict(frame, conf=conf, iou=iou, stream=True, verbose=False)
    detections = []
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
    model,
    frames: list,
    conf: float = 0.25,
    iou: float = 0.45,
    tracker: str = "bytetrack.yaml",
):
    """Run YOLOv8 tracking on a list of frames. Returns results or None."""
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


def parse_yolo_results(results, model) -> dict:
    yolo_dict = {}
    for idx, r in enumerate(results):
        dets = []
        if not hasattr(r, "boxes"):
            yolo_dict[idx] = dets
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            track_id = None
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

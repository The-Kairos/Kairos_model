# src/yolo_inference.py
from ultralytics import YOLO
import numpy as np


def run_yolo_on_frame(
    model,
    frame: np.ndarray, #process a single frame (np.ndarray)
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Run YOLOv8 on a single frame (np.ndarray).

    Args:
        model: Loaded YOLO model object
        frame: np.ndarray frame (BGR/RGB image)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        detections: list of dictionaries with:
            {
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
    """
    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        verbose=False
    )

    detections = []

    for r in results:
        if not hasattr(r, "boxes"):
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy,
            })

    return detections



def detect_object_yolo(
    scenes: list, # process full scenes list
    model_size: str = "model/yolov8s.pt",
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Run YOLO on a list of scenes.
    Adds a dict: scene["yolo_detections"] = { index: [detections], ... }

    Args:
        scenes: list of scene dictionaries
        model_size: YOLO model name (e.g., yolov8s)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        updated scenes with "yolo_detections" added
    """

    model = YOLO(model_size)

    results_scenes = []

    for scene in scenes:
        new_scene = dict(scene)

        frames = scene.get("frames", [])
        yolo_dict = {}

        # process each frame in scene
        for idx, frame in enumerate(frames):
            detections = run_yolo_on_frame(
                model,
                frame,
                conf=conf,
                iou=iou
            )
            yolo_dict[idx] = detections

        new_scene["yolo_detections"] = yolo_dict
        results_scenes.append(new_scene)

    return results_scenes

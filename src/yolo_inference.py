# src/yolo_inference.py — YOLO inference module

from ultralytics import YOLO
import os

def run_yolo_on_frames(
    scenes,
    model_size="yolov8s",
    conf=0.25,
    iou=0.45,
):
    """
    Run YOLOv8 on sampled frames.

    Args:
        scenes: list of scenes, each containing "frames": list of file paths
        model_size: yolov8s/yolov8m/yolov8l
        conf: confidence threshold
        iou: NMS IoU threshold

    Returns:
        scenes with YOLO detections added to each frame
    """

    model = YOLO(model_size)  # downloads automatically if not found

    results_scenes = []

    for scene in scenes:
        new_scene = dict(scene)
        new_frames = []

        for frame_path in scene["frames"]:
            result = model.predict(
                frame_path,
                conf=conf,
                iou=iou,
                verbose=False
            )

            detections = []
            for r in result:
                for box in getattr(r, "boxes", []):
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf_score = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    detections.append({
                        "label": label,
                        "confidence": conf_score,
                        "bbox": xyxy,
                    })

            new_frames.append({
                "path": frame_path,
                "detections": detections,
            })

        new_scene["frames"] = new_frames
        results_scenes.append(new_scene)

    return results_scenes

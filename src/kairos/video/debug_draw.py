"""Debug drawing utilities for YOLO detections."""

import os
import random

import cv2
import numpy as np

YOLO_COLOR_MAP = {}


def get_color_for_label(label: str):
    if label not in YOLO_COLOR_MAP:
        YOLO_COLOR_MAP[label] = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
    return YOLO_COLOR_MAP[label]


def debug_draw_yolo(frame: np.ndarray, detections: list, save_path: str = None) -> np.ndarray:
    """Draw YOLO detections on a frame for debugging."""
    pad = 20
    drawn = cv2.copyMakeBorder(frame, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    for det in detections:
        label, conf = det["label"], det["confidence"]
        x1, y1, x2, y2 = (int(v) + pad for v in det["bbox"])
        color = get_color_for_label(label)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

        track_id = det.get("track_id")
        text = f"{label}#{track_id} {conf:.2f}" if track_id is not None else f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(drawn, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(drawn, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(save_path, drawn)

    return drawn

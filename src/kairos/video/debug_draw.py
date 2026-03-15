"""Debug drawing utilities for YOLO detections."""

import os
import random

import cv2
import numpy as np

YOLO_COLOR_MAP: dict[str, tuple[int, int, int]] = {}


def get_color_for_label(label: str) -> tuple[int, int, int]:
    """Get a consistent random BGR color for a given detection label.

    If the label has not been seen before, a new random color is generated
    and cached in ``YOLO_COLOR_MAP``. Subsequent calls with the same label
    return the cached color.

    Args:
        label: The detection class label (e.g. ``"person"``, ``"car"``).

    Returns:
        A BGR color tuple with each channel in the range [80, 255].
    """
    if label not in YOLO_COLOR_MAP:
        YOLO_COLOR_MAP[label] = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )
    return YOLO_COLOR_MAP[label]


def debug_draw_yolo(
    frame: np.ndarray,
    detections: list[dict],
    save_path: str | None = None,
) -> np.ndarray:
    """Draw YOLO detections on a frame for debugging.

    Each detection is drawn as a bounding box with a label and confidence
    score.  If the detection includes a ``track_id``, it is appended to the
    label text.  A black padding border is added around the frame so that
    boxes near the edges remain fully visible.

    Args:
        frame: The BGR image (NumPy array) to draw on.  The original array
            is **not** modified; a padded copy is used instead.
        detections: A list of detection dictionaries.  Each dictionary must
            contain the keys ``"label"`` (str), ``"confidence"`` (float),
            and ``"bbox"`` (sequence of four numeric values
            ``[x1, y1, x2, y2]``).  An optional ``"track_id"`` (int) key
            is also supported.
        save_path: If provided, the annotated image is written to this
            path.  Parent directories are created automatically.

    Returns:
        The annotated image as a NumPy array (with padding).
    """
    pad = 20
    drawn = cv2.copyMakeBorder(
        frame, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    for det in detections:
        label, conf = det["label"], det["confidence"]
        x1, y1, x2, y2 = (int(v) + pad for v in det["bbox"])
        color = get_color_for_label(label)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

        track_id = det.get("track_id")
        text = (
            f"{label}#{track_id} {conf:.2f}"
            if track_id is not None
            else f"{label} {conf:.2f}"
        )
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(drawn, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            drawn, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(save_path, drawn)

    return drawn

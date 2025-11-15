# YOLO-Only Video Scene Analysis

This branch implements a **YOLO-only pipeline** for video scene analysis.

## Pipeline

1. Scene detection using `scenedetect`.
2. Frame sampling from each scene.
3. YOLOv8 inference on each sampled frame.

## Requirements

```bash
pip install -r requirements.txt

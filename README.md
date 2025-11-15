## Kairos Video Analysis Pipeline

This project implements a video understanding pipeline that combines:
- YOLOv8 object detection per frame
- BLIP image captioning per frame
- LLM-based segment captioning using Google Gemini (fuses BLIP captions + YOLO outputs to generate concise segment descriptions)

It also tracks execution time and memory usage for each stage.


### Features

- Scene detection & frame sampling
- Per-frame BLIP captions + YOLO detections
- Segment-level descriptions via Gemini LLM
- Metrics collection (time & memory per step)
- JSON output storing all intermediate and final results
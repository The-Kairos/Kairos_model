# CCTV Dogs Processing Report

## Issue Summary
The results for `.CCTV Dogs` appear empty (0 scenes, 0 whisper segments, 0 AST processing). This is **not a pipeline error**, but a result of intentional visual and audio filters.

## 1. Zero-Scene Detection (Static Visuals)
The `PySceneDetect` component found **0 scene changes** because the video is a static feed. 
- **Impact**: Without defined scenes, the pipeline has no time-windows to label.

## 2. Audio Silence Filtering
The pre-scan stage calculates a dynamic silence threshold.
- **Result**: The background "beeeee" noise is filtered out. 
- **Detection**: `has_speech` was `False` and `Scenes with audio` was `0/0`. 

## Conclusion
The pipeline correctly identifies this as a **static, non-verbal video** and skips processing to save compute. 

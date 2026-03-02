# Base Processing Optimization Strategies

The current bottleneck in the benchmarking pipeline is the **ASR stage** (~87% of time) due to sequential API calls per scene.

## 📍 Current Implementation: Scene-by-Scene
The system currently processes audio by cutting the video into small fragments before calling the API.

- **Scene Detection**: Uses **PySceneDetect** (Utility: `src.scene_cutting.get_scene_list`). It identifies visual cuts and returns a list of start/end timestamps.
- **Audio Extraction**: Uses **FFmpeg** (Utility: `src.audio_utils.extract_scene_audio_ffmpeg`). It extracts a `.wav` file for every single scene.
- **Sequential ASR**: The `process_base.py` loop calls the Azure Whisper API one-by-one for these files.

---

## 🚀 Optimization 1: ASR Workflow (High Impact)

### 1.1 Single-Call Transcription (Recommended)
Instead of calling the Whisper API for every scene, transcribe the **entire video once**.
- **Pros**: Reduces 75+ API calls to 1. Eliminates per-scene extraction overhead and network latency.
- **Implementation**: Request `verbose_json` or `vtt/srt` output from Whisper, which includes timestamps. Map these timestamps to your `base_data.json` scenes in post-processing.

### 1.2 Parallel API Execution
If you must process per scene, use a thread pool to overlap network I/O.
- **Mechanism**: `concurrent.futures.ThreadPoolExecutor`.

### 1.3 Local High-Speed Inference
Switching from the Azure Cloud API to local GPU-accelerated models.
- **Faster-Whisper**: Reimplementation using CTranslate2 (~4x faster).

## 🧵 Optimization 2: Pipeline Parallelization (Medium Impact)

- **Parallel Stages**: Run ASR (I/O bound) and YOLO (Compute bound) concurrently using `multiprocessing`.

## 📊 Performance Comparison

| Strategy | Current (Sequential) | Optimized (Single Call) |
| :--- | :--- | :--- |
| **ASR Logic** | Scene-by-Scene (FFmpeg cuts) | Full Video (Temporal Mapping) |
| **API Calls** | 75+ | 1 |
| **Total Time** | ~780s | ~120s |

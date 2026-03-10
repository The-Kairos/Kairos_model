# Comprehensive Pipeline Comparison: Titanic (1997)

This report provides a side-by-side comparison of the **Legacy Pipeline** (Local Laptop) vs. the **New Enterprise Pipeline** (Cloud L4 GPUs + Azure API) for a 3-hour 15-minute video.

## Visual Summary: 15.2 Hours -> 2.0 Hours

| Feature | Legacy Pipeline (Laptop) | New Enterprise Pipeline (L4 + Azure) | Speedup / Gain |
| :--- | :--- | :--- | :--- |
| **Environment** | Windows 11 / 16GB RAM / GTX 1660 Ti | Cloud Linux / 188GB RAM / 4x NVIDIA L4 | **Scalability** |
| **Total Process Time** | **15.2 Hours** | **2.0 Hours** | **~7.6x Faster** |
| **Video Length** | 3h 15m | 3h 15m | - |

---

## Component Breakdown

| Pipeline Stage | Legacy (Local) | New (Cloud/API) | Speedup | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Scene Detection** | 3.1 m | 2.8 m | 1.1x | PySceneDetect |
| **Clip Extraction** | 49.8 m | 20.1 m | **2.5x** | IO/FFmpeg Optimization |
| **Object Detection** | 1.0 h | 6.1 m | **10.0x** | GPU-Accelerated YOLOv8 |
| **VLM Captioning** | 5.3 h | 28.3 m | **11.2x** | Optimized BLIP Model |
| **ASR (Audio)** | 3.8 h | 6.8 m | **33.5x** | **Azure Whisper API** |
| **AST (Sound)** | 32.7 m | 19.9 m | 1.6x | Parallel Analysis |
| **Narrative Gen** | 2.0 h | 29.7 m | **4.0x** | Optimized Prompting |

---

## Memory & Hardware Efficiency

| Metric | Legacy (Laptop) | New (Cloud L4) | Improvement |
| :--- | :---: | :---: | :--- |
| **Peak System RAM** | 11.2 GiB | 22.1 GiB | **Infinite Runway** (188GB Available) |
| **GPU Utilization** | 100% (Near Crash) | 20-45% (Cool/Stable) | **High Stability** |
| **Audio Model** | Whisper Small (Local) | Whisper Medium (Azure API) | **Higher Accuracy** |

## Conclusion
The transition to the Cloud/Azure infrastructure has transformed a process that used to take nearly an entire day into one that completes in less than the time it takes to watch the movie. 

> [!TIP]
> **Key Driver**: The integration of the Azure Whisper API provided the single largest performance jump (**33x**), while the move to L4 GPUs effectively eliminated VLM and YOLO bottlenecks.

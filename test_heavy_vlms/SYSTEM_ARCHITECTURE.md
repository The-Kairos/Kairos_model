# System Architecture: Heavy VLM Benchmarking

This document details the internal workings of the benchmarking pipeline, focusing on execution flow, caching strategies, and resource management on Google Cloud VMs.

## 1. Execution Model: Isolated Subprocesses

Heavy VLMs (7B+ parameters) consume significant VRAM (12GB-20GB+). Standard Python memory management often fails to fully reclaim GPU memory between model swaps, leading to CUDA Out-of-Memory (OOM) errors.

### The Orchestrator Pattern
- **`run_benchmark.py`**: Acts as the orchestrator. It manages the loop over videos and models but **never loads a VLM itself**.
- **`run_single_vlm.py`**: Acts as an isolated worker. It is launched via `subprocess.run()` for every single (VLM, Video) pair.
- **Cleanup**: When the worker process exits, the OS reclaims all allocated VRAM, ensuring the next model starts with a clean slate.

## 2. Caching Strategy

The system uses a multi-layered caching strategy to maximize performance and minimize redundant API/GPU costs.

### A. Base Data Cache (`process_base.py`)
Base processing (Scene Detection, ASR, AST, YOLO) is heavy and expensive.
- **Location**: `test_heavy_vlms/results/base/<video_name>/base_data.json`
- **Behavior**: If this file exists, `run_benchmark.py` skips base processing and passes the existing path to the VLM worker.
- **Content**: Includes scene boundaries, Whisper transcriptions (ASR), AST audio events, and YOLO object detections.

## 3. Model Storage & Loading: Hugging Face & Local

The system relies on several deep learning models. Understanding where they are stored and how they are loaded is key to managing disk space and performance on the VM.

### A. Hugging Face Models (VLMs & AST)
Most models are downloaded and managed via the `transformers` library.
- **Cache Location**: `/home/usr_60302531_udst_edu_qa/.cache/huggingface/hub/`
- **Models using this**:
  - `llava-hf/llava-v1.6-vicuna-7b-hf` (~15GB)
  - `Salesforce/instructblip-vicuna-7b` (~15GB)
  - `llava-hf/LLaVA-Next-Video-7B-hf` (~15GB)
  - `MIT/ast-finetuned-audioset-10-10-0.4593` (AST model, ~0.5GB)
- **Loading Pattern**: We use `from_pretrained(model_id)`.
  - **First Run**: Downloads the weights from the HF Hub.
  - **Subsequent Runs**: Detects the weight in the cache and loads directly from the VM's local disk (NVMe/SSD), making startup much faster.

### B. Torch Hub Models (Silero VAD)
Voice Activity Detection (VAD) is used to strip speech before AST processing.
- **Cache Location**: `/home/usr_60302531_udst_edu_qa/.cache/torch/hub/`
- **Loading Pattern**: `torch.hub.load(repo_or_dir="snakers4/silero-vad", ...)`

### C. Local Weights (YOLO)
The YOLOv8 model for object detection is stored directly in the project directory for portability.
- **Location**: `test_heavy_vlms/yolov8n.pt`
- **Behavior**: Downloaded automatically on the first run of `process_base.py` if not present.

### D. Efficient Loading Techniques
To fit these "Heavy" models on the VM's GPU (e.g., NVIDIA L4 with 24GB VRAM):
1. **`device_map="auto"`**: Automatically places model layers on the optimal device (GPU 0).
2. **`torch_dtype="auto"`**: Uses the model's native precision (usually Float16 or BFloat16) to halve memory usage compared to standard Float32.
3. **Quantization (4-bit/8-bit)**: Used for `llava_video` to ensure it fits comfortably within 10GB-12GB of VRAM while leaving room for the video tensor.

### C. Audio Scene Caching
- **Location**: `test_heavy_vlms/results/base/<video_name>/audio/scene_XX.wav`
- **Behavior**: Individual scene audio clips are extracted once and reused for both ASR and AST processing.

## 3. GPU & Resource Management

### Device Mapping
We use `device_map="auto"` in all VLM loading scripts. This leverages the `accelerate` library to:
1.  Map model layers across available GPUs (if multiple exist).
2.  Offload to CPU/Disk if VRAM is insufficient (though we aim to stay within VRAM for speed).

### Resource Monitoring (`src/system_metrics.py`)
The system captures high-resolution metrics throughout execution:
- **RAM**: Tracking system memory usage via `psutil`.
- **GPU**: Tracking VRAM allocation via `pynvml`.
- **Timing**: Accurate `time.time()` deltas for every stage (Scene, ASR, AST, YOLO, VLM).

## 4. Processing Pipeline Details

### Process Base Flow
1.  **Scene Detection**: Uses `PySceneDetect` to find natural cuts.
2.  **ASR (Speech-to-Text)**: Uses Azure OpenAI Whisper API (or local Whisper) to transcribe speech per scene.
3.  **AST (Audio Spectrogram Transformer)**: Classifies environmental sounds (e.g., "applause", "music") to provide non-speech context.
4.  **YOLO Detection**: Runs V8 detection on sampled frames to identify objects for the VLM to focus on.

### VLM Inference Flow
1.  **Frame Sampling**: Extracts 2-8 frames per scene using `cv2`.
2.  **Context Construction**: Combines ASR + AST + YOLO data into a prompt alongside the frames.
3.  **Fusion**: The final JSON output is a "fused" representation of all data points plus the VLM description.

## 5. Directory Structure
```text
Kairos_model/
├── Videos/               # Source video files
├── src/                  # Core utility modules (metrics, sampling, etc.)
└── test_heavy_vlms/
    ├── results/          # Generated data
    │   ├── base/         # Cached ASR/AST/YOLO data
    │   └── <vlm_name>/   # VLM-specific outputs
    ├── run_benchmark.py  # Orchestrator
    ├── run_single_vlm.py # Isolated Worker
    └── test_<vlm>.py     # Model-specific implementations
```

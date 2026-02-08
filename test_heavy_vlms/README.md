# Heavy VLM Pipeline Testing Suite

This folder contains a comprehensive benchmarking suite to evaluate "heavyweight" VLMs (LLaVA, InternVL, Qwen-VL) as integrated components in the full video analysis pipeline.

## Pipeline Overview
The `main_test.py` script executes the following steps for each video-model pair:
1. **Scene Detection**: Identifies cuts.
2. **Audio API (ASR/AST)**: Transcribes speech and labels sounds using Azure/Google Cloud.
3. **YOLO Detection**: Performs object tracking for context.
4. **Heavy VLM Captioning**: Generates rich descriptions for representative frames.
5. **LLM Fusion**: Combines visual, audio, and object data into a semantic scene summary using Gemini.
6. **Metrics**: Records VRAM, duration, and system usage.

## Setup

1. Activate your virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r test_heavy_vlms/requirements.txt
   ```
3. Ensure `.env` is configured with `AZURE_OPENAI_*`, `GOOGLE_APPLICATION_CREDENTIALS`, and `GEMINI_API_KEY`.

## Running the Benchmark

To run the full pipeline test across all videos and all models:
```bash
python test_heavy_vlms/main_test.py
```

### Results
- Shared data: `results/_shared_base_data/`
- Per-VLM metrics: `results/{vlm_name}/{video_name}/pipeline_results.json`
- Final comparison: `vlm_metrics.json`

## Troubleshooting VM Issues

### 1. Out of Memory / Ctrl+Z
If you stopped the script with `^Z` (Ctrl+Z), the GPU memory is still locked. Run:
```bash
pkill -f python
```

### 2. Missing Dependencies
If you get `ModuleNotFoundError`, ensure the latest specialized libraries are installed:
```bash
pip install tiktoken timm torchvision torchaudio
```

### 3. File Path Errors
If you get `Errno 2: describe_scene.txt not found`, ensure you have synced the latest `src/scene_description.py` which now uses absolute paths.

### 4. Syncing changes to VM
If your VM code doesn't match the local code:
1. PUSH from local machine: `git add . && git commit -m "update" && git push`
2. PULL on VM: `git pull origin main`

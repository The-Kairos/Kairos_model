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

## Results & Metrics
- **Results Folder**: Detailed JSON results per video are saved in `test_heavy_vlms/results/{vlm_name}/{video_name}/`.
- **Metrics Summary**: A consolidated benchmark report is saved to `test_heavy_vlms/vlm_metrics.json`.

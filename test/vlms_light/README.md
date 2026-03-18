# Light VLM Pipeline Testing Suite

Same code structure and pipeline as `test_heavy_vlms`, but uses **light** VLMs for comparison. Uses the **same videos** (project `Videos/` folder) and same benchmarks so you can compare performance logs with heavy VLMs.

## Light models

| Name (id) | Description |
|-----------|-------------|
| **BLIP-2 OPT** (`blip2`) | Salesforce BLIP-2 OPT 2.7B – image captioning (OPT small only) |
| **SigLIP** (`siglip`) | Retrieval-first: best-matching template description |
| **MobileVLM** (`mobilevlm`) | Meituan MobileVLM V2 1.7B – fast vision-language for mobile devices |
| **TinyLLaVA** (`tinyllava`) | TinyLLaVA Phi-2-SigLIP 3.1B – small multimodal with strong performance |

## Videos

Videos are read from the **same folder** as heavy VLMs:

- **Path:** `Kairos_model/Videos/` (project root)
- Place `.mp4` files there (same as for `test_heavy_vlms`). Any `.mp4` not starting with `_` is included.

## Setup

**One-time install (all modules):**

```bash
# From project root
pip install -r test_light_vlms/requirements_full.txt
```

Or use the install script:
- **Windows:** `test_light_vlms\install_deps.bat`
- **Linux/Mac:** `bash test_light_vlms/install_deps.sh`

If you use a venv, activate it first:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r test_light_vlms/requirements_full.txt
```

2. Ensure `.env` has `AZURE_OPENAI_*`, `GOOGLE_APPLICATION_CREDENTIALS`, and `GEMINI_API_KEY` (same as heavy pipeline).

**MobileVLM** requires the MobileVLM repo. Install with:
```bash
pip install git+https://github.com/Meituan-AutoML/MobileVLM.git
```
Or clone the repo and add it to `PYTHONPATH`.

## Resource utilization

The pipeline is configured to use all available CPU and GPU resources:

- **CPU threads:** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. are set to the number of logical CPUs. PyTorch also uses `torch.set_num_threads()`.
- **GPU:** cuDNN benchmark mode is enabled for faster convolutions. Models use `device_map="auto"` for automatic GPU placement.
- **Parallel cache building:** Use `--cache-workers N` to process N videos in parallel during pre-processing (scene detection, audio, YOLO). Default is 1; increase if you have enough GPU memory (each worker loads YOLO).

```bash
python test_light_vlms/main_test.py --cache-workers 2
```

## Run everything (main entry)

From project root:

```bash
python test_light_vlms/main_test.py
```

This will:

- Run the full pipeline (scene detection → audio → YOLO → **light VLM** captioning → LLM fusion) for each **video × light VLM** pair.
- Write **per-video results** under `vlms_light/results/<vlm_name>/<video_name>/pipeline_results.json`.
- Write **per-video aggregates** (all VLMs for one video) to `vlms_light/results/by_video/<video>/all_metrics.json`.
- Write **aggregate metrics** to `vlms_light/light_vlm_metrics.json`.
- Build a **summary table** at `vlms_light/results/summary_table.md` (duration, scenes, GPU, CPU, memory per run).
- Build a **pivot table** at `vlms_light/results/summary_pivot.md` (videos × VLMs for side-by-side comparison).

## Shared pre-processing cache

Scene detection, audio (ASR + AST), and YOLO are computed **once per video** and saved under `test_light_vlms/cache/`. When you run different light VLMs on the same videos, these steps are skipped and loaded from cache, so only the VLM captioning step is redone for each model.

- **Cache path:** `test_light_vlms/cache/<video_stem>_preproc.json`
- **Audio intermediates:** `test_light_vlms/cache/audio/<video_stem>/`

## Results layout

```
vlms_light/
  cache/               # Shared pre-processing (scene cuts, audio, YOLO)
  results/
    blip2/
    siglip/
    mobilevlm/
    tinyllava/
      <video1>/
        pipeline_results.json
      <video2>/
        pipeline_results.json
    by_video/             # Per-video aggregated metrics (all VLMs for one video)
      <video1>/
        all_metrics.json
      <video2>/
        all_metrics.json
    summary_table.md      # Full metrics table (duration, scenes, GPU, CPU, memory)
    summary_pivot.md      # Pivot: videos × VLMs for side-by-side comparison
  light_vlm_metrics.json  # Full metrics JSON
```

## Quick single-video test

To test one light VLM on one video (first few scenes only, no full pipeline):

```bash
python test_light_vlms/test_videos.py
```

Edit the script to change `target_video` and `model_to_test` (`blip2`, `siglip`, `mobilevlm`, `tinyllava`).

## Comparing with heavy VLMs

- **Heavy results:** `test_heavy_vlms/results/`, `test_heavy_vlms/vlm_metrics.json`
- **Light results:** `vlms_light/results/`, `vlms_light/light_vlm_metrics.json`, `vlms_light/results/summary_table.md`, `vlms_light/results/summary_pivot.md`

Use the same videos in `Videos/` and compare duration, GPU usage, and scene counts across the two suites.

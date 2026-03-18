# Light VLM Pipeline Testing Suite

Same code structure and pipeline as `test_heavy_vlms`, but uses **light** VLMs for comparison. Uses the **same videos** (project `Videos/` folder) and same benchmarks so you can compare performance logs with heavy VLMs.

## Light models

| Name (id) | Description |
|-----------|-------------|
| **SigLIP** (`siglip`) | Retrieval-first: best-matching template description |
| **BLIP-2 (OPT small)** (`blip2`) | Salesforce BLIP-2 OPT 1.3B image captioning |
| **MobileVLM** (`mobilevlm`) | Mobile-friendly captioning VLM for general scenes |
| **TinyLLaVA** (`tinyllava`) | Tiny LLaVA-style multimodal captioning model |

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

## Run everything (main entry)

From project root:

```bash
python test_light_vlms/main_test.py
```

This will:

- Run the full pipeline (scene detection → audio → YOLO → **light VLM** captioning → LLM fusion) for each **video × light VLM** pair.
- Write **per-video results** under `vlms_light/results/<vlm_name>/<video_name>/pipeline_results.json`.
- Write **aggregate metrics** to `vlms_light/results/light_vlm_metrics.json`.
- Build a **summary table** at `vlms_light/results/summary_table.md` (duration, scenes, GPU, CPU, memory per run, plus by-video VLM comparison).

## Results layout

```
vlms_light/
  results/
    <vlm_name>/
      <video1>/
        pipeline_results.json
      <video2>/
        pipeline_results.json
    light_vlm_metrics.json  # Aggregate metrics JSON
    summary_table.md       # Summary table (all runs + by-video VLM comparison)
```

## Quick single-video test

To test the light VLM on one video (first few scenes only, no full pipeline):

```bash
python test_light_vlms/test_videos.py
```

Edit the script to change `target_video` and `model_to_test` (`siglip`, `mobilevlm`, `tinyllava`, `blip2`).

## Comparing with heavy VLMs

- **Heavy results:** `test_heavy_vlms/results/`, `test_heavy_vlms/vlm_metrics.json`
- **Light results:** `vlms_light/results/` (per-video JSON, `light_vlm_metrics.json`, `summary_table.md`)

Use the same videos in `Videos/` and compare duration, GPU usage, and scene counts across the two suites.

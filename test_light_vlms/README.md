# Light VLM Pipeline Testing Suite

Same code structure and pipeline as `test_heavy_vlms`, but uses **light** VLMs for comparison. Uses the **same videos** (project `Videos/` folder) and same benchmarks so you can compare performance logs with heavy VLMs.

## Light models

| Name (id) | Description |
|-----------|-------------|
| **BLIP-2** (`blip2`) | Salesforce BLIP-2 OPT 2.7B image captioning |
| **InstructBLIP** (`instructblip`) | Instruction-following captioning (flan-t5-xl) |
| **LLaVA v1.6 Mistral 7B** (`llava_mistral`) | LLaVA with Mistral 7B backend |
| **Phi-3.5 Vision** (`phi3_vision`) | Microsoft Phi-3.5 vision (OCR-oriented prompts) |
| **SigLIP** (`siglip`) | Retrieval-first: best-matching template description |

## Videos

Videos are read from the **same folder** as heavy VLMs:

- **Path:** `Kairos_model/Videos/` (project root)
- Place `.mp4` files there (same as for `test_heavy_vlms`). Any `.mp4` not starting with `_` is included.

## Setup

1. Activate venv and install deps:

   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r test_light_vlms/requirements.txt
   ```

2. Ensure `.env` has `AZURE_OPENAI_*`, `GOOGLE_APPLICATION_CREDENTIALS`, and `GEMINI_API_KEY` (same as heavy pipeline).

## Run everything (main entry)

From project root:

```bash
python test_light_vlms/main_test.py
```

This will:

- Run the full pipeline (scene detection → audio → YOLO → **light VLM** captioning → LLM fusion) for each **video × light VLM** pair.
- Write **per-video results** under `test_light_vlms/results/<vlm_name>/<video_name>/pipeline_results.json`.
- Write **aggregate metrics** to `test_light_vlms/light_vlm_metrics.json`.
- Build a **summary table** at `test_light_vlms/results/summary_table.md` (duration, scene count, GPU usage per run).

## Results layout

```
test_light_vlms/
  results/
    blip2/
      <video1>/
        pipeline_results.json
      <video2>/
        pipeline_results.json
    instructblip/
      ...
    summary_table.md      # Table summary of all runs
  light_vlm_metrics.json  # Full metrics JSON
```

## Quick single-video test

To test one light VLM on one video (first few scenes only, no full pipeline):

```bash
python test_light_vlms/test_videos.py
```

Edit the script to change `target_video` and `model_to_test` (`blip2`, `instructblip`, `llava_mistral`, `phi3_vision`, `siglip`).

## Comparing with heavy VLMs

- **Heavy results:** `test_heavy_vlms/results/`, `test_heavy_vlms/vlm_metrics.json`
- **Light results:** `test_light_vlms/results/`, `test_light_vlms/light_vlm_metrics.json`, `test_light_vlms/results/summary_table.md`

Use the same videos in `Videos/` and compare duration, GPU usage, and scene counts across the two suites.

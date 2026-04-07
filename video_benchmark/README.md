# Video QA benchmark

Fair comparison harness for long-video understanding backends (same videos, same questions). You only maintain three JSON files under `data/` plus optional YAML tweaks under `config/`.

## Young Sheldon (Kairos comparison import)

If you already have `log_reports/comparison_results/Young_Sheldon_-_First_Day_of_High_School.mp4_comparison.json`, generate the benchmark dataset and a frozen Kairos response file:

```bash
python scripts/import_kairos_comparison.py --input ../log_reports/comparison_results/Young_Sheldon_-_First_Day_of_High_School.mp4_comparison.json --strategy flat
```

Then point `data/datasets/young_sheldon/video_info.json` at your local copy of the video (`file` is relative to `data/`). Run other vendors on the same questions:

```bash
python scripts/run_benchmark.py --dataset dataset.young_sheldon --systems google_gemini --skip-eval
```

Replay Kairos answers from the import (no pipeline rerun):

```bash
# Enable kairos_recorded in config/systems.yaml, then:
python scripts/run_benchmark.py --dataset dataset.young_sheldon --systems kairos_recorded --skip-eval
```

`answers.json` is seeded with Kairos text as `gold_answer` for convenience; replace with human-verified answers if your main metric is accuracy vs ground truth (see `annotation_status` in each row).

## Quick start

```bash
cd video_benchmark
pip install -r requirements.txt
# Optional: copy .env with API keys (repo root .env is also loaded)
python scripts/run_benchmark.py --systems mock --skip-eval
```

With evaluation (requires `ANTHROPIC_API_KEY` for the judge):

```bash
python scripts/run_benchmark.py --systems mock
```

Re-score an existing run without re-querying vendors:

```bash
python scripts/evaluate.py --responses outputs/raw/mock/<timestamp>_responses.json
```

Latency-only aggregation (no judge API):

```bash
python scripts/evaluate.py --responses outputs/raw/mock/<file>.json --skip-judge
```

## What you edit

| File | Purpose |
|------|---------|
| `data/metadata/video_info.json` | `video_id`, path to file (`file` relative to `data/`), `category` (`long` / `medium`), optional `duration_seconds` |
| `data/queries/queries.json` | `question_id`, `video_id`, `question`, `type`, optional `difficulty` |
| `data/ground_truth/answers.json` | Same `question_id` + `video_id`, `gold_answer`, optional `acceptable_variants`, optional `relevant_segment_ids` for recall@K |

`config/systems.yaml` — enable backends and credentials:

- **mock** — no network; echoes questions (pipeline test).
- **google_gemini** — `GEMINI_API_KEY`; uses `google-genai` upload + `generate_content`.
- **twelve_labs** — `TWELVE_LABS_API_KEY`, `TWELVE_LABS_INDEX_ID`; indexes via task then `POST /analyze`.
- **sentrysearch** — `HttpJsonSystem`: set `SENTRYSEARCH_BASE_URL` and adjust `upload` / `ask` templates to match your API.

`config/evaluation.yaml` — judge model name, temperature, optional retrieval `k_values`.

## Outputs

- `outputs/raw/<system>/<run_id>_responses.json` — one row per (video, question): `response`, `latency_sec`, optional `ranked_segment_ids`, `gold`, `error`.
- `outputs/evaluated/<system>_<run_id>_scored.json` — same rows plus `verdict` / `explanation` when the judge ran, plus aggregate `metrics`.
- `outputs/raw/<system>/upload_state.json` — caches remote upload ids per local file fingerprint; use `--fresh-upload` to ignore.

## Metrics

- **Primary:** share of `correct` verdicts (and optional partial-credit score) from the LLM judge.
- **Latency:** mean / median / stdev of `latency_sec` per question.
- **Optional recall@K / precision@K:** if a system returns `ranked_segment_ids` and gold lists `relevant_segment_ids`, aggregates are computed in `scripts/evaluate.py`.

## Single system shorthand

```bash
python scripts/run_single_system.py google_gemini --skip-eval
```

## Hybrid evaluation (recommended)

Automate query runs + LLM judging; manually write gold answers and spot-check a sample of judge outputs so rubric drift does not dominate your results.

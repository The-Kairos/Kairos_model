# Benchmarking Remote Handoff

## Purpose

This file is the short handoff for continuing Kairos benchmarking on a remote machine or in a new Codex chat.

Read this first together with:
- `tehreem_current_benchmarking_plan.md`
- `log_reports/BENCHMARKING_PUBLICATION_NOTE.md`

## Current Status

Kairos benchmarking is active on two tasks:
- `SceneWalk` for scene-level description evaluation
- `TIB` for full-video synopsis evaluation

The current work is focused on `SceneWalk` prompt tuning in a publishable way.

## Methodology Rules

Follow these rules strictly:

1. Do not treat prompt-tuned development runs as final benchmark evidence.
2. Never edit an already-tested benchmark prompt in place.
3. Save every prompt iteration as a new versioned prompt file.
4. Record every tested prompt version and its metrics.
5. Freeze the final prompt before running held-out videos.
6. Do not overfit to the wording of the 2-video SceneWalk development subset.

## Current Development Baseline

Pre-versioning reference:
- `Matched BERTScore F1`: `0.5723`
- `Matched ROUGE-L F1`: `0.1771`
- `SODA F1`: `0.0686`

## Current Versioned Prompt Candidate

Current candidate:
- `v1`

Files:
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`

## `v1` Development Results

From the completed 2-video SceneWalk development run:
- `Matched BERTScore F1`: `0.5745`
- `Matched ROUGE-L F1`: `0.2208`
- `SODA F1`: `0.0859`
- `SODA Precision`: `0.0553`
- `SODA Recall`: `0.1925`
- `Total matched pairs`: `113`

Interpretation:
- `v1` is better than the pre-versioning baseline
- biggest gains are in `SODA` and `ROUGE-L`
- `BERTScore` improved slightly
- recommendation is to try one more controlled prompt version (`v2`) before freezing

## Files That Matter Most

- `tehreem_current_benchmarking_plan.md`
- `log_reports/BENCHMARKING_PUBLICATION_NOTE.md`
- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- `prompts/describe_scene_short.txt`
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`
- `test/benchmarks/run_scenewalk_benchmark.py`
- `test/benchmarks/dataload/scenewalk_loader.py`
- `test/benchmarks/metrics/bertscore_metric.py`
- `test/benchmarks/results/scenewalk_benchmark_report.md`
- `test/benchmarks/results/scenewalk_comparison.md`

## Important Repo Fixes Already Made

- `SceneWalk` loader now prefers `.env` `HF_TOKEN`
- `yt-dlp` is configured to use `node`
- `yt-dlp` is configured with `--remote-components ejs:github`
- `BERTScore` now falls back to CPU when CUDA is unavailable

## What To Do Next

1. Read `tehreem_current_benchmarking_plan.md`
2. Read `log_reports/BENCHMARKING_PUBLICATION_NOTE.md`
3. Inspect `test/benchmarks/results/scenewalk_comparison.md`
4. Create a new prompt version `v2`
5. Copy `v2` into the active prompt files
6. Run the same 2-video SceneWalk development benchmark
7. Compare `v2` vs `v1`
8. Decide whether to freeze the prompt or iterate once more

## Paper Position

The publishable story is:
- use the 2-video SceneWalk run as a development set for prompt selection
- freeze the best prompt version
- run held-out SceneWalk videos for final paper results
- report `BERTScore` as the primary semantic metric
- report `SODA` as the temporal alignment metric
- report `ROUGE-L` as a secondary overlap metric

Do not present the development-tuned run as the final held-out benchmark.

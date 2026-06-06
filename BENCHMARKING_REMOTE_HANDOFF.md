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

## Current Frozen Prompt Candidate

Current best development candidate:
- `v3`

Files:
- `prompts/benchmark_versions/describe_scene_v2.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v2.txt`
- `prompts/benchmark_versions/describe_scene_short_v2.txt`
- `prompts/benchmark_versions/describe_scene_v3.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v3.txt`
- `prompts/benchmark_versions/describe_scene_short_v3.txt`

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

## `v2` Development Results

From the completed 2-video SceneWalk development run:
- `Matched BERTScore F1`: `0.5771`
- `BERTScore Precision`: `0.6015`
- `BERTScore Recall`: `0.5556`
- `Matched ROUGE-L F1`: `0.2095`
- `SODA F1`: `0.0815`
- `SODA Precision`: `0.0525`
- `SODA Recall`: `0.1827`
- `Total matched pairs`: `113`

Interpretation:
- `v2` improves the primary `BERTScore F1` over both the pre-versioning baseline and `v1`
- `v2` lowers `SODA` and `ROUGE-L` compared with `v1`, but both remain above the pre-versioning baseline
- `v2` raises BERTScore precision and lowers recall, consistent with a stricter, more conservative visual prompt
- `v2` is not frozen because the target is a stronger BERTScore before held-out evaluation

## `v3` Development Results

From the completed 2-video SceneWalk development run:
- `Matched BERTScore F1`: `0.5836`
- `BERTScore Precision`: `0.5933`
- `BERTScore Recall`: `0.5750`
- `Matched ROUGE-L F1`: `0.2256`
- `SODA F1`: `0.0878`
- `SODA Precision`: `0.0566`
- `SODA Recall`: `0.1968`
- `Total matched pairs`: `113`

Interpretation:
- `v3` is the best development candidate so far
- it improves BERTScore, SODA, and ROUGE-L over both `v1` and `v2`
- it restores recall by using a SceneWalk-like structure: setting, subject, action, shot progression, and key objects
- remaining risk is that some outputs include too much transcript/audio detail, so any `v4` should preserve the `v3` structure while tightening visible-only audio use

## `v4` Development Results

From the completed 2-video SceneWalk development run:
- `Matched BERTScore F1`: `0.5830`
- `BERTScore Precision`: `0.6006`
- `BERTScore Recall`: `0.5671`
- `Matched ROUGE-L F1`: `0.2284`
- `SODA F1`: `0.0886`
- `SODA Precision`: `0.0570`
- `SODA Recall`: `0.1986`
- `Total matched pairs`: `113`

Interpretation:
- `v4` preserves `v3`'s SceneWalk-like visual structure while reducing dialogue/audio carryover
- it improves `SODA` and `ROUGE-L` slightly compared with `v3`
- it raises BERTScore precision but lowers recall, so BERTScore F1 drops slightly from `0.5836` to `0.5830`
- if selecting by primary BERTScore F1, `v3` remains the best candidate so far

## Files That Matter Most

- `tehreem_current_benchmarking_plan.md`
- `log_reports/BENCHMARKING_PUBLICATION_NOTE.md`
- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- `prompts/describe_scene_short.txt`
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`
- `prompts/benchmark_versions/describe_scene_v2.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v2.txt`
- `prompts/benchmark_versions/describe_scene_short_v2.txt`
- `prompts/benchmark_versions/describe_scene_v3.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v3.txt`
- `prompts/benchmark_versions/describe_scene_short_v3.txt`
- `prompts/benchmark_versions/describe_scene_v4.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v4.txt`
- `prompts/benchmark_versions/describe_scene_short_v4.txt`
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
4. Decide whether to freeze `v3` or attempt a different non-overfitting improvement
5. If selecting by BERTScore F1, `v3` is currently the best development candidate
6. Once the best version is chosen, freeze it before held-out SceneWalk evaluation
7. Run held-out SceneWalk evaluation on additional unseen videos
8. Save held-out reports separately from development results

## Latest Aggregation Update

After prompt versions `v1` through `v4`, fixed-window aggregation was tested as a development-only scene-merging policy. The best current development configuration is:

- prompt/output source: `v4`
- aggregation policy: `fixed_window`
- window: `13s`
- max gap: `5s`
- result file: `test/benchmarks/results/scenewalk_results_20260606_104931.json`
- `Matched BERTScore F1`: `0.5879`
- `BERTScore Precision`: `0.5938`
- `BERTScore Recall`: `0.5834`
- `Matched ROUGE-L F1`: `0.2358`
- `SODA F1`: `0.1450`
- `Total matched pairs`: `129`

An abstractive aggregation rewrite was then tested using:

- rewrite prompt: `prompts/benchmark_versions/aggregate_scene_segments_v1.txt`
- result file: `test/benchmarks/results/scenewalk_results_20260606_110228.json`
- `Matched BERTScore F1`: `0.5777`
- `BERTScore Precision`: `0.6136`
- `BERTScore Recall`: `0.5468`
- `Matched ROUGE-L F1`: `0.2063`
- `SODA F1`: `0.1273`
- `Total matched pairs`: `129`

Interpretation:

- the rewrite trial completed on both development videos
- it should not be frozen because it reduced BERTScore F1, ROUGE-L, and SODA
- current best remains `v4 + fixed_window 13s`
- the rewrite result is still useful because it shows the likely ceiling is not solved by simply summarizing grouped predictions
- any further attempt to reach `0.6+` should be a small, predeclared aggregation-policy or evidence-selection ablation, followed by held-out evaluation after freezing

## Paper Position

The publishable story is:
- use the 2-video SceneWalk run as a development set for prompt selection
- freeze the best prompt version
- run held-out SceneWalk videos for final paper results
- report `BERTScore` as the primary semantic metric
- report `SODA` as the temporal alignment metric
- report `ROUGE-L` as a secondary overlap metric

Do not present the development-tuned run as the final held-out benchmark.

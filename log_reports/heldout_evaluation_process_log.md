# Held-Out SceneWalk Evaluation Process Log

This file tracks the currently running held-out SceneWalk evaluation using the frozen development-selected configuration.

## Run Configuration

- Codex tool session: `19594`
- working directory: `/home/usr_60302531_udst_edu_qa/Kairos_model`
- dataset: SceneWalk
- manifest: `test/benchmarks/cache/scenewalk_heldout_manifest.json`
- output cache: `test/benchmarks/cache/scenewalk_outputs_heldout/`
- excluded development videos:
- `mDvkux01G3A`
- `X9MAf245Yag`
- frozen prompt/output configuration: active `v4` prompts
- aggregation: `fixed_window`
- aggregation window: `13s`
- aggregation max gap: `5s`

Command:

```bash
python test/benchmarks/run_scenewalk_benchmark.py --max-videos 2 --manifest-name scenewalk_heldout_manifest.json --exclude-video-id mDvkux01G3A --exclude-video-id X9MAf245Yag --max-download-candidates 100 --output-cache-name scenewalk_outputs_heldout --aggregate-predictions fixed_window --aggregation-window-sec 13 --aggregation-max-gap-sec 5
```

## Status Entries

### 2026-06-06 15:20:14 UTC

- process/session status: still running in Codex tool session `19594`
- terminal output: continuing H.264 decoder warnings; no fatal error printed
- latest visible artifact status:
- `test/benchmarks/cache/scenewalk_outputs_heldout/video_000/checkpoint.json`
- `test/benchmarks/cache/scenewalk_outputs_heldout/video_000/synopsis.json`
- `test/benchmarks/cache/scenewalk_outputs_heldout/video_000/synopsis.md`
- `test/benchmarks/cache/scenewalk_outputs_heldout/video_001/checkpoint.json`
- video `000` checkpoint status:
- `310` scenes
- `310` scenes with `llm_scene_description`
- synopsis present
- video `001` status:
- checkpoint created, still processing
- metrics status:
- no new held-out `scenewalk_results_*.json` yet
- latest results file remains `test/benchmarks/results/scenewalk_results_20260606_110228.json`, which is the prior dev rewrite run, not held-out evaluation

### 2026-06-06 Current Status Check

- process/session status: still emitting output in Codex tool session `19594`
- shell `ps` status: no visible `run_scenewalk_benchmark.py` or `run_tib_benchmark.py` process from the current shell view
- SceneWalk held-out artifact status:
- video `000` has completed checkpoint and synopsis artifacts
- video `001` checkpoint exists but is still incomplete
- video `001` checkpoint details:
- `261` scenes detected
- `0` scenes with `llm_scene_description`
- synopsis not present
- metrics status:
- no held-out metrics JSON has been produced yet
- TIB status:
- no active TIB benchmark process found

### 2026-06-06 15:36:21 UTC

- process/session status: still running in Codex tool session `19594`
- metrics status: no held-out results JSON yet
- video `000` status:
- `310` scenes
- `310` scenes with `llm_scene_description`
- synopsis present
- video `001` status:
- `261` scenes
- `0` scenes with `llm_scene_description`
- synopsis not present
- checkpoint updated recently, so the run is still progressing before final LLM descriptions/metrics

### 2026-06-06 15:37:16 UTC

- process/session status: still running in Codex tool session `19594`
- metrics status: no held-out results JSON yet
- video `001` progress:
- `261` scenes detected
- `261` scenes with frame captions
- `252` scenes with YOLO detections
- `0` scenes with `llm_scene_description`
- synopsis not present
- interpretation: video `001` is still in pre-LLM pipeline stages; final metrics cannot run until LLM descriptions and synopsis finish

### 2026-06-06 15:38:44 UTC

- process/session status: still running in Codex tool session `19594`
- metrics status: no held-out results JSON yet
- video `001` checkpoint last updated at `2026-06-06 15:38:21 UTC`
- video `001` progress:
- `261` scenes detected
- `261` scenes with frame captions
- `252` scenes with YOLO detections
- `0` scenes with audio descriptions
- `0` scenes with speech transcripts
- `0` scenes with `llm_scene_description`
- synopsis not present
- interpretation: checkpoint is still updating, but final description generation/scoring has not started yet

### 2026-06-06 15:40:11 UTC

- process/session status: still running in Codex tool session `19594`
- metrics status: no held-out results JSON yet
- video `001` checkpoint last updated at `2026-06-06 15:40:03 UTC`
- video `001` progress:
- `261` scenes detected
- `261` scenes with frame captions
- `252` scenes with YOLO detections
- `261` scenes with `llm_scene_description`
- synopsis not present yet
- interpretation: LLM scene descriptions for both held-out videos are now complete; run should proceed to synopsis and metric computation next

### 2026-06-06 15:40:48 UTC

- process/session status: completed with exit code `0`
- held-out videos:
- `c0VPJWt_f0w`
- `NkMWgw6hNrE`
- excluded development videos:
- `mDvkux01G3A`
- `X9MAf245Yag`
- frozen configuration:
- active `v4` scene prompts
- `fixed_window` aggregation
- `13s` window
- `5s` max gap
- result file:
- `test/benchmarks/results/scenewalk_results_20260606_154048.json`
- generated report:
- `test/benchmarks/results/scenewalk_benchmark_report.md`
- generated comparison:
- `test/benchmarks/results/scenewalk_comparison.md`
- dedicated qualitative comparison:
- `log_reports/scenewalk_heldout_description_comparison.md`
- `log_reports/scenewalk_heldout_description_comparison.json`
- final held-out metrics:
- `Matched BERTScore F1`: `0.5886`
- `BERTScore Precision`: `0.5885`
- `BERTScore Recall`: `0.5903`
- `Matched ROUGE-L F1`: `0.2305`
- `SODA F1`: `0.1382`
- `SODA Precision`: `0.1029`
- `SODA Recall`: `0.2107`
- `Total matched pairs`: `129`

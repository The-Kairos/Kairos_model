# QVHighlights Benchmark — File Manifest

**Branch:** `eval`
**Date:** 2026-06-29
**Benchmark:** QVHighlights Moment Retrieval (Lei et al., NeurIPS 2021)
**Split:** test (1,542 queries, 1,529 videos)
**Final Results:** R1@0.5 = 38.91%, mAP Avg = 20.64% (zero-shot)

---

## Results (Final)

| File | Size | Description |
|------|------|-------------|
| [qvhighlights_results_MERGED_20260628_152004.json](qvhighlights_results_MERGED_20260628_152004.json) | 4 KB | Final official metrics — R1@0.5, R1@0.7, mAP@0.5, mAP@0.75, mAP Avg, length-bucket breakdowns. `match_number=true`, 1,542/1,542 queries. |
| [qvhighlights_predictions_MERGED_20260628_152004.jsonl](qvhighlights_predictions_MERGED_20260628_152004.jsonl) | 404 KB | All 1,542 predictions in official Moment-DETR submission format (`qid`, `query`, `vid`, `pred_relevant_windows`, `pred_saliency_scores`). Reproducible evaluation input. |
| [qvhighlights_benchmark_report.md](qvhighlights_benchmark_report.md) | 3 KB | Summary comparison table — Kairos vs. all Table 3 baselines (MCN, CAL, CLIP, XML, XML+, Moment-DETR). Key findings and source verification. |
| [qvhighlights_comprehensive_analysis.md](qvhighlights_comprehensive_analysis.md) | 34 KB | Full publishable analysis: dataset background, task definition, metric formulas, eval code verification, all 6 baseline method descriptions with references, split verification with paper quotes, validity analysis, caveats, and conclusion. |

## Benchmark Code

| File | Size | Description |
|------|------|-------------|
| [run_qvhighlights_benchmark.py](run_qvhighlights_benchmark.py) | 36 KB | Main benchmark runner. Handles video extraction, Kairos scene retrieval, prediction formatting, batch merging, and official metric evaluation. Includes corrected Table 3 baseline numbers. |
| [../../dataload/qvhighlights_loader.py](../../dataload/qvhighlights_loader.py) | 16 KB | Data loader — downloads QVHighlights annotations, extracts videos from tarball, prepares test split data for benchmarking. |
| [run_qvh_test_benchmark.sh](run_qvh_test_benchmark.sh) | 4 KB | Shell script for batch execution with `extract`, `batch`, `merge`, `all` subcommands. Manages parallelized batch runs. |

## Evaluation Code (Official)

| File | Size | Description |
|------|------|-------------|
| [../../metrics/qvhighlights/standalone_eval/eval.py](../../metrics/qvhighlights/standalone_eval/eval.py) | 12 KB | Official Moment-DETR evaluation code (adapted from [jayleicn/moment_detr](https://github.com/jayleicn/moment_detr)). Functions: `eval_submission()`, `eval_moment_retrieval()`, `compute_mr_ap()`, `compute_mr_r1()`. |
| [../../metrics/qvhighlights/standalone_eval/utils.py](../../metrics/qvhighlights/standalone_eval/utils.py) | 8 KB | IoU computation utilities: `compute_temporal_iou_batch_cross()`, `compute_temporal_iou_batch_paired()`, `compute_average_precision_detection()`. |
| [../../metrics/qvhighlights/moment_retrieval_metric.py](../../metrics/qvhighlights/moment_retrieval_metric.py) | — | Kairos-side metric wrapper that calls `standalone_eval` for QVHighlights evaluation. |

## Supporting Documentation

| File | Size | Description |
|------|------|-------------|
| [../../../../log_reports/qvhighlights_clip_retrieval_benchmark.md](../../../../log_reports/qvhighlights_clip_retrieval_benchmark.md) | 16 KB | Evaluation strategy document — defines clip retrieval approach, metric mapping, and benchmark design decisions. |
| [../../../../log_reports/qvhighlights_metrics_explainer.md](../../../../log_reports/qvhighlights_metrics_explainer.md) | 12 KB | Metric explainer — detailed walkthrough of R1@0.5, mAP@0.5, and mAP Avg with worked examples. |

---

## Key Settings

- **top_k:** 5
- **max_pred_windows:** 10
- **scene_merging:** gap = 5.0s
- **clip_length:** 2s (QVHighlights native segmentation)
- **match_number:** True (exact query coverage)
- **IoU thresholds:** np.linspace(0.5, 0.95, 10)

## Cache (gitignored, on-disk only)

| Path | Description |
|------|-------------|
| `cache/qvhighlights/qvhilights_videos.tar.gz` | Original tarball (~134 GB) |
| `cache/qvhighlights/qvh_videos/` | Extracted 150s video clips (~35 GB) |
| `cache/qvhighlights/qvhighlights_test_outputs/` | Per-video Kairos scene outputs (~3 GB) |
| `cache/qvhighlights/highlight_test_with_gt.jsonl` | Official test-split ground truth (1,541 entries) |

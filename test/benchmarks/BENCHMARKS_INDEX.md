# Kairos Benchmarks — Master Index

This document maps every benchmark file so any result, runner, or data loader can be found quickly.

---

## Directory Structure

```
test/benchmarks/
├── BENCHMARKS_INDEX.md                 ← you are here
├── dataload/                           ← dataset downloaders & loaders
│   ├── qvhighlights_loader.py
│   ├── scenewalk_loader.py
│   └── tib_loader.py
├── metrics/
│   ├── qvhighlights/                   ← QVHighlights-specific metrics
│   │   ├── moment_retrieval_metric.py
│   │   └── standalone_eval/            ← official Moment-DETR eval code
│   │       ├── eval.py
│   │       └── utils.py
│   ├── scenewalk/                      ← SceneWalk-specific metrics
│   │   └── soda_metric.py
│   ├── tib/                            ← TIB-specific metrics
│   │   └── bleu_metric.py
│   └── shared/                         ← shared across benchmarks
│       ├── bertscore_metric.py
│       └── rouge_metric.py
├── results/
│   ├── qvhighlights/                   ← QVHighlights results, runner, reports
│   ├── scenewalk/                      ← SceneWalk results, runner, reports
│   └── tib/                            ← TIB results, runner, reports
└── cache/                              ← (gitignored) downloaded data
    ├── qvhighlights/                   ← tarball, videos, annotations, outputs
    ├── scenewalk/                      ← outputs, manifests, version backups
    └── tib/                            ← outputs, manifests, videos
```

---

## 1. QVHighlights — Moment Retrieval

**Paper:** Lei et al., "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries", NeurIPS 2021 ([arXiv:2107.09609](https://arxiv.org/abs/2107.09609))
**Task:** Given a text query and a video, localize temporal moment windows
**Split:** test (1,542 queries, 1,529 videos)
**Kairos Result:** R1@0.5 = 38.91%, mAP Avg = 20.64% (zero-shot)

### Results & Reports

| File | Description |
|------|-------------|
| [results/qvhighlights/qvhighlights_results_MERGED_20260628_152004.json](results/qvhighlights/qvhighlights_results_MERGED_20260628_152004.json) | Final official metrics (R1@0.5, R1@0.7, mAP@0.5, mAP@0.75, mAP Avg). 1,542 queries, `match_number=true`. |
| [results/qvhighlights/qvhighlights_predictions_MERGED_20260628_152004.jsonl](results/qvhighlights/qvhighlights_predictions_MERGED_20260628_152004.jsonl) | All 1,542 predictions in Moment-DETR submission format. |
| [results/qvhighlights/qvhighlights_benchmark_report.md](results/qvhighlights/qvhighlights_benchmark_report.md) | Summary comparison table — Kairos vs. Table 3 baselines. |
| [results/qvhighlights/qvhighlights_comprehensive_analysis.md](results/qvhighlights/qvhighlights_comprehensive_analysis.md) | Full publishable analysis (34 KB): dataset, metrics, all baselines, validity assessment. |
| [results/qvhighlights/QVHIGHLIGHTS_BENCHMARK_MANIFEST.md](results/qvhighlights/QVHIGHLIGHTS_BENCHMARK_MANIFEST.md) | Detailed file manifest with sizes and settings. |

### Runner & Loader

| File | Description |
|------|-------------|
| [results/qvhighlights/run_qvhighlights_benchmark.py](results/qvhighlights/run_qvhighlights_benchmark.py) | Main benchmark runner (extraction, retrieval, evaluation). |
| [results/qvhighlights/run_qvh_test_benchmark.sh](results/qvhighlights/run_qvh_test_benchmark.sh) | Shell script for batch execution. |
| [dataload/qvhighlights_loader.py](dataload/qvhighlights_loader.py) | Downloads annotations & videos, prepares test split. |

### Metrics

| File | Description |
|------|-------------|
| [metrics/qvhighlights/standalone_eval/eval.py](metrics/qvhighlights/standalone_eval/eval.py) | Official Moment-DETR evaluation (R1, mAP). |
| [metrics/qvhighlights/standalone_eval/utils.py](metrics/qvhighlights/standalone_eval/utils.py) | IoU computation utilities. |
| [metrics/qvhighlights/moment_retrieval_metric.py](metrics/qvhighlights/moment_retrieval_metric.py) | Kairos-side MR metric wrapper. |

### Related Log Reports

| File | Description |
|------|-------------|
| [../../log_reports/qvhighlights_clip_retrieval_benchmark.md](../../log_reports/qvhighlights_clip_retrieval_benchmark.md) | Evaluation strategy and design decisions. |
| [../../log_reports/qvhighlights_metrics_explainer.md](../../log_reports/qvhighlights_metrics_explainer.md) | Metric definitions with worked examples. |
| [../../log_reports/qvhighlights_full_val_benchmark_plan.md](../../log_reports/qvhighlights_full_val_benchmark_plan.md) | Original val-split benchmark plan (superseded by test-split run). |

### Cache (gitignored, on-disk only)

| Path | Description |
|------|-------------|
| `cache/qvhighlights/qvhilights_videos.tar.gz` | Original tarball (~134 GB) |
| `cache/qvhighlights/qvh_videos/` | Extracted 150s video clips (~35 GB) |
| `cache/qvhighlights/qvhighlights_test_outputs/` | Per-video Kairos scene outputs (~3 GB) |
| `cache/qvhighlights/highlight_test_with_gt.jsonl` | Official test-split ground truth (1,541 entries) |
| `cache/qvhighlights/highlight_val_release.jsonl` | Val-split annotations |

---

## 2. SceneWalk — Scene Description Quality

**Paper:** IVLLab/SceneWalk
**Task:** Generate scene-level descriptions and compare against human-annotated temporal segments
**Split:** held-out (2 videos, 129 matched pairs)
**Kairos Result:** SODA F1 = 0.138, BERTScore F1 = 0.589, ROUGE-L F1 = 0.231

### Results & Reports

| File | Description |
|------|-------------|
| [results/scenewalk/scenewalk_benchmark_report.md](results/scenewalk/scenewalk_benchmark_report.md) | Aggregate metrics and per-video breakdown. |
| [results/scenewalk/scenewalk_comparison.md](results/scenewalk/scenewalk_comparison.md) | Detailed comparison analysis. |
| [results/scenewalk/scenewalk_comparison.json](results/scenewalk/scenewalk_comparison.json) | Raw comparison data (machine-readable). |
| [results/scenewalk/scenewalk_individual_predictions/](results/scenewalk/scenewalk_individual_predictions/) | 25 individual per-run result JSON files. |

### Runner & Loader

| File | Description |
|------|-------------|
| [results/scenewalk/run_scenewalk_benchmark.py](results/scenewalk/run_scenewalk_benchmark.py) | SceneWalk benchmark runner. |
| [dataload/scenewalk_loader.py](dataload/scenewalk_loader.py) | Downloads SceneWalk data and prepares splits. |

### Metrics

| File | Description |
|------|-------------|
| [metrics/scenewalk/soda_metric.py](metrics/scenewalk/soda_metric.py) | SODA scoring (temporal IoU matching + caption quality). |
| [metrics/shared/bertscore_metric.py](metrics/shared/bertscore_metric.py) | BERTScore evaluation (shared with TIB). |
| [metrics/shared/rouge_metric.py](metrics/shared/rouge_metric.py) | ROUGE-L evaluation (shared with TIB). |

### Related Log Reports

| File | Description |
|------|-------------|
| [../../log_reports/scenewalk_heldout_description_comparison.md](../../log_reports/scenewalk_heldout_description_comparison.md) | Held-out set description comparison analysis. |
| [../../log_reports/scenewalk_heldout_description_comparison.json](../../log_reports/scenewalk_heldout_description_comparison.json) | Raw comparison data for held-out set. |

### Cache (gitignored, on-disk only)

| Path | Description |
|------|-------------|
| `cache/scenewalk/scenewalk_outputs/` | Kairos scene outputs for SceneWalk videos |
| `cache/scenewalk/scenewalk_outputs_v*_backup_*/` | Version backups (v1, v2, v3) |
| `cache/scenewalk/scenewalk_manifest.json` | Video manifest |
| `cache/scenewalk/scenewalk_heldout_manifest.json` | Held-out split manifest |
| `cache/scenewalk/aggregation_rewrites/` | Aggregation rewrite cache |

---

## 3. TIB — Long-Form Video Summarization

**Paper:** TIB AV-Portal academic video archive
**Task:** Compare Kairos full-video synopsis against human-written abstracts
**Split:** 10 long-form presentations (60–113 min each)
**Kairos Result:** BERTScore F1 = 0.593, ROUGE-L F1 = 0.116, BLEU-1 = 0.080

### Results & Reports

| File | Description |
|------|-------------|
| [results/tib/tib_benchmark_report.md](results/tib/tib_benchmark_report.md) | Aggregate metrics and per-video breakdown (10 videos). |
| [results/tib/tib_comparison.md](results/tib/tib_comparison.md) | Detailed comparison analysis. |
| [results/tib/tib_comparison.json](results/tib/tib_comparison.json) | Raw comparison data (machine-readable). |
| [results/tib/tib_individual_predictions/](results/tib/tib_individual_predictions/) | 5 individual per-run result JSON files. |

### Runner & Loader

| File | Description |
|------|-------------|
| [results/tib/run_tib_benchmark.py](results/tib/run_tib_benchmark.py) | TIB benchmark runner. |
| [dataload/tib_loader.py](dataload/tib_loader.py) | Downloads TIB data and prepares evaluation set. |

### Metrics

| File | Description |
|------|-------------|
| [metrics/tib/bleu_metric.py](metrics/tib/bleu_metric.py) | BLEU scoring (TIB-specific). |
| [metrics/shared/bertscore_metric.py](metrics/shared/bertscore_metric.py) | BERTScore evaluation (shared with SceneWalk). |
| [metrics/shared/rouge_metric.py](metrics/shared/rouge_metric.py) | ROUGE-L evaluation (shared with SceneWalk). |

### Cache (gitignored, on-disk only)

| Path | Description |
|------|-------------|
| `cache/tib/tib_outputs/` | Per-video Kairos pipeline outputs |
| `cache/tib/tib_manifest.json` | Video manifest |
| `cache/tib/videos/` | Downloaded TIB videos |

---

## Cross-Benchmark Log Reports

| File | Description |
|------|-------------|
| [../../log_reports/benchmarking_final_results.md](../../log_reports/benchmarking_final_results.md) | Combined final results across all benchmarks. |
| [../../log_reports/moment_retrieval_dataset_comparison.md](../../log_reports/moment_retrieval_dataset_comparison.md) | Comparison of moment retrieval datasets. |

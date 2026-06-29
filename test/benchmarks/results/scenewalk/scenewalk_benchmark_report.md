# SceneWalk Benchmark Report — Kairos

**Dataset:** SceneWalk (IVLLab/SceneWalk)
**Date:** 2026-06-06
**Videos:** 2

**Prediction source:** `scenewalk_outputs_heldout`
**Manifest:** `scenewalk_heldout_manifest.json`
**Aggregation:** `fixed_window`

---

## Aggregate Metrics

| Metric                    |   Score |
|---------------------------|---------|
| SODA F1 (ROUGE-L scorer)  |  0.1382 |
| SODA Precision            |  0.1029 |
| SODA Recall               |  0.2107 |
| Matched BERTScore F1      |  0.5886 |
| Matched BERTScore Precision |  0.5885 |
| Matched BERTScore Recall  |  0.5903 |
| Matched ROUGE-L F1        |  0.2305 |
| Total Matched Pairs       |     129 |

---

## Per-Video Breakdown

| # | Video ID | Duration | Kairos Scenes | Raw Scenes | GT Segments | Matched | SODA F1 |
|---|----------|----------|---------------|------------|-------------|---------|---------|
| 1 | c0VPJWt_f0w | 31 min | 154 | 310 | 71 | 70 | 0.1467 |
| 2 | NkMWgw6hNrE | 30 min | 134 | 261 | 70 | 59 | 0.1296 |
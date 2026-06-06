# SceneWalk Benchmark Report — Kairos

**Dataset:** SceneWalk (IVLLab/SceneWalk)
**Date:** 2026-06-06
**Videos:** 2

**Prediction source:** `scenewalk_outputs`
**Aggregation:** `fixed_window`

---

## Aggregate Metrics

| Metric                    |   Score |
|---------------------------|---------|
| SODA F1 (ROUGE-L scorer)  |  0.1273 |
| SODA Precision            |  0.0924 |
| SODA Recall               |  0.2055 |
| Matched BERTScore F1      |  0.5777 |
| Matched BERTScore Precision |  0.6136 |
| Matched BERTScore Recall  |  0.5468 |
| Matched ROUGE-L F1        |  0.2063 |
| Total Matched Pairs       |     129 |

---

## Per-Video Breakdown

| # | Video ID | Duration | Kairos Scenes | Raw Scenes | GT Segments | Matched | SODA F1 |
|---|----------|----------|---------------|------------|-------------|---------|---------|
| 1 | mDvkux01G3A | 38 min | 193 | 301 | 80 | 78 | 0.1143 |
| 2 | X9MAf245Yag | 32 min | 106 | 166 | 51 | 51 | 0.1403 |
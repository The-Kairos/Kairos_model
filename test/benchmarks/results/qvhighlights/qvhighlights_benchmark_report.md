# QVHighlights Benchmark — Kairos Moment Retrieval

**Dataset:** QVHighlights (Lei et al., NeurIPS 2021, [arXiv:2107.09609](https://arxiv.org/abs/2107.09609))  
**Date:** 2026-06-28  
**Split:** test (same as paper Table 3)  
**Videos evaluated:** 1,529  
**Queries evaluated:** 1,542 (`match_number=True`)  
**Top-K:** 5  
**Scene Merging:** Yes (gap=5.0s)  
**Evaluation:** Official Moment-DETR `standalone_eval` code

---

## Moment Retrieval — Comparison with Paper Table 3 Baselines

All numbers below from Table 3 of Lei et al. (2021), evaluated on the **QVHighlights test split**.

| Method | Training | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|----------|--------|--------|---------|----------|---------|
| MCN | Supervised | 11.41 | 2.72 | 24.94 | 8.22 | 10.67 |
| CAL | Supervised | 25.49 | 11.54 | 23.40 | 7.65 | 9.89 |
| CLIP | Zero-shot | 16.88 | 5.19 | 18.11 | 7.00 | 7.67 |
| XML | Supervised | 41.83 | 30.35 | 44.63 | 31.73 | 32.14 |
| XML+ | Supervised | 46.69 | 33.46 | 47.89 | 34.67 | 34.90 |
| Moment-DETR | Supervised | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| Moment-DETR w/ PT | Supervised + PT | 59.78 | 40.33 | 60.51 | 35.36 | 36.14 |
| **Kairos** | **Zero-shot** | **38.91** | **22.83** | **36.95** | **18.74** | **20.64** |

### Key Findings

- **Kairos vs CLIP (both zero-shot):** Kairos outperforms CLIP by **2.3x** on R1@0.5 (38.91% vs 16.88%) and **4.4x** on R1@0.7 (22.83% vs 5.19%).
- **Kairos vs supervised baselines:** Kairos zero-shot surpasses supervised MCN and CAL on all metrics. Approaches XML (41.83% R1@0.5) despite having no training data.
- **Kairos vs Moment-DETR (supervised):** Moment-DETR outperforms Kairos across all metrics, as expected given it is trained with moment-level supervision on QVHighlights.

---

## Moment Retrieval by GT Window Length

| Length Bucket | Kairos mAP Avg |
|--------------|----------------|
| Short (0-10s) | 5.37 |
| Middle (10-30s) | 21.34 |
| Long (30-150s) | 23.06 |
| Full (all) | 20.64 |

---

## Source Verification

- **Table 3 split:** Paper caption states "Baseline Comparison on QVHIGHLIGHTS **test split**" (page 8)
- **Baseline numbers:** Copied directly from Table 3 of [arXiv:2107.09609](https://arxiv.org/abs/2107.09609)
- **Evaluation code:** Official `standalone_eval/eval.py` from [Moment-DETR repo](https://github.com/jayleicn/moment_detr)
- **Ground truth:** `highlight_test_with_gt.jsonl` from the official data release

*Kairos operates fully zero-shot — no training or fine-tuning on QVHighlights. CLIP baseline uses TAG watershed temporal grouping for moment retrieval. All supervised baselines are trained on the QVHighlights training split with moment-level annotations.*

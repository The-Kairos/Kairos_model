# Benchmark Research Index

**Date:** 2026-08-22
**Purpose:** Audit Kairos's QVHighlights benchmark positioning, identify gaps, and plan next steps for publication-grade evaluation.

---

## Documents

| # | File | Topic |
|---|------|-------|
| 1 | [01_QVHIGHLIGHTS_CURRENT_RESULTS.md](01_QVHIGHLIGHTS_CURRENT_RESULTS.md) | Our current QVHighlights results, what they mean, reproducibility status |
| 2 | [02_QVHIGHLIGHTS_PROBLEMS.md](02_QVHIGHLIGHTS_PROBLEMS.md) | Problems with QVHighlights as a benchmark for Kairos |
| 3 | [03_SOTA_LANDSCAPE_2026.md](03_SOTA_LANDSCAPE_2026.md) | Current state-of-the-art: supervised and zero-shot leaderboards with links |
| 4 | [04_SIMILAR_PIPELINE_SYSTEMS.md](04_SIMILAR_PIPELINE_SYSTEMS.md) | Systems with architectures similar to Kairos (describe-then-retrieve) |
| 5 | [05_CANDIDATE_BENCHMARKS.md](05_CANDIDATE_BENCHMARKS.md) | Datasets with active leaderboards we should consider benchmarking on |
| 6 | [06_IMPROVEMENT_STRATEGIES.md](06_IMPROVEMENT_STRATEGIES.md) | Concrete strategies to improve Kairos's moment retrieval performance |

---

## Key Takeaways (TL;DR)

1. **Our QVHighlights baselines are from 2021.** The field has moved 2x since then. Comparing only against CLIP and Moment-DETR will not survive peer review in 2026.

2. **Kairos sits mid-pack among zero-shot methods** (R1@0.5=38.91, mAP Avg=20.64). Training-free SOTA is now mAP Avg=40.32 (REZE, 2026). We're 2x above CLIP but 2x below current best.

3. **QVHighlights is the wrong benchmark for Kairos's strength.** Its videos are 150 seconds. Kairos handles 7-hour videos. Long-video benchmarks (MAD, Ego4D NLQ) would better showcase the pipeline.

4. **The "describe-then-retrieve" pipeline is not novel in 2026.** VTG-GPT, LLoVi, LangRepo, and the ACL 2026 MAGMaR winner all use similar architectures. Kairos's novelty is in multimodal fusion depth + scene-level segmentation + long-video scale.

5. **Kairos has processed videos up to 7 hours** (Web Summit Qatar, 7h03m). The longest QVHighlights video is 2.5 minutes. This mismatch is the biggest gap in our evaluation story.

# Benchmark Research Index

**Date:** 2026-08-27
**Purpose:** Audit Kairos's QVHighlights benchmark positioning, identify gaps, and plan next steps for publication-grade evaluation (journal paper).

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
| 7 | [07_BENCHMARK_ALTERNATIVES_HONEST_ASSESSMENT.md](07_BENCHMARK_ALTERNATIVES_HONEST_ASSESSMENT.md) | Honest assessment of MAD, Video-MME, HourVideo + synopsis evaluation metrics (CAPTURE, Video-ChatGPT, VC-Inspector) |
| -- | [QVHIGHLIGHTS_TEAM_GUIDE.md](QVHIGHLIGHTS_TEAM_GUIDE.md) | Comprehensive Q&A-style team reference for the full QVHighlights benchmarking methodology |

---

## Key Takeaways (TL;DR)

1. **Our QVHighlights baselines are from 2021.** The field has moved 2x since then. Comparing only against CLIP and Moment-DETR will not survive peer review in 2026. Updated comparison tables with 2025-2026 methods are in the team guide (Section 6).

2. **Kairos sits mid-pack among zero-shot methods** (R1@0.5=38.91, mAP Avg=20.64). Training-free SOTA is now mAP Avg=40.32 (REZE, 2026). We're 2x above CLIP but 2x below current best.

3. **QVHighlights is the best MR benchmark available.** It has the most baselines to compare against, no access barriers, and every MR paper reports results on it. The 150s video length is fine because we test long-video capability through other evaluations. Alternatives (MAD, Ego4D NLQ) have access barriers or domain shift issues — see doc 07.

4. **The "describe-then-retrieve" pipeline is not novel in 2026.** VTG-GPT, LLoVi, LangRepo, Moment-GPT, and GranAlign all use similar architectures. Kairos's novelty is in multimodal fusion depth (BLIP + YOLO + Whisper + AST + LLM) + scene-level segmentation + long-video scale (7 hours).

5. **For synopsis evaluation, standard metrics fail Kairos.** ROUGE-L, BERTScore, BLEU, and CIDEr all penalize Kairos's rich descriptions. Better alternatives: Video-ChatGPT protocol (LLM-as-judge), CAPTURE metric (element-level matching), VC-Inspector (reference-free factual accuracy). See doc 07.

6. **Temporal offset metric confirms scene boundaries are the bottleneck.** Center offset is near zero (Kairos finds the right region), but ABE is 23.6s (scene boundaries don't match GT moment boundaries). Results in the team guide, Section 9.

---

## What was dropped and why

| Benchmark | Reason |
|-----------|--------|
| **MAGMaR** | Wrong task — cross-video corpus search across 110K videos, not within-video retrieval |
| **V-RAGBench** | Egocentric domain shift + Ego4D license friction + 5.4TB download |
| **SceneWalk / TIB** | BERTScore and ROUGE-L penalize Kairos's rich descriptions — metric mismatch, not bad output |

See [07_BENCHMARK_ALTERNATIVES_HONEST_ASSESSMENT.md](07_BENCHMARK_ALTERNATIVES_HONEST_ASSESSMENT.md) for the full breakdown.

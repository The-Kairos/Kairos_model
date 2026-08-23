# Candidate Benchmarks for Kairos

Evaluation of datasets with active leaderboards and suitability for demonstrating Kairos's strengths.

---

## Current Benchmarks (Already Evaluated)

| Benchmark | Task | Videos | Avg Length | Our Results |
|-----------|------|--------|-----------|-------------|
| QVHighlights | Moment Retrieval | 1,529 test | 150s | R1@0.5=38.91, mAP Avg=20.64 |
| SceneWalk | Scene Description | -- | -- | SODA F1=0.138, BERTScore F1=0.589 |
| TIB | Video Summarization | -- | -- | BERTScore F1=0.593, ROUGE-L F1=0.116 |

---

## Candidate 1: MAD (Movie Audio Descriptions) — RECOMMENDED

**Why this matters for Kairos:** Hour-long videos. This is where Kairos's long-video pipeline has a genuine advantage.

| Property | Value |
|----------|-------|
| **Task** | Moment retrieval in full-length movies |
| **Videos** | 650 movies, 384K sentence-level annotations |
| **Avg Length** | ~1.85 hours |
| **Max Length** | ~3 hours |
| **Domains** | 22 movie genres |
| **Leaderboard** | Repository-based (no web leaderboard) |
| **Access** | NDA required. Must source movies independently. |
| **GitHub** | https://github.com/Soldelli/MAD |
| **Paper** | [arXiv:2112.00431](https://arxiv.org/abs/2112.00431) |

### Current Results on MAD

| Method | Type | R@1 | R@5 | Avg |
|--------|------|-----|-----|-----|
| CLIP ZS | Zero-shot | 2.2% | -- | -- |
| CONE | Supervised | 6.87 | 16.11 | -- |
| RevisionLLM | Supervised | -- | -- | 14.4% |
| **P2S** | **Zero-shot** | -- | -- | **14.5%** |

**P2S (2025)** is the first zero-shot method to beat a supervised baseline on MAD. The numbers are low overall — MAD is hard.

### Feasibility for Kairos

- **Pro:** Kairos has processed 3h+ videos (Titanic). The pipeline should work.
- **Pro:** Low zero-shot baselines (CLIP=2.2%) — room to make an impact.
- **Pro:** P2S only gets 14.5% — Kairos could potentially compete.
- **Con:** NDA required for the dataset.
- **Con:** Must source 650+ movies independently (not downloadable).
- **Con:** Pipeline processing time: ~3-5 hours per movie. 650 movies = months of compute.
- **Con:** Scene boundaries in movies may not align with annotation granularity.

### Verdict: HIGH VALUE but HIGH COST

If the NDA and movie sourcing can be resolved, even benchmarking on a subset (e.g., 50 movies) would be valuable. The numbers would be more meaningful than QVHighlights for showing Kairos's long-video advantage.

---

## Candidate 2: Ego4D NLQ (Natural Language Queries) — RECOMMENDED

**Why this matters for Kairos:** Medium-length videos (8-20 min), active annual challenge, well-maintained.

| Property | Value |
|----------|-------|
| **Task** | Moment retrieval in egocentric video |
| **Videos** | ~74K queries across egocentric clips |
| **Avg Length** | 8-20 minutes per clip (hours full) |
| **Domains** | Daily activities (first-person) |
| **Leaderboard** | Active, annual CVPR/ECCV challenge |
| **Access** | License agreement required (~5.4 TB download) |
| **Website** | https://ego4d-data.org/ |
| **Paper** | [arXiv:2110.07058](https://arxiv.org/abs/2110.07058) |

### Current Results on Ego4D NLQ

| Method | Type | R@1 IoU=0.3 | R@1 IoU=0.5 |
|--------|------|-------------|-------------|
| CONE | Baseline | 14.15% | 8.18% |
| EgoVLP | Supervised | Challenge winner (CVPR 2022) | -- |
| R2-Tuning | Supervised (ECCV 2024) | Claims SOTA | -- |

### Feasibility for Kairos

- **Pro:** Videos are 8-20 minutes — between QVHighlights (2.5 min) and MAD (1.85 hrs). Good middle ground.
- **Pro:** Active leaderboard with annual challenges.
- **Pro:** Well-documented evaluation protocol.
- **Con:** Egocentric (first-person) video — significant domain shift from Kairos's third-person processing.
- **Con:** 5.4 TB download.
- **Con:** License agreement bureaucracy.
- **Con:** Kairos's visual pipeline (BLIP, YOLO) may perform poorly on first-person footage.

### Verdict: MEDIUM VALUE, MEDIUM COST

Worth considering if the domain shift isn't too damaging. The egocentric perspective might hurt Kairos's visual captioning quality significantly.

---

## Candidate 3: Charades-STA — LOW PRIORITY

| Property | Value |
|----------|-------|
| **Task** | Moment retrieval in indoor activity videos |
| **Videos** | ~6,672 videos, ~16K query-moment pairs |
| **Avg Length** | ~30 seconds |
| **Leaderboard** | Active on Papers With Code |
| **Access** | Freely downloadable |
| **Paper** | [arXiv:1705.02101](https://arxiv.org/abs/1705.02101) |

### Feasibility for Kairos

- **Pro:** Standard benchmark, expected in MR papers.
- **Pro:** Freely downloadable, well-documented.
- **Con:** Videos are only ~30 seconds — even shorter than QVHighlights. Kairos's scene segmentation would produce very few scenes per video.
- **Con:** Indoor activities only — narrow domain.

### Verdict: LOW VALUE for Kairos

Would add completeness to the paper but doesn't showcase any Kairos strength. Include only if reviewers demand it.

---

## Candidate 4: ActivityNet Captions — LOW PRIORITY

| Property | Value |
|----------|-------|
| **Task** | Moment retrieval / dense video captioning |
| **Videos** | ~20K videos, ~72K query-moment pairs |
| **Avg Length** | ~2 minutes |
| **Leaderboard** | Declining, ~30-40% video rot |
| **Access** | YouTube videos (many now unavailable) |
| **Paper** | [arXiv:1705.00754](https://arxiv.org/abs/1705.00754) |

### Verdict: LOW VALUE

Video rot makes reproducibility questionable. Length is similar to QVHighlights. Not worth the effort.

---

## Candidate 5: Video-MME / MLVU / HourVideo — FOR VIDEO UNDERSTANDING

These are **not moment retrieval benchmarks** but are relevant for Kairos's broader video understanding claims.

| Benchmark | Task | Video Length | Leaderboard |
|-----------|------|-------------|-------------|
| Video-MME | Video QA (multi-choice) | Short/Medium/Long | Active (Papers With Code) |
| MLVU | Long video understanding | Up to hours | Active |
| HourVideo | Hour-long video QA | Hours | Active |

### Relevance to Kairos

These benchmarks evaluate whether a system *understands* long videos, not whether it can localize moments. Kairos's scene descriptions + RAG chatbot could be evaluated here via the QA task. This would complement the MR evaluation by showing that the same pipeline supports multiple downstream tasks.

---

## Candidate 6: MAGMaR — NEW, HIGHLY RELEVANT

| Property | Value |
|----------|-------|
| **Task** | Video corpus retrieval + temporal grounding |
| **Leaderboard** | Active (ACL 2026 Workshop) |
| **Significance** | The #1 system uses a pipeline very similar to Kairos |

The ACL 2026 MAGMaR Workshop has a retrieval leaderboard where the winning system ("Decoupling Semantics and Logic") uses chunk → describe → retrieve → rerank. This is Kairos's architecture. Worth investigating the evaluation protocol and data requirements.

---

## Recommended Benchmark Strategy

### For the capstone paper (minimum viable):
1. **Keep QVHighlights** but update the comparison table with Moment-GPT, GranAlign, REZE, UniVTG ZS
2. **Add honest positioning** — Kairos is mid-pack among zero-shot methods, with distinctive multimodal depth

### For a journal paper (comprehensive):
1. Everything above, plus:
2. **Add MAD** (even a subset of 50 movies) to demonstrate long-video capability
3. **Add Video-MME or MLVU** to show the pipeline generalizes beyond MR
4. **Reference MAGMaR** to position Kairos in the Video RAG context

### Stretch goals:
- Ego4D NLQ (if domain shift isn't fatal)
- Charades-STA (for completeness if reviewers ask)

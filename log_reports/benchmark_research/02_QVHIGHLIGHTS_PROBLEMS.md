# Problems with QVHighlights as a Benchmark for Kairos

---

## Problem 1: Stale Baselines

Our comparison table uses baselines from the original 2021 paper (MCN, CAL, CLIP, XML, Moment-DETR). The field has advanced dramatically:

- **Supervised SOTA** has gone from Moment-DETR's 52.89 R1@0.5 (2021) to UniTime-SP's **77.76** (2025) — a 47% relative improvement.
- **Zero-shot SOTA** has gone from CLIP's 16.88 R1@0.5 to GranAlign's **59.92** (AAAI 2026) and REZE's **40.32 mAP Avg** (2026).
- Comparing only against 2021 methods **will not survive peer review** in 2026.

### What Needs to Be Added to the Comparison Table

At minimum, these zero-shot/training-free methods:

| Method | Year/Venue | R1@0.5 | mAP Avg | Why include |
|--------|-----------|--------|---------|-------------|
| Moment-GPT | AAAI 2025 | 58.30 | 35.00 | Training-free pipeline, most comparable |
| GranAlign | AAAI 2026 | 59.92 | 38.23 | Current training-free R1 SOTA |
| REZE | 2026 | -- | 40.32 | Current training-free mAP SOTA |
| UniVTG ZS | ICCV 2023 | 25.16 | 10.87 | Widely cited zero-shot baseline |

Adding these would show Kairos at **mid-pack** rather than near the top.

---

## Problem 2: QVHighlights Videos Are Too Short

QVHighlights videos are exactly **150 seconds** (2.5 minutes). Kairos has been tested on:

| Video | Duration |
|-------|----------|
| Web Summit Qatar 2026 Day Three | **7h 03m** |
| Titanic (1997) | **3h 14m** |
| UDST Graduation | **2h 22m** |
| Learning: SVMs | 49m |
| NYC Manhattan Walk | 44m |

The pipeline's ability to handle hour-long+ videos is completely invisible in a 150-second benchmark. This is Kairos's biggest unreported strength.

---

## Problem 3: Temporal Granularity Mismatch

QVHighlights uses **2-second clips** as the base unit. Kairos uses **variable-length scenes** (typically 5-15 seconds). This creates a structural disadvantage:

- A 4-second ground truth moment gets IoU < 0.5 against a 12-second Kairos scene, even if the scene contains the right content.
- This is reflected in the numbers: **mAP Avg = 5.37 for short moments vs 23.06 for long moments** — a 4.3x gap.
- CLIP and trained models operate at 2-second granularity and don't have this disadvantage.

The scene-level approach is actually a feature (semantic coherence), but QVHighlights penalizes it.

---

## Problem 4: Single Embedding Space

Kairos uses **Gemini text embeddings** for both scene descriptions and queries. This means:
- The query ("a person opens a door") is matched against a generated text description, not directly against visual/audio features.
- If the scene description doesn't mention "door" but shows it visually, the retrieval fails.
- Newer methods like REZE directly score video clips against queries using VLMs, bypassing the description bottleneck.

---

## Problem 5: No Query Decomposition

Kairos treats the query as a single embedding. Newer methods decompose queries:
- **TFVTG** (ECCV 2024): LLM breaks "a man walks his dog then sits on a bench" into sub-events with temporal relations.
- **Moment-GPT** (AAAI 2025): LLaMA-3 debiases the query before matching.
- **GranAlign** (AAAI 2026): Generates queries at varied semantic granularity.

Kairos has none of this — the raw query string goes directly into the embedding model.

---

## Problem 6: No Proposal Refinement

Kairos's "proposals" are fixed scene boundaries + simple adjacent merging. It cannot:
- Predict sub-scene boundaries (e.g., the moment starts 3 seconds into a scene)
- Score partial overlaps
- Refine boundaries based on the query

Trained models and even newer zero-shot methods (REZE, VTimeCoT) can predict arbitrary start/end times within a video.

---

## What These Problems Mean for the Paper

These are **not fatal** — they're honest limitations to disclose. The paper's positioning should be:

1. Kairos is a **general-purpose video understanding system** (not a moment retrieval specialist)
2. Moment retrieval is **one downstream application** of its scene-level pipeline
3. Zero-shot performance is **competitive with pre-2024 methods** and **well above naive baselines**
4. The architecture enables **capabilities that MR benchmarks don't measure** (long-video synopsis, RAG chatbot, structured output)
5. The **long-video scalability** (7 hours) is a genuine contribution that needs a suitable benchmark

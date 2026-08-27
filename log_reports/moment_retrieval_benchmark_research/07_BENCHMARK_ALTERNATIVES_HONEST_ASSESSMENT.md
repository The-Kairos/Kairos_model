# Benchmark Alternatives for Kairos — Honest Assessment

> This document evaluates benchmark alternatives to QVHighlights for evaluating Kairos's moment retrieval capability. Each benchmark is broken down with paper references and a clear verdict.

---

## 1. MAD (Movie Audio Descriptions)

**Paper:** [arXiv:2112.00431](https://arxiv.org/abs/2112.00431) — "MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions"

### What it actually is

MAD is a **moment retrieval benchmark on full-length movies** — the same task as QVHighlights but on videos that are 1-3 hours long instead of 150 seconds.

### What the task actually requires

The system receives a **text query and a full movie** and must predict the **temporal window** [start, end] where the described event happens. This is exactly the same task as QVHighlights moment retrieval.

From the paper: *"short temporal moments (typically seconds long) must be accurately grounded in diverse long-form videos."*

- **Scale:** 384,000 natural language annotations across 1,200+ hours of video
- **Video length:** up to 3 hours per movie
- **Evaluation:** Standard R@K at IoU thresholds (same as QVHighlights)

### Why this would be ideal for Kairos

| Aspect | Assessment |
|--------|------------|
| **Same task as QVHighlights** | YES — predict [start, end] given a query and a video. Same metrics (R@K, mAP) |
| **Long videos** | YES — 1-3 hour movies. This is where Kairos's pipeline is designed to shine |
| **Kairos has done this** | YES — processed the Titanic (3+ hours). The pipeline handles long videos |
| **Low zero-shot baselines** | YES — CLIP zero-shot gets only 2.2%. Huge room for Kairos to show impact |
| **Standard evaluation** | YES — same R@K at IoU metrics as QVHighlights. Results directly comparable |

### Why it's hard to actually do

| Problem | Details |
|---------|---------|
| **Movie access** | You must source the movies yourself. They are copyrighted films — no download link provided |
| **NDA** | The dataset page requires an NDA/license agreement to access annotations |
| **Compute cost** | Processing one 2-hour movie through Kairos takes 3-5 hours. 650 movies = months of GPU time |
| **Few zero-shot comparisons** | Only CLIP ZS (2.2%) and P2S (14.5%) have published zero-shot results. Thin comparison table |

### Verdict: BEST TASK FIT, but ACCESS + COST is the barrier

MAD is the most natural benchmark for Kairos after QVHighlights — same task, same metrics, but on long videos where Kairos's scene pipeline actually matters. The barrier is accessing the movies and the compute cost. Even a subset (e.g., 20-50 movies) would be valuable.

---

## 2. Video-MME

**Paper:** [arXiv:2405.21075](https://arxiv.org/abs/2405.21075) — "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis"

### What it actually is

A **multiple-choice QA benchmark** for video understanding. The system watches a video and answers multiple-choice questions about it.

### What the task actually requires

- **Input:** A video + a multiple-choice question (4 options)
- **Output:** The letter of the correct answer (A, B, C, or D)
- **Videos:** Short (<2 min), Medium (4-15 min), Long (30-60 min) — 900 videos total
- **Questions:** 2,700 total across 6 perception categories

### Why this is relevant

| Aspect | Assessment |
|--------|------------|
| **Tests video understanding** | YES — tests whether a system actually understands what happens in a video |
| **Multiple video lengths** | YES — includes 30-60 minute videos |
| **Active leaderboard** | YES — well-maintained, many baselines |
| **Publicly available** | YES — downloadable from HuggingFace |
| **Easy evaluation** | YES — multiple choice, so accuracy is objective |

### Why it doesn't directly fit

| Problem | Details |
|---------|---------|
| **Multiple-choice QA** | Kairos would need to be adapted to answer A/B/C/D questions. Currently it generates free-form text |
| **Not moment retrieval** | Does not test temporal localization at all |
| **Tests a different capability** | Tests comprehension, not retrieval |

### Verdict: GOOD for showing Kairos understands videos, but DIFFERENT TASK

Would complement the MR evaluation by showing the same pipeline supports video understanding, not just clip retrieval. But requires adapting Kairos to answer multiple-choice questions.

---

## 3. HourVideo

**Paper:** [arXiv:2411.04998](https://arxiv.org/abs/2411.04998) — "HourVideo: 1-Hour Video-Language Understanding"

### What it actually is

A **QA benchmark specifically for hour-long videos** — 500 egocentric videos, each ~1 hour, with 12,976 multiple-choice questions.

### What the task actually requires

- **Input:** A ~1 hour egocentric video + a multiple-choice question
- **Output:** The letter of the correct answer
- **Question types:** Summarization, perception (spatial, temporal, appearance), reasoning (navigation, counting, causal)

### Relevance to Kairos

| Aspect | Assessment |
|--------|------------|
| **Hour-long videos** | YES — tests long-video capability |
| **QA task** | Tests comprehension, aligns with RAG chatbot capability |
| **Active benchmark** | YES |
| **Egocentric** | PROBLEM — first-person head-mounted camera footage. BLIP and YOLO were not designed for shaky, motion-heavy, unusual-angle video |

### Verdict: INTERESTING but EGOCENTRIC DOMAIN SHIFT

First-person footage may hurt BLIP/YOLO performance. Worth considering only if the domain shift can be tolerated.

---

## Synopsis / Scene Description — The Metric Problem

### Q: Why did SceneWalk and TIB fail for Kairos?

The benchmarks themselves were fine — the **metrics** were the problem. BERTScore, ROUGE-L, BLEU, CIDEr, and METEOR all measure surface-level text overlap between the generated description and a reference description. Kairos produces rich, multi-sentence paragraphs with visual details, objects, audio, and dialogue. The reference descriptions in SceneWalk/TIB are short, sparse summaries. Even when Kairos's description is accurate and detailed, the metrics score it low because the words don't match.

```
Reference:  "A woman walks down the street."
Kairos:     "A young woman in a red jacket walks briskly along a
             tree-lined sidewalk. A dog is visible on a leash beside
             her. Background audio includes traffic noise and birdsong.
             Dialogue: none detected."

ROUGE-L:  ~0.12  (low — most of Kairos's words aren't in the reference)
BERTScore: ~0.59  (mediocre — embeddings partially overlap but style differs)

Both scores say "bad." The description is actually more accurate and more detailed
than the reference. The metric is wrong, not the description.
```

### Q: What metrics would actually work?

The problem is that all traditional captioning metrics are **reference-based** and reward matching the reference style. Kairos needs either **reference-free metrics** (judge the description against the video itself) or **LLM-as-judge** (have a language model rate quality on defined dimensions).

| Approach | How it works | Reference needed? |
|----------|-------------|-------------------|
| **ROUGE-L, BLEU, CIDEr, METEOR** | Count n-gram overlap with reference text | YES — fails for Kairos |
| **BERTScore** | Cosine similarity between token embeddings of generated and reference text | YES — better but still penalizes extra detail |
| **CAPTURE** | Extracts visual elements (objects, attributes, relations) from both texts, matches at element level | YES — but matches meaning, not words |
| **CLIPScore** | CLIP similarity between video frames and generated text | NO — judges text-video alignment directly |
| **LLM-as-judge** (GPT-4 evaluation) | An LLM rates the description on defined dimensions (accuracy, detail, coherence) | NO — judges quality directly |

### Option 1: Video-ChatGPT Evaluation Protocol

**Paper:** [arXiv:2306.05424](https://arxiv.org/abs/2306.05424) — "Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models"

**What it is:** An evaluation protocol using GPT as a judge. 100 videos from ActivityNet with question-answer pairs. GPT rates responses on **5 dimensions**, each scored 1-5:

1. **Correctness of Information** — are the facts in the description accurate?
2. **Detail Orientation** — does the description capture specific details?
3. **Contextual Understanding** — does it understand the broader context?
4. **Temporal Understanding** — does it capture what happens over time?
5. **Consistency** — is the description internally consistent?

**Why this fits Kairos:**

| Aspect | Assessment |
|--------|------------|
| **Rewards detail** | YES — dimension 2 explicitly scores detail. Kairos's rich descriptions would score HIGHER, not lower |
| **Rewards accuracy** | YES — dimension 1 checks facts, not word overlap |
| **Temporal understanding** | YES — dimension 4 is exactly what scene descriptions should capture |
| **No reference style bias** | YES — GPT judges quality, not similarity to a short reference |
| **Accessible** | YES — evaluation code is public, videos from ActivityNet |
| **Widely used** | YES — adopted as standard evaluation by Video-LLaVA, LLaMA-VID, VideoChat2, and many others |

**What would need to happen:** Run Kairos on the 100 ActivityNet videos, generate scene descriptions, feed them to GPT-4 with the evaluation prompts from the paper, collect scores across 5 dimensions. Compare against published baselines.

**Limitation:** ActivityNet videos have some video rot (30-40% of YouTube links may be dead). Would need to check availability first. Also, GPT-as-judge has known biases (prefers longer, more verbose responses — which actually helps Kairos here, but reviewers may flag it).

### Option 2: Reference-Free Video-Text Alignment

Two approaches exist for measuring how well a description matches the actual video content, without needing a reference caption.

**A) CLIPScore** ([arXiv:2104.08718](https://arxiv.org/abs/2104.08718))

Measures cosine similarity between CLIP embeddings of video frames and the generated text. No reference needed.

**Critical limitation:** CLIP has a **hard 77-token limit** on text input. Kairos's scene descriptions are multi-sentence paragraphs — well beyond 77 tokens. CLIPScore would truncate most of the description and only score the first ~20 words. **This makes vanilla CLIPScore unsuitable for Kairos.**

**Workaround — Long-CLIP** ([arXiv:2403.15378](https://arxiv.org/abs/2403.15378), ECCV 2024): A plug-and-play replacement for CLIP that extends the token limit to support long text input. Achieves ~20% improvement on long-caption retrieval. Would fix the truncation problem but requires custom integration (not a drop-in metric).

**B) VC-Inspector** ([arXiv:2509.16538](https://arxiv.org/abs/2509.16538), ACL 2026)

A lightweight open-source model (Qwen2.5-VL 3B/7B with LoRA) specifically designed for **reference-free factual accuracy evaluation of video captions**. Unlike CLIPScore, it:

- Outputs a **quality score (1-5)** AND a **natural language explanation** identifying specific factual errors (wrong objects, wrong actions)
- Runs at **~0.30 seconds per video**
- Available on HuggingFace (3B and 7B versions)
- Outperforms GPT-4o-based evaluation on VATEX-Eval benchmark

| Aspect | CLIPScore | VC-Inspector |
|--------|-----------|-------------|
| **Reference-free** | YES | YES |
| **Handles long text** | NO — 77 token limit | YES |
| **Video input** | NO — image only, average per-frame | YES — native video |
| **Explains errors** | NO — just a number | YES — identifies specific mistakes |
| **Open-source** | YES | YES (HuggingFace) |
| **Compute cost** | Low | Low (~0.3s per video) |

**Recommendation:** VC-Inspector is the better reference-free option for Kairos. It handles video natively, supports long descriptions, and tells you exactly what's wrong (e.g., "the caption says 'two dogs' but only one dog is visible"). CLIPScore is not practical for paragraph-length text without Long-CLIP modification.

### Option 3: CAPTURE Metric (Built for Detailed Descriptions)

**Paper:** [arXiv:2405.19092](https://arxiv.org/abs/2405.19092) — "Benchmarking and Improving Detail Image Caption"

**What it is:** A metric specifically designed for evaluating **detailed, paragraph-length descriptions**. Instead of comparing words or n-grams, CAPTURE extracts visual elements (objects, attributes, relations) from both the generated and reference descriptions, then matches at the element level.

**How it works:**
1. Extract visual elements from the candidate description (e.g., "red jacket", "tree-lined sidewalk", "dog on leash")
2. Extract visual elements from the reference description (e.g., "woman", "street")
3. Match elements across the two sets using semantic similarity
4. Score based on how many elements match — extra detail is NOT penalized

**How it compares to other metrics** (Pearson correlation with expert human judgments):

| Metric | Correlation with humans |
|--------|------------------------|
| BLEU | 0.261 |
| ROUGE-L | 0.295 |
| CIDEr | 0.115 |
| METEOR | 0.402 |
| CLIPScore | 0.356 |
| **CAPTURE** | **0.509** |

CAPTURE has the highest correlation with human judgments — almost 2x better than ROUGE-L and 4.4x better than CIDEr.

**Why this fits Kairos:**

| Aspect | Assessment |
|--------|------------|
| **Designed for detailed descriptions** | YES — built specifically to evaluate paragraph-length captions with rich detail |
| **Does not penalize extra detail** | YES — matches at the element level, so extra objects/attributes don't hurt the score |
| **Easy to use** | YES — pip-installable (`pip install capture_metric`) |
| **Publicly available** | YES — Apache 2.0 license |

**Limitation:** Designed for images, not video. Would need to apply it per-scene (treat each scene's keyframes as the "image"). Still requires a reference description — but scores meaning overlap, not word overlap, so the style mismatch problem is much smaller.

**Comes with a benchmark:** DetailCaps-4870 (4,870 images, ~122 words per reference caption) — could serve as a comparison point for description quality, though it's image-based not video-based.

### Option 4: Design Our Own Evaluation Protocol

Since Kairos's descriptions are unique (multimodal: visual + objects + audio + dialogue), no existing benchmark was designed to evaluate this exact output. The most honest approach may be to **design a targeted evaluation** rather than forcing an existing benchmark:

1. **Sample 50-100 videos** from QVHighlights or another available dataset
2. **Run Kairos pipeline** to generate scene descriptions
3. **Have GPT-4 rate each description** on dimensions tailored to Kairos:
   - Visual accuracy (do the described visuals match the video?)
   - Object accuracy (are the detected objects actually present?)
   - Audio accuracy (does the audio description match?)
   - Temporal coherence (does the scene flow make sense?)
   - Completeness (are important elements captured?)
4. **Human spot-check** on a 20-video subset to validate GPT-4's ratings

**Why this fits:** It evaluates what Kairos actually produces without penalizing it for not matching a reference style. The dimensions can be designed to test each modality (visual, object, audio) separately.

**Limitation:** Not a published benchmark — reviewers may question validity. Mitigate by: (a) citing Video-ChatGPT as precedent for GPT-based evaluation, (b) including human validation, (c) publishing the evaluation protocol for reproducibility.

### Honest recommendation for synopsis

**Best practical option:** Video-ChatGPT evaluation protocol (Option 1). It's published, widely adopted by Video-LLaVA/LLaMA-VID/VideoChat2 and many others, rewards detail and accuracy rather than word overlap, and has baselines to compare against. Check ActivityNet video availability first.

**Best metric for reference-based evaluation:** CAPTURE (Option 3). If we need to compare against reference descriptions (e.g., SceneWalk references), CAPTURE is the only metric designed for paragraph-length text. It has 2x better human correlation than ROUGE-L and is pip-installable. Use this instead of BERTScore/ROUGE-L.

**Complementary reference-free signal:** VC-Inspector (Option 2B) alongside whichever primary metric we choose. It handles video natively, supports long text, and identifies specific factual errors — unlike CLIPScore which truncates at 77 tokens. Two independent signals are stronger than one.

**If ActivityNet videos are unavailable:** Design our own protocol (Option 4) using QVHighlights videos we already have, citing Video-ChatGPT's approach as methodological precedent.

**What to avoid:** Any benchmark that uses only ROUGE-L, BERTScore, BLEU, CIDEr, or METEOR as primary metrics. Every dense video captioning benchmark (ActivityNet Captions, YouCook2, ViTT, VidChapters) uses these same overlap metrics. None of them will work for Kairos's description style.

---

## What was dropped and why

| Benchmark | Why it was dropped |
|-----------|-------------------|
| **MAGMaR** (arXiv:2606.07924) | Wrong task entirely. Searches across 110,000 multilingual videos to find relevant ones — Kairos processes ONE video. Also requires persona-constrained generation, multilingual support, and submission to an organizer-controlled server. Not a downloadable benchmark |
| **V-RAGBench** (arXiv:2606.13141) | Egocentric video (first-person head-mounted camera) — too much domain shift for BLIP/YOLO. Requires Ego4D license agreement + 5.4TB download. Only 300 test queries. Fixed 2-minute chunks conflict with Kairos's scene-based approach |
| **SceneWalk / TIB** | Synopsis benchmarks. BERTScore and ROUGE-L penalize Kairos's rich multimodal descriptions because they don't match the reference style. These metrics measure surface-level text overlap, not description quality — the scores (BERTScore F1 ~0.59, ROUGE-L ~0.12) reflect a metric mismatch, not bad descriptions |

---

## Summary: What should Kairos actually benchmark on?

| Benchmark | Task | Fits Kairos? | Accessible? | Recommended? |
|-----------|------|-------------|-------------|--------------|
| **QVHighlights** | Moment retrieval (150s videos) | YES — same task | YES — free | YES — already done, keep it |
| **MAD** | Moment retrieval (1-3hr movies) | YES — same task, long videos | HARD — NDA + source movies | YES IF ACCESS CAN BE ARRANGED — even a 20-movie subset |
| **Video-ChatGPT eval** | Scene description quality (LLM-as-judge) | YES — rewards detail + accuracy | YES — public code + ActivityNet videos | YES — best option for synopsis |
| **CAPTURE metric** | Element-level description matching | YES — built for paragraph-length text | YES — pip install | YES — best reference-based metric for detailed descriptions |
| **VC-Inspector** | Reference-free factual accuracy (video caption) | YES — identifies specific errors | YES — HuggingFace | YES — best reference-free metric for synopsis |
| **Video-MME** | Multiple-choice video QA | PARTIAL — tests understanding, not MR | YES — HuggingFace | MAYBE — requires MC adapter, good for QA track |
| **HourVideo** | Hour-long video QA (egocentric) | PARTIAL — right length, wrong domain | MODERATE | MAYBE — only if egocentric domain shift is tolerable |

### Honest recommendation for the journal paper

**For moment retrieval:** QVHighlights is the best and most practical choice. MAD would be ideal as a second MR benchmark if movie access can be arranged — it directly tests Kairos on long videos with the same metrics.

**For synopsis / scene descriptions:** Use the Video-ChatGPT evaluation protocol (GPT-as-judge on 5 dimensions) as the primary metric, with VC-Inspector as a complementary reference-free factual accuracy check. If comparing against reference descriptions, use CAPTURE instead of BERTScore/ROUGE-L. Avoid any benchmark that uses only ROUGE-L, BERTScore, BLEU, or CIDEr — these all penalize Kairos's rich description style regardless of accuracy.

**For video understanding (QA):** Video-MME is the most accessible and well-maintained. Would require adapting Kairos to answer multiple-choice questions, but this is straightforward if the RAG pipeline already answers free-form questions.

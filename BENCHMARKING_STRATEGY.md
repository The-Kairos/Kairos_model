# Kairos Benchmarking Strategy for Journal Paper

## Context

Kairos produces rich, structured output (per-scene multimodal descriptions, synopses, timelines, highlights, Q&A, character descriptions, and RAG-based retrieval) that is fundamentally different from what standard video understanding reference datasets provide. Traditional reference-based metrics (ROUGE-L, BERTScore) penalize Kairos for being *more detailed* than short reference captions, making them unsuitable as primary metrics. We need a multi-tier evaluation strategy that is credible for peer-reviewed publication.

---

## Complete Benchmark Landscape

### A. Long-Form Video QA Benchmarks (MCQ-Based -- Directly Testable)

These benchmarks let Kairos process videos and answer MCQs via its RAG system. Accuracy is directly comparable to published baselines for GPT-4o, Gemini, and 20+ open-source models.

| Benchmark | Venue | Videos | Duration | Questions | Format | Metrics | Open Data |
|-----------|-------|--------|----------|-----------|--------|---------|-----------|
| **Video-MME** [1] | arXiv 2024 | 900 | 11s--1hr | 2,700 | 4-way MCQ | Accuracy (%) by duration + with/without audio/subtitles | Yes (HuggingFace, ~101GB) |
| **MLVU** [2] | **CVPR 2025** | varied | 3min--2hr | MCQ + free-form | 6-way MCQ + generation | M-Avg (MCQ), G-Avg (generation) | Yes (HuggingFace, CC-BY-NC-SA) |
| **HourVideo** [3] | **NeurIPS 2024** | 500 | 20--120min | 12,976 | 5-way MCQ | Accuracy (%). Human 85%, best model 37.3% | Yes (HuggingFace + EvalAI leaderboard) |
| **LongVideoBench** [4] | **NeurIPS 2024** | 3,763 | 8s--1hr | 6,678 | MCQ | Accuracy by duration bucket | Yes (HuggingFace, CC-BY-NC-SA) |
| **EgoSchema** [5] | NeurIPS 2023 | 5,000+ clips | 3min each | 5,000+ | 5-way MCQ | Accuracy (%). Human 76%, best model 33% | Yes (CC BY 4.0) |
| **CinePile** [6] | arXiv 2024 | 9,396 clips | movie scenes | 305,000 | 5-way MCQ | Accuracy. 86 question templates, hard split | Yes (HuggingFace, CC-BY-NC-SA) |
| **MovieChat-1K** [7] | **CVPR 2024** | 1,000 clips | 4--8.5min | 13K QA + 1K dense captions | Open-ended + MCQ | Accuracy + GPT-assisted Score | Yes (HuggingFace) |
| **LVBench** [8] | **ICCV 2025** | ~7 long videos | up to 2hr | MCQ | MCQ | Accuracy across 6 capability categories | Yes (HuggingFace, CC-BY-NC-SA) |

**Key tasks in MLVU that directly test Kairos outputs:**
- Sub-Scene Captioning (SSC) -- generate descriptions for video sub-segments
- Video Summary (VS) -- generate overall video summaries
- These are free-form generation tasks scored by G-Avg

**Key tasks in HourVideo that test Kairos:**
- Summarization (key events/objects, temporal sequencing, compare/contrast)
- Perception (factual recall, sequence recall, temporal distance)
- Visual reasoning (spatial, temporal, predictive, causal, counterfactual)

---

### B. Audio-Visual Benchmarks (Tests Kairos's Multimodal Pipeline)

Kairos's unique strength is its audio branch (Whisper + MIT AST). These benchmarks specifically test audio-visual understanding:

| Benchmark | Venue | Videos | Questions | What It Tests | Open Data |
|-----------|-------|--------|-----------|---------------|-----------|
| **Video-MME (audio split)** [1] | arXiv 2024 | 900 | 2,700 | Reports with/without audio -- can show audio contribution | Yes |
| **OmniEval** [9] | arXiv 2025 | 810 | 2,617 | Tasks CANNOT be solved by single modality -- requires true AV fusion | Yes (CC BY 4.0) |
| **LongShOTBench** [10] | arXiv 2025 | long-form | varied | Probes visual, speech, ambient-audio, temporal, cross-modal reasoning | Yes (CC-BY-NC-SA) |
| **OmniVideoBench** [11] | arXiv 2025 | 628 | 1,000 | Synergistic AV understanding, 13 question types, step-by-step reasoning | Planned |
| **AVQA** [12] | ACM MM 2022 | ~200K source | manual | AV QA across 8 semantic categories | Yes |
| **LVOmniBench** [13] | arXiv 2026 | 275 | 1,014 | Long-form AV (10--90min). Open-source models <35%, Gemini 3 Pro ~65% | Yes (CC BY 4.0) |
| **R-AVST** [14] | **AAAI 2026** | 5,000+ | 8,000+ | Spatio-temporal reasoning with 100 AV event types | Yes (GitHub) |

---

### C. Reference-Free Metrics (Replaces ROUGE/BERTScore)

These metrics evaluate description quality WITHOUT requiring reference text -- they compare directly against video content:

| Metric | Venue | How It Works | Reference-Free? | Best For |
|--------|-------|-------------|-----------------|----------|
| **EMScore** [15] | CVPR 2022 | CLIP-based video-text similarity at coarse + fine-grained levels | **Yes** | Video captioning quality. Higher human correlation than reference-based metrics |
| **PAC-S** [16] | **CVPR 2023 Highlight** | Augmented contrastive learning for image/video-text similarity | **Yes** | State-of-art correlation with human judgments on video captioning |
| **ViCLIP** [17] | arXiv 2023 | Video-native contrastive model (ViT-L, trained on 7M+ videos). Direct video-text similarity | **Yes** | Captures temporal semantics unlike frame-averaged CLIPScore |
| **CLIPScore** [18] | EMNLP 2021 | CLIP cosine similarity between image/frame and text embeddings | **Yes** | Simple baseline, but frame-averaged (no temporal) |
| **VideoScore** [19] | **EMNLP 2024** | Fine-tuned 8B model scoring 5 dimensions (visual quality, temporal consistency, factual consistency, etc.) | **Yes** | Multi-aspect quality assessment |

**Recommendation:** Use **EMScore** + **PAC-S** as primary reference-free metrics. Both are open-source, published at top venues, and explicitly designed for video captioning evaluation.

---

### D. LLM-as-Judge Frameworks

| Framework | Venue | LLM Used | Dimensions Scored | Scale | Precedent |
|-----------|-------|----------|-------------------|-------|-----------|
| **Video-ChatGPT Eval** [20] | **ACL 2024** | GPT-3.5-Turbo | Correctness, Detail, Context, Temporal Understanding, Consistency | 0--5 each | De facto standard. Used by Video-LLaVA, VideoChat2, many others |
| **G-Eval** [21] | EMNLP 2023 | GPT-4 | Chain-of-thought + form-filling for NLG | varies | Spearman 0.514 with human on summarization |
| **MVBench** [22] | **CVPR 2024 Highlight** | None (anti-LLM-judge) | 20 temporal tasks via auto-generated MCQ | Accuracy | Shows MCQ evaluation avoids LLM scoring biases |

---

### E. Hallucination Benchmarks (Evaluates Factual Accuracy)

| Benchmark | Year | Focus | How Kairos Can Use It |
|-----------|------|-------|----------------------|
| **VideoHallucer** [23] | 2024 | Intrinsic + extrinsic hallucinations in video LLMs. Adversarial binary QA | Test whether Kairos descriptions contain fabricated objects/events |
| **EventHallusion** [24] | 2024 | Event-level hallucination from language priors and vision-language biases | Test Kairos's temporal event accuracy |
| **HAVEN** [25] | 2025 | 6,000 questions across 3 hallucination dimensions. 16 LMMs evaluated | Comprehensive hallucination scoring |
| **Vript-HAL** [26] | **NeurIPS 2024** | Action + object hallucination in video LLMs. 12K videos, 420K clip captions | Test whether YOLO-detected objects match described objects |
| **FActScore** [27] | EMNLP 2023 | Decompose text into atomic facts, verify each. <2% error vs human | Decompose Kairos scene descriptions into verifiable claims |
| **NarrativeTrack** [28] | arXiv 2026 | Entity-centric narrative coherence. Tests character tracking across scenes | Evaluate Kairos's character identification in synopses |

---

### F. Dense Captioning / Summarization Datasets

| Dataset | Venue | What It Has | How Kairos Uses It |
|---------|-------|-------------|-------------------|
| **ActivityNet Captions** [29] | ICCV 2017 | 20K videos, 100K temporal captions | Compare scene boundary alignment + use LLM-as-judge for description quality |
| **YouCook2** [30] | AAAI 2018 | 2K cooking videos, step-by-step with temporal boundaries | Test temporal segmentation accuracy |
| **Vript** [26] | NeurIPS 2024 | 12K videos, 420K captions (avg 145 words), includes camera operations | Closest to Kairos's rich description format |

---

## Recommended Evaluation Strategy

### Pillar 1: QA Benchmarking on Established Leaderboards (Primary)

**Process benchmark videos -> Kairos pipeline -> RAG answers -> report accuracy**

**Pick 2--3 from:**
1. **Video-MME** [1] -- best for showing audio modality contribution (has with/without audio splits)
2. **MLVU** [2] -- best for testing scene descriptions + summaries directly (has SSC + VS free-form tasks). CVPR 2025 venue.
3. **HourVideo** [3] -- best for long-form (hour-long videos). Huge human-AI gap. NeurIPS 2024 + EvalAI leaderboard.

**How to run:**
1. Download videos from HuggingFace
2. Run Kairos pipeline: `python main.py process --video <path>`
3. For each MCQ, query `rag_convo.py` with the question text
4. Map free-form RAG answer to closest MCQ option (LLM classification or cosine similarity)
5. Report accuracy vs. published baselines in comparison table

---

### Pillar 2: Reference-Free Description Quality (Replaces ROUGE/BERTScore)

**Two complementary approaches:**

**A) Automated reference-free metrics:**
- **EMScore** [15] -- video-text alignment via CLIP embeddings
- **PAC-S** [16] -- augmented contrastive score, highest human correlation
- Run on same videos as Pillar 1. Compare Kairos descriptions vs. Gemini/GPT-4o direct video descriptions.

**B) LLM-as-Judge following Video-ChatGPT framework [20] (ACL 2024):**
- 5 dimensions: Correctness, Detail Orientation, Contextual Understanding, Temporal Understanding, Consistency
- Score 0--5 each via GPT-4o
- Run on 50--100 video subset
- Compare Kairos vs. competitors on same rubric

---

### Pillar 3: Ablation Study

Disable individual Kairos components, measure accuracy drop on same benchmark:

| Config | BLIP | YOLO | Whisper | MIT AST | Expected Insight |
|--------|------|------|---------|---------|-----------------|
| Full Kairos | Y | Y | Y | Y | Baseline |
| No Audio | Y | Y | N | N | Audio branch contribution |
| No Objects | Y | N | Y | Y | YOLO contribution |
| No Captions | N | Y | Y | Y | BLIP contribution |
| Visual Only | Y | Y | N | N | Visual vs. multimodal gap |

---

### Pillar 4: Competitor Comparison (Supporting)

Process same benchmark videos through:
- **Gemini 2.5 Pro** (direct video upload + same questions)
- **GPT-4o** (frame sampling + same questions)
- **TwelveLabs API** (if accessible)

Frame as: *"Kairos's modular pipeline vs. monolithic multimodal models"* -- a valid research contribution regardless of which wins, because it illuminates the architectural tradeoffs.

---

## Paper Evaluation Section Structure

```
5. Evaluation
   5.1 Experimental Setup
       - Hardware, models, hyperparameters
       - Benchmark datasets and selection rationale
       
   5.2 Video Question Answering Performance
       - Results on Video-MME / MLVU / HourVideo
       - Comparison table against published baselines (20+ models)
       - Analysis by video duration (hypothesis: Kairos improves on longer videos)
       - Audio modality analysis (Video-MME with/without audio split)
       
   5.3 Scene Description Quality
       - Reference-free metrics (EMScore, PAC-S) on benchmark videos
       - LLM-as-Judge rubric scores (Video-ChatGPT 5-dimension framework)
       - Kairos vs. Gemini vs. GPT-4o on same rubric
       - Inter-annotator agreement validation (LLM-judge vs. human subset on 20-30 videos)
       
   5.4 Ablation Study
       - Component contribution analysis (audio, visual, object detection)
       - Modality fusion analysis (each branch's impact on QA accuracy)
       
   5.5 Synopsis & Structured Output Evaluation
       - MLVU Sub-Scene Captioning and Video Summary tasks (G-Avg scores)
       - FActScore-style atomic fact decomposition on synopsis outputs
       - Timeline accuracy evaluation
       
   5.6 Qualitative Analysis
       - Side-by-side examples: Kairos vs. competitors
       - Failure cases and limitations
       - Audio-critical scenarios where Kairos excels
```

---

## Feasibility & Practical Notes

- **Start small:** Pilot on 50 videos from Video-MME (mix of short/medium/long) to validate the RAG-to-MCQ pipeline
- **Video-MME:** ~101GB on HuggingFace. 900 videos feasible in batches.
- **MLVU:** Dev + test sets on HuggingFace. SSC and VS tasks most relevant.
- **HourVideo:** EvalAI leaderboard -- submit results officially.
- **EMScore/PAC-S:** Open-source, pip-installable, lightweight (CLIP-based).
- **Checkpoint system:** `main.py --redo` supports re-running specific stages without full reprocessing.
- **API costs:** LLM-as-judge on 100 videos x 5 dimensions = 500 GPT-4o calls (manageable).

---

## What This Gives Reviewers

1. **Quantitative scores on established leaderboards** -- no "built our own dataset" concern
2. **Reference-free quality metrics** published at CVPR/EMNLP (EMScore, PAC-S)
3. **LLM-as-judge** following the ACL 2024 Video-ChatGPT standard
4. **Component-level evidence** via ablation that the modular design is justified
5. **Head-to-head** with SOTA (Gemini, GPT-4o) on same benchmarks
6. **Reproducibility** -- all benchmarks open-access, all Kairos parameters logged
7. **Audio-visual evaluation** -- a differentiator vs. vision-only systems

---

## References

### Benchmarks

[1] Fu, C. et al. "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis." arXiv:2405.21075, 2024.
https://arxiv.org/abs/2405.21075 | https://video-mme.github.io

[2] Zhou, J. et al. "MLVU: Benchmarking Multi-task Long Video Understanding." CVPR 2025. arXiv:2406.04264.
https://arxiv.org/abs/2406.04264 | https://github.com/JUNJIE99/MLVU

[3] Chandrasegaran, K. et al. "HourVideo: 1-Hour Video-Language Understanding." NeurIPS 2024 Datasets & Benchmarks. arXiv:2411.04998.
https://arxiv.org/abs/2411.04998 | https://hourvideo.stanford.edu | https://huggingface.co/datasets/HourVideo/HourVideo

[4] Wu, H. et al. "LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding." NeurIPS 2024 Datasets & Benchmarks. arXiv:2407.15754.
https://arxiv.org/abs/2407.15754 | https://longvideobench.github.io

[5] Mangalam, K. et al. "EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding." NeurIPS 2023. arXiv:2308.09126.
https://arxiv.org/abs/2308.09126 | https://github.com/egoschema/EgoSchema

[6] Rawal, R. et al. "CinePile: A Long Video Question Answering Dataset and Benchmark." arXiv:2405.08813, 2024.
https://arxiv.org/abs/2405.08813 | https://huggingface.co/datasets/tomg-group-umd/cinepile

[7] Song, E. et al. "MovieChat: From Dense Token to Sparse Memory for Long Video Understanding." CVPR 2024. arXiv:2307.16449.
https://arxiv.org/abs/2307.16449 | https://huggingface.co/datasets/Enxin/MovieChat-1K-test

[8] Wang, H. et al. "LVBench: An Extreme Long Video Understanding Benchmark." ICCV 2025. arXiv:2406.08035.
https://arxiv.org/abs/2406.08035 | https://lvbench.github.io

### Audio-Visual Benchmarks

[9] Zhang, L. et al. "OmniEval: A Benchmark for Evaluating Omni-modal Models." arXiv:2506.20960, 2025.
https://arxiv.org/abs/2506.20960 | https://omnieval-benchmark.github.io

[10] MBZUAI-Oryx. "LongShOTBench: A Benchmark and Agentic Framework for Omni-Modal Reasoning in Long Videos." arXiv:2512.16978, 2025.
https://arxiv.org/abs/2512.16978 | https://github.com/mbzuai-oryx/longshot

[11] NJU-LINK. "OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs." arXiv:2510.10689, 2025.
https://arxiv.org/abs/2510.10689 | https://github.com/NJU-LINK/OmniVideoBench

[12] AVQA: "A Dataset for Audio-Visual Question Answering on Videos." ACM Multimedia 2022.
https://mn.cs.tsinghua.edu.cn/avqa/

[13] KD-TAO et al. "LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation." arXiv:2603.19217, 2026.
https://arxiv.org/abs/2603.19217 | https://github.com/KD-TAO/LVOmniBench

[14] R-AVST: "Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios." AAAI 2026. arXiv:2511.16901.
https://arxiv.org/abs/2511.16901

### Metrics

[15] Shi, Y. et al. "EMScore: Evaluating Video Captioning via Coarse-Grained and Fine-Grained Embedding Matching." CVPR 2022. arXiv:2111.08919.
https://arxiv.org/abs/2111.08919 | https://github.com/ShiYaya/emscore

[16] Sarto, S. et al. "Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation." CVPR 2023 (Highlight). arXiv:2303.12112.
https://arxiv.org/abs/2303.12112 | https://github.com/aimagelab/pacscore

[17] Wang, Y. et al. "InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation." arXiv:2307.06942, 2023.
https://arxiv.org/abs/2307.06942 | https://huggingface.co/OpenGVLab/ViCLIP

[18] Hessel, J. et al. "CLIPScore: A Reference-free Evaluation Metric for Image Captioning." EMNLP 2021. arXiv:2104.08718.
https://arxiv.org/abs/2104.08718 | https://github.com/jmhessel/clipscore

[19] He, X. et al. "VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation." EMNLP 2024. arXiv:2406.15252.
https://arxiv.org/abs/2406.15252 | https://github.com/TIGER-AI-Lab/VideoScore

### Evaluation Frameworks

[20] Maaz, M. et al. "Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models." ACL 2024. arXiv:2306.05424.
https://arxiv.org/abs/2306.05424 | https://github.com/mbzuai-oryx/Video-ChatGPT

[21] Liu, Y. et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." EMNLP 2023. arXiv:2303.16634.
https://arxiv.org/abs/2303.16634

[22] Li, K. et al. "MVBench: A Comprehensive Multi-modal Video Understanding Benchmark." CVPR 2024 (Highlight). arXiv:2311.17005.
https://arxiv.org/abs/2311.17005 | https://github.com/OpenGVLab/Ask-Anything

### Hallucination & Factuality

[23] Wang, Y. et al. "VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models." arXiv:2406.16338, 2024.
https://arxiv.org/abs/2406.16338 | https://github.com/patrick-tssn/VideoHallucer

[24] Zhang, J. et al. "EventHallusion: Diagnosing Event Hallucinations in Video LLMs." arXiv:2409.16597, 2024.
https://arxiv.org/abs/2409.16597 | https://github.com/Stevetich/EventHallusion

[25] Gao, H. et al. "HAVEN: Exploring Hallucination of Large Multimodal Models in Video Understanding." arXiv:2503.19622, 2025.
https://arxiv.org/abs/2503.19622

[26] Yang, D. et al. "Vript: A Video Is Worth Thousands of Words." NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.06040.
https://arxiv.org/abs/2406.06040 | https://github.com/mutonix/Vript

[27] Min, S. et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023. arXiv:2305.14251.
https://arxiv.org/abs/2305.14251 | https://github.com/shmsw25/FActScore

[28] Ha, H. et al. "NarrativeTrack: Evaluating Entity-Centric Reasoning for Narrative Understanding." Apple ML Research. arXiv:2601.01095, 2026.
https://arxiv.org/abs/2601.01095

### Datasets

[29] Krishna, R. et al. "Dense-Captioning Events in Videos." ICCV 2017. arXiv:1705.00754.
https://arxiv.org/abs/1705.00754 | https://cs.stanford.edu/people/ranjaykrishna/densevid/

[30] Zhou, L. et al. "Towards Automatic Learning of Procedures from Web Instructional Videos." AAAI 2018. arXiv:1703.09788.
https://arxiv.org/abs/1703.09788 | http://youcook2.eecs.umich.edu/

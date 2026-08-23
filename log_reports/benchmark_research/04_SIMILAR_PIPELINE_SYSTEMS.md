# Systems with Architectures Similar to Kairos

Kairos uses a "segment → describe → embed → retrieve" pipeline. This document catalogs other systems with comparable architectures to assess Kairos's novelty.

---

## Tier 1: Most Similar (Describe-Then-Retrieve for MR)

### VTG-GPT (2024)
- **Paper:** [arXiv:2403.02076](https://arxiv.org/abs/2403.02076) (Applied Sciences)
- **Pipeline:** LLM debiases query → MiniGPT-v2 captions individual frames → proposal generator matches debiased query against captions
- **Difference from Kairos:** Captions individual frames, not semantically segmented scenes. No audio/ASR. Uses proposal matching, not embedding retrieval.
- **Predecessor to:** Moment-GPT

### Moment-GPT (AAAI 2025)
- **Paper:** [arXiv:2501.07972](https://arxiv.org/abs/2501.07972)
- **Pipeline:** LLaMA-3 (query debiasing) + MiniGPT-v2 (captioning) + VideoChatGPT (span scoring)
- **Results:** R1@0.5=58.30, mAP Avg=35.00 on QVHighlights test
- **Difference from Kairos:** 3 separate MLLMs per query. No scene segmentation — generates candidate spans and scores them directly. No audio fusion.
- **Significance:** Most-cited training-free MR method. Direct competitor.

### Decoupling Semantics and Logic (ACL 2026 MAGMaR Workshop)
- **Paper:** [arXiv:2606.07924](https://arxiv.org/abs/2606.07924)
- **Pipeline:** Chunk video → generate multimodal descriptions → dense embedding retrieval → LLM reranking with full context (including OCR/ASR)
- **Results:** **#1 on MAGMaR Retrieval leaderboard**
- **Difference from Kairos:** Two-stage retrieval (coarse dense retrieval + fine LLM reranking). Strategically excludes noisy modalities (OCR/ASR) during initial retrieval but re-incorporates them during reranking.
- **Significance:** **Closest architectural match to Kairos.** Also a Video RAG pipeline. The two-stage retrieval strategy is something Kairos could adopt.

### GranAlign (AAAI 2026)
- **Paper:** [arXiv:2601.00584](https://arxiv.org/abs/2601.00584)
- **Pipeline:** Generates queries at varied semantic granularity + query-aware caption generation + multi-level matching
- **Results:** R1@0.5=59.92, mAP Avg=38.23
- **Difference from Kairos:** Generates both query-agnostic and query-aware captions. Multi-level granularity matching. No scene segmentation.

---

## Tier 2: Similar Paradigm, Different Task (QA / Understanding)

### LLoVi (EMNLP 2024)
- **Paper:** [arXiv:2312.17235](https://arxiv.org/abs/2312.17235)
- **Pipeline:** Dense short-term visual captioning (BLIP2/LaViLa/LLaVA, 0.5-8s clips) → LLM aggregates captions for temporal reasoning
- **Results:** EgoSchema 50.3% (+18.1% over prior SOTA)
- **Difference from Kairos:** Targets QA, not MR. Uses LLM reasoning over captions, not embedding retrieval. No scene segmentation.
- **Significance:** Demonstrates the "describe-then-reason" paradigm works for long video QA. Same philosophical approach as Kairos.

### LangRepo (2024)
- **Paper:** [arXiv:2403.14622](https://arxiv.org/abs/2403.14622)
- **Pipeline:** Multi-scale video chunking → text descriptions at multiple temporal scales → structured "Language Repository" with write/read operations
- **Results:** SOTA on EgoSchema, NExT-QA at its model scale
- **Difference from Kairos:** Uses iterative refinement to maintain the text repository. Multi-scale temporal representations. Targets QA.

### VidIL (2022)
- **Paper:** [arXiv:2205.10747](https://arxiv.org/abs/2205.10747)
- **Pipeline:** Frame-level visual descriptions (captions, objects, attributes, events) → temporal structure template → LLM generates output
- **Difference from Kairos:** Few-shot in-context learning, not embedding retrieval. No scene segmentation. An early ancestor of this pipeline family.

---

## Tier 3: Different Approach but Same Zero-Shot MR Goal

### REZE (2026)
- **Paper:** [arXiv:2608.04480](https://arxiv.org/abs/2608.04480)
- **Approach:** Split video into clips → VLM directly scores each clip against query → deterministic algorithms (max-subarray, Otsu thresholding) predict temporal windows
- **Results:** mAP Avg=40.32 (training-free SOTA)
- **Difference from Kairos:** No description generation. Direct VLM scoring per clip. Deterministic post-processing instead of embedding retrieval.

### VTimeCoT (ICCV 2025)
- **Approach:** Overlays visual progress bars on video frames → visuotemporal chain-of-thought prompting with VLMs
- **Results:** GPT-4o: R1@0.5=59.74
- **Difference from Kairos:** Frame-level operation with visual annotations. No scene segmentation or description generation.

### TFVTG (ECCV 2024)
- **Paper:** [arXiv:2408.16219](https://arxiv.org/abs/2408.16219)
- **Approach:** LLM decomposes query into sub-events with temporal relations → VLM scores temporal proposals → LLM integrates constraints
- **Results:** Best zero-shot on Charades-STA
- **Difference from Kairos:** Query decomposition approach. No scene-level description indexing.

---

## Tier 4: Video RAG Systems

### VideoRAG (Ren et al., Feb 2025)
- **Paper:** [arXiv:2502.01549](https://arxiv.org/abs/2502.01549)
- **Approach:** Dual-channel retrieval: graph-based textual knowledge + multimodal context. Cross-video knowledge graphs.
- **Target:** Cross-video understanding, not within-video MR.

### CARVE / V-RAGBench (2026)
- **Paper:** [arXiv:2606.13141](https://arxiv.org/abs/2606.13141)
- **Approach:** Parallel retrievers across modality-granularity configurations → chunk-adaptive reranking → interleaved evidence generation
- **Significance:** Establishes a benchmark for Video RAG systems. Operates at chunk level within videos.

### ForeSea (ECCV 2026)
- **Paper:** [arXiv:2603.22872](https://arxiv.org/abs/2603.22872)
- **Approach:** 3-stage: tracking module filters irrelevant footage → multimodal embedding indexes clips → top-K retrieval for Video LLM
- **Target:** Surveillance/forensic search. Modular pipeline similar to Kairos.

---

## Kairos's Novelty Assessment

### What IS novel:
1. **Scene-level segmentation via shot detection** as the temporal unit — most systems use fixed-length chunks or sliding windows
2. **Multimodal fusion depth** — BLIP + YOLO + Whisper + AST + LLM synthesis (no other MR system fuses this many modalities into text)
3. **Long-video scalability** — tested up to 7 hours. Most zero-shot MR methods only evaluate on <3 minute videos.
4. **Full pipeline** — MR is just one capability; the same scene descriptions power synopsis, RAG chatbot, Q&A

### What is NOT novel:
1. The "describe-then-retrieve" paradigm itself — established by VidIL (2022), LLoVi (2024), VTG-GPT (2024)
2. Text embedding retrieval for video search — common in Video RAG systems
3. Adjacent scene merging — simple heuristic used by multiple systems

### Recommended positioning:
Kairos should be positioned as a **general-purpose video understanding system with MR as one downstream capability**, not as a moment retrieval specialist. The novelty claim should focus on multimodal fusion depth + long-video scale + unified pipeline, not on the retrieve-by-description concept.

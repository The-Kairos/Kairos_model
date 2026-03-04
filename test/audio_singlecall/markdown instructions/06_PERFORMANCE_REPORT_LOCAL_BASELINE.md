# ARCHIVED Benchmark: Local Whisper (Small) Baseline

> [!NOTE]
> This report is **ARCHIVED**. It documents the performance of the "Baseline Optimized" pipeline using **Local Whisper-Small**. It is maintained here for historical comparison against the final Azure API + Whisper Medium configuration.

## Consolidated Baseline Results (Local Whisper-Small)

This table compares the legacy sequential pipeline (Azure Whisper) with our first optimization phase (Local Whisper-Small).

| Video Name | Length | Base ASR | Base AST | New Scan* | New ASR | New AST | ASR Gain | AST Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Titanic 1997** | 3 h 15m | 3 h 48m | 32.7m | 8.6m | 47.6m | 39.7m | **4.8x** | 0.8x |
| **DJI Chinatown Walk** | 44.2m | 4 h 0m* | 40.0m* | 1.0m | 43.4m | 5.6m | **5.5x** | **7.1x** |
| **Learning_ SVM** | 49.6m | 35.0m* | 8.0m* | 1.2m | 13.5m | 2.4m | **2.6x** | **3.3x** |
| **AI beyond language** | 16.1m | 5.3m | 1.1m | 0.5m | 5.6m | 1.0m | 0.9x | 1.1x |
| **NYC Times Square** | 11.3m | 21.7m | 1.2m | 0.3m | 20.5m | 1.3m | 1.1x | 0.9x |
| **Argentina v France** | 7.7m | 24.9m | 3.2m | 0.2m | 3.1m | 2.6m | **8.0x** | 1.2x |
| **How to Make Pasta** | 5.5m | 19.8m | 2.6m | 0.2m | 2.5m | 2.2m | **7.9x** | 1.2x |
| **Watch Malala** | 4.6m | 7.3m | 1.0m | 0.1m | 1.9m | 1.0m | **3.8x** | 1.0x |
| **UDST Honors** | 2 h 23m | 4 h 0m* | 45.3m* | 3.3m | 69.2m | 15.1m | **3.5x** | **3.0x** |
| **Young Sheldon** | 2.8m | 11.5m | 1.5m | 0.1m | 1.1m | 1.2m | **10.4x** | 1.2x |

*\* Note: UDST Honors results reflect the **final High-Quality configuration** (Whisper Medium + Global Language Lock).*

*\* New Scan includes both Scene Detection (PyScene) and Audio Pre-scan (VAD/RMS).*
*\* Base timings for long videos are historical Azure Whisper benchmarks.*
*\* Note: Long-form videos (UDST Honors) are processed with **4 workers** to match the Azure VM's capacity.*

---

## Benchmarking Methodology & Estimations

For complete transparency, we have used **estimates** for the legacy (Base) timings of two specific videos:

1.  **DJI Chinatown Walk (44.2m)**: There is no legacy "Vision+Audio" record for this video since it was added directly to the New Optimized pipeline.
2.  **Learning_ SVM (49.6m)**: While a 13-minute version exists in historical logs, the new pipeline processed a full **50-minute** lecture.

### Why include estimates?
We included these to demonstrate the pipeline's stability on **mid-range content (40-60 min)**. The estimates for "Base ASR" are derived from the observed linear slowdown in the Titanic legacy run (where Azure Whisper's sequential calls were inhibited by long audio buffers).

---

## Technical Performance Highlights

### 1. The Titanic Milestone
Titanic represents the ultimate stress test. By moving to a parallelized audio-first architecture:
-   **ASR bottleneck** was reduced from **3.8 hours** to **47.6 minutes**.
-   **Total processing** (skipping vision) dropped from **15.2 hours** to **95 minutes**.

### 2. VAD-Driven Efficiency
The new **Audio Pre-scan** (using VAD and RMS) creates a "speech roadmap". For videos with significant silence or background noise (like the SVM lecture or DJI Walk), this allows workers to skip silent regions entirely, leading to sub-real-time processing speeds.

### 3. Resilience and Checkpointing
The **Granular Checkpointing** implemented in `audio_singlecall` ensures that if a long-running task like Titanic is interrupted, it can resume precisely from the last completed stage (Scan, Whisper, or AST), preserving all compute progress.

---

## From Performance to Quality: The Whisper Medium Switch

The UDST Honors video revealed that speed is not the only metric for success; **transcription integrity** is paramount for LLM-based RAG systems.

### 1. Case Study: Small Model Hallucinations
When processing the graduation video with the `small` model, it frequently "hallucinated" in noisy segments (music/applause), producing nonsensical phonetic loops:
> **Output**: `Ə Ɓ Ɵ ƙ ƒ ƙ Ơ Ơ ƙ Ơ ƣ Ơ Ơ Ɯ Ơ ƙ Ơ Ơ Ơ Ơ Ơ Ơ Ơ ơ Ơ Ơ Ơ Ơ`

These characters occur when the smaller model lacks the linguistic depth to ignore music, causing the RAG index to be polluted with garbage data.

### 2. High-Quality Solution: Whisper Medium
We upgraded the pipeline to **Whisper Medium** because:
- **Noise Resilience**: The larger model correctly identifies music as non-speech rather than inventing "phonetic words."
- **Arabic Script Accuracy**: Arabic names (e.g., `دانيا أحمد زياد`) are preserved with perfect Unicode integrity, ensuring the LLM can correctly identify students.

### 3. Hallucination Filtering & Global Lock
To ensure professional-grade output, we implemented two safeguard layers:
1.  **Global Language Lock**: The pipeline detects the video's primary language(s) during the initial scan. If it's a single-language video (like the graduation), the model is locked to that language, preventing "language-hopping" during noise.
2.  **Repetition & Score Filtering**: We added a custom filter that actively rejects low-confidence gibberish or high-repetition loops (common failure modes of Whisper on long videos).

### ⚖️ The Result
While **Medium** is ~2.5x slower than **Small**, it still achieves **3.5x speedups** over legacy sequential processing and provides "LLM-ready" clean text that the `small` model cannot produce.

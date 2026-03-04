# 🧩 Codebase Features Overview
### Architecture & Capabilities of Core `.py` Files

The `audio_singlecall` pipeline consists of several specialized Python modules that work together to extract, transcribe, classify, and filter audio data at scale. Below is a comprehensive breakdown of the key features inside each active script.

---

## 🚀 1. `main.py` (The Orchestrator)
**Purpose:** Serves as the primary entry point, coordinating the extraction of video subclips, generating JSON checkpoints, and sequentially triggering the downstream audio modules (Detector, Whisper, AST).

**Key Features:**
*   **Granular Checkpointing Model**: Saves state at every major step (`checkpoint.json` -> `audio_checkpoint.json` -> `audio_results.json`). If the VM crashes or is shut down, running the script again instantly resumes from the exact failure point.
*   **CLI Argument Parsing**: Fully driven by terminal arguments (e.g., `--parallel`, `--workers`, `--use-api`, `--language`, `--cpu`), allowing rapid switching between local computation and cloud processing.
*   **Environment Integration**: Automatically loads `.env` variables ensuring secure credential management for external APIs.
*   **Scene Synchronization**: Extracts timestamp definitions from the visual pipeline (PySceneDetect) and ensures audio outputs perfectly align with those visual scene boundaries.
*   **Timing Metrics Extraction**: Tracks execution time (`wall_time_sec`) for each independent pipeline stage and logs them into `timing.json` for performance reporting.

---

## 🎙️ 2. `audio_detector.py` (Pre-Scan & VAD)
**Purpose:** Rapidly analyzes the raw audio waveform before transcription begins to classify language, map human speech, and mute voices for background sound detection.

**Key Features:**
*   **Dynamic Thresholding**: Automatically adjusts sensitivity multipliers and dBFS silence thresholds based on the total duration of the video.
*   **Silero VAD (Voice Activity Detection)**: Employs a highly accurate neural network to scan the entire audio track and return exact timestamps of human speech, allowing downstream transcription to completely skip dead air.
*   **Global Language Locking**: Samples up to 5 randomized speech regions and runs a fast Whisper `detect_language` pass.
    *   *Multilingual Unlock:* If a secondary language (like Arabic) is detected $\ge 2$ times, it flags the video as multilingual, allowing native-script transcription.
    *   *English Lock:* If only English is reliably found, it creates a hard-lock, preventing Whisper from "drifting" into hallucinated languages.
*   **Speech Masking for AST Purity**: Generates two distinct audio files: one normal (for Whisper) and one "speech-masked" file where all human voices are muted. This guarantees that background noises (like a dog barking) aren't drowned out by people talking when passed to the AST classifier.

---

## 📝 3. `whisper_singlecall.py` (Transcription & Filtering)
**Purpose:** Responsible for turning human speech into text, primarily utilizing the Azure OpenAI Cloud API, with aggressive post-processing to eliminate hallucinations.

**Key Features:**
*   **Azure OpenAI API Integration**: Securely transmits audio chunks to Azure's Whisper `verbose_json` endpoint. 
    *   *Rate-Limit Protection (Exponential Backoff)*: Catches `429 Too Many Requests` limits, automatically pauses for 65 seconds, and safely retries.
*   **Parallel Dynamic Chunking**: Instead of processing an entire video linearly, this module slices the audio into highly optimized 10-minute (600s) chunks and dispatches them across multiple CPU workers simultaneously, slashing extraction time.
*   **Advanced Hallucination Filters**:
    *   *Emoji & Symbol Cleansing:* Strips fabricated characters (e.g., `♪♫♥️✿❀`) using native regex and the explicit Python `emoji` library.
    *   *Logprob Confidence Floor:* Drops any API transcription with an `avg_logprob` below `-0.9`, heavily penalizing noise-guessing.
    *   *Repetitive Loop Collapse (*`clean_repetitive_text`*):* Grammatically structures and collapses infinite stuttering loops (e.g., `"Bye. Bye. Bye. Bye."` $\rightarrow$ `"Bye."`).
*   **Scene Re-Mapping**: Takes the single continuous transcript layout and mathematically cuts it back into the discrete timestamps required by original visual scenes.

---

## 🔊 4. `ast_parallel.py` (Natural Audio Classification)
**Purpose:** Employs the Audio Spectrogram Transformer (AST) to identify and classify environmental, mechanical, and ambient background noises within scenes.

**Key Features:**
*   **ProcessPool Parallelization**: Spins up distinct memory-isolated worker processes (based on the user's `--workers` parameter) to dramatically accelerate the heavily CPU-bound AST classification stage.
*   **Confidence Filtering**: Rejects generic or weakly confident auditory guesses, accepting only specific classes that pass tailored confidence thresholds (e.g., `traffic noise (conf=0.55)`).
*   **Hardware Acceleration Detection**: Dynamically checks for CUDA/MPS architecture, safely falling back to CPU matrices if GPUs are unassigned.

---

## ⚖️ 5. `evaluation.py` (Benchmarking)
**Purpose:** A utility script to mathematically compare the output quality and speed of the single-call pipeline against legacy JSON files.

**Key Features:**
*   **Confusion Matrix Generation**: Computes True Positives, False Negatives, F1-Scores, and Recall to statistically prove that the new optimizations did not lose any data compared to previous architectures.
*   **Timing Aggregation**: Reads both pipeline timestamps and computes exact multiplicative speedups ($\times$ faster).

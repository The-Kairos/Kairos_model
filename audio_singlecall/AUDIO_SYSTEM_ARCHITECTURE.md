# Audio System Architecture: ASR & AST Pipeline

This document details the design and implementation of the high-speed, parallelized audio processing system used in the Kairos model.

## 1. Pipeline Overview
The audio pipeline follows a three-stage execution flow designed for maximum throughput and accuracy:
1.  **VAD Pre-Scan**: Fast identification of speech regions and primary language.
2.  **ASR (Whisper)**: "Transcribe then Map" strategy for speech-to-text.
3.  **AST (Audio Sounds)**: Parallel scene-based sound classification.

---

## 2. ASR Strategy: "Transcribe then Map"
Unlike visual processing, which handles frames scene-by-scene, the ASR pipeline processes the **entire audio stream** independently of visual scenes to maintain conversational context and minimize API overhead.

### Step A: Transcription
- **Single-Call Mode**: For videos under 15 minutes, the entire audio track is sent to the Azure OpenAI Whisper API in a single request.
- **Parallel Chunking**: For long videos (e.g., movies), the audio is divided into large **10-minute blocks** with 30-second overlaps. These blocks are transcribed in parallel.
- **Global Language Lock**: The detected "Primary Language" from the pre-scan is forced into the Whisper prompt. This prevents the model from hallucinating language switches (e.g., switching to Urdu during English silence).

### Step B: Temporal Mapping
The Whisper output (a list of text segments with precise timestamps) is mapped back to the visual scenes detected by PySceneDetect:
- **Scene Intersection**: The script iterates through each visual scene duration and collects all Whisper segments that overlap with that timing.
- **Boundary Handling**: If a sentence crosses a scene boundary, it is assigned based on a 20% overlap ratio or a minimum 0.5s duration. This ensures speech isn't lost during fast visual cuts.

---

## 3. AST Strategy: Scene-Based Classification
The AST (Audio Spectrogram Transformer) pipeline focuses on identifying environmental sounds (e.g., "Siren", "Applause", "Silence") and is tightly coupled to visual scenes.

- **Scene-Level Processing**: For every visual scene, the specific audio snippet for that duration is extracted.
- **RMS Filter**: Scenes with an average loudness below -60dBFS (silence) are automatically skipped to save CPU cycles.
- **Parallel Inference**: The snippets are processed in parallel using a `ProcessPoolExecutor`. This allows the AST model to classify multiple scenes simultaneously across all available CPU cores.

---

## 4. Robustness & Filtering

### Hallucination Filtering
The pipeline includes a specialized filter to protect against Whisper "babbling" in noisy or silent videos:
- **Repetition Check**: Detects and deletes segments where the AI repeats symbols or phrases (e.g., "♪♪♪", "Thank you for watching").
- **Probabilistic Cleaning**: Segments with a low average log-probability or a high "no speech" probability are discarded.

### Unicode Support
Results are saved to `checkpoint.json` with `ensure_ascii=False`. This ensures that:
- **Foreign Scripts**: Arabic, Urdu, and Chinese are saved in their native characters.
- **Special Names**: Names like **Mbappé** are saved with correct accents rather than escape codes.

---

## 5. Directory Structure
All audio-related logic is integrated into the following core files:
- `src/audio_detector.py`: VAD Pre-scan and Language Identification.
- `src/audio_whisper_parallel.py`: Whisper API integration and "Transcribe then Map" logic.
- `src/audio_MIT_ast_parallel.py`: Parallel Audio Spectrogram Transformer classification.

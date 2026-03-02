# Audio Pipeline Modular Flow

This document explains the modular design of the optimized audio processing pipeline and how the components interact.

## Modular Components

### 1. `audio_detector.py` (The Gatekeeper)
- **Purpose**: Performs a high-speed initial scan of the entire video audio track.
- **Logic**: 
    - Uses **RMS Energy** for a fast silence check.
    - Uses **Silero VAD** for robust speech detection.
    - Uses **Spectral Flatness** to distinguish between meaningful background audio and static noise.
- **Dynamic Thresholding**: Automatically adjusts sensitivity based on video duration (e.g., more sensitive for 3-hour videos to catch brief events).
- **Outcome**: Decides if Whisper (speech) or AST (natural sounds) should be skipped entirely for the whole video.

### 2. `whisper_singlecall.py` (Speech Optimizer)
- **Purpose**: Replaces the expensive "transcribe every scene" approach with a "transcribe once" approach.
- **Workflow**:
    - Cleans the full audio once (noise reduction + VAD).
    - Calls Whisper on the entire audio track in one go.
    - **Timestamp Mapping**: Takes the Whisper segments (with timestamps) and maps them back to the specific scenes provided by PySceneDetect.
- **Efficiency**: Reduces cold-start overhead and redundant processing of scene boundaries.

### 3. `ast_processor.py` (Environmental Sound Classifier)
- **Purpose**: Detects events like "dog barking", "door slam", or "music".
- **Optimization**:
    - **Parallelization**: Uses a `ThreadPoolExecutor` to run AST inference on multiple scenes concurrently.
    - **Skip Logic**: Checks the per-scene RMS results from the `audio_detector`. If a scene is silent, AST is skipped instantly.
- **Accuracy**: Masks out speech regions before classification to ensure natural sounds aren't confused with vocals.

### 4. `main.py` (The Orchestrator)
- **Purpose**: Coordinate the entire flow.
- **Execution Order**:
    1. **Scene Detection**: Invokes `PySceneDetect` to get scene boundaries.
    2. **Pre-Scan**: Runs `audio_detector` to get the master audio state and dynamic thresholds.
    3. **ASR stage**: Runs `whisper_singlecall` (only if speech is detected).
    4. **AST stage**: Runs `ast_processor` (only if background audio is detected).
    5. **Aggregation**: Saves combined results and prints a detailed timing comparison.

### 5. `evaluation.py` (Quality Assurance)
- **Purpose**: Compares the new modular pipeline results against the legacy results.
- **Metrics**: Generates confusion matrices for speech/sound detection and calculates word-overlap similarity for transcriptions.

## Key Design Principles
- **Detection over Speed**: Dynamic thresholds ensure that even a 1-second "glass break" in a 2-hour video is captured.
- **Redundancy Reduction**: If a video is silent, the entire pipeline finishes in seconds by skipping deep learning models.
- **Compatibility**: Maintains the exact same JSON schema as the original pipeline for seamless integration.

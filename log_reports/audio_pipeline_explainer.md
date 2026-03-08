# Audio Pipeline Architecture: Explainer

The audio system operates as a **two-stage pipeline**: a high-speed pre-scan (Detector) followed by a deep analysis (Whisper/AST).

## 1. `audio_detector.py` (The Gatekeeper)
Before running expensive AI transcription, this script performs a "pre-scan" of the entire audio track.

### **Key Concepts**
*   **RMS Energy (Root Mean Square):** 
    *   **What is it?** It is a mathematical way to calculate the "average loudness" of an audio signal. Unlike a peak (which just looks at the highest spike), RMS represents the actual power/intensity of the sound.
    *   **Usage:** If the RMS is below `-60 dBFS`, the system considers the segment "dead silent" and skips it entirely to save time.
*   **Dynamic Thresholding:**
    *   **The Logic:** The longer a video is, the more likely it is to have subtle or quiet sections.
    *   **The Math:** The script applies a `sensitivity_multiplier` based on video duration (`1.0 + 0.1 * log2(minutes)`).
    *   **Result:** For a 7-hour video (Web Summit), the detector becomes **~1.3x more sensitive** than for a 5-minute clip.
*   **Silero VAD (Voice Activity Detection):**
    *   Identifies exactly where human speech occurs. This allows the system to ignore background noise or silence.
*   **Spectral Flatness:**
    *   Measures if a sound is "tonal" (like a voice or instrument) or "flat" (like white noise or wind). High flatness = noise; Low flatness = meaningful audio.

---

## 2. `audio_whisper_parallel.py` (The Transcriber)
Once the detector confirms there is speech, this script takes over.

*   **Transcription:** Uses Whisper (local or Azure API) to turn audio into text.
*   **Scene Mapping:** It doesn't just give you a block of text; it uses an **Overlap Logic** to match text to scenes.
    *   *Rule:** If a spoken sentence spans two scenes, it is assigned to whichever scene holds more than 50% of its duration.

---

## 3. Workflow Summary
1.  **Extract:** Pull raw audio (PCM 16kHz) from video.
2.  **Scan:** Use **RMS** and **VAD** to decide: "Is it worth processing?"
3.  **Transcribe:** Run Whisper on valid speech regions.
4.  **Map:** Bind the text to the correct `scene_index` in your `checkpoint.json`.

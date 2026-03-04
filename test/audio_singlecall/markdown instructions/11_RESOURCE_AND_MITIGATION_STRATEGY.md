# 🧠 Resource & Error Mitigation Strategy (188GB RAM VM)

This document outlines how the Kairos pipeline manages resources and mitigates common audio processing errors to ensure a professional-grade experience.

---

### 1. Unified Resource Management (Memory Safety)
With **188GB of RAM**, we have significant headroom, but parallel processing (BLIP + Whisper + AST) can still hit limits if not managed.

*   **Isolation Strategy**: Vision (BLIP) and Audio (Whisper/AST) tasks are processed sequentially in the root `main.py`. This ensures that the massive weights for VLM and ASR models never fight for the same memory block.
*   **Worker Balancing**: 
    *   **Whisper**: Limited to **2-4 workers** max. Each Whisper Medium model takes ~5GB. 
    *   **AST**: Can scale to **8 workers** as the model is lightweight (~500MB).
    *   **Overhead**: Even at full load, the pipeline will consume < 40GB, leaving **148GB safety margin** for OS and high-speed I/O.

---

### 2. Audio Separation (ASR vs. AST Masking)
To handle "noisy" or "overlapping" audio, we use a **Mutual Exclusion** strategy:

*   **For ASR (Speech)**: We apply aggressive noise reduction (`noisereduce` at 0.95) and Silero VAD. This "masks" background hums/music to let Whisper focus purely on the speech frequencies.
*   **For AST (Natural Sounds)**: We use the **Speech Masking** feature. We take the exact timestamps where speech was detected and **zero out** those sections in the AST buffer. 
    *   *Result*: If someone speaks over a barking dog, the AST model only hears the dog, providing a much higher confidence score for environmental sounds.

---

### 3. Handling Edge Cases & Error States
*   **Overlapping Speakers**: Whisper (Azure API) is highly robust and handles background chatter and speaker overlap by prioritizing the most dominant voice.
*   **API Resilience**: We use the **Azure OpenAI Whisper API**, providing enterprise-grade stability. 
    *   **Rate Limits**: Our parallel chunking strategy is optimized to stay within standard Azure RPM (Requests Per Minute) limits.
    *   **File Constraints**: By chunking the audio into 10-minute segments, we ensure every remote call stays well below the **25MB API limit**.
*   **Corrupt Audio**: Our PyAV-based extractor includes `fflags: +genpts` and `ignore_editlist: 1`, which sanitizes problematic MP4 files before they are sent to the API.
*   **Hallucinations**: The **Global Language Lock** and our custom **Post-API Hallucination Filter** ensure that even if the API trips over noise, the garbage text is rejected.

---

### 📊 Stability Guarantee
By using these mitigations, the pipeline is designed to never "crash" due to audio complexity. It will instead provide a "Confidence Score" for every classification, allowing the LLM to filter out low-quality data.

# 🚀 Final Azure API Run Summary
### Results of the 13-Video Benchmark (Audio Pipeline)

**Date:** March 2026
**Configuration:** `--all --parallel --api --workers 4 --cpu`
**Hardware:** Azure VM (188GB RAM, CPU-Only) + Azure OpenAI Whisper API

---

## 1. Top-Line Results

The integration of the Azure OpenAI Whisper API alongside the Parallel AST process has drastically reduced the compute time for the audio portion of the pipeline.

### The Longest Videos
| Video | Duration | Previous Local | Azure API Run | Speedup |
|:---|:---|:---|:---|:---|
| **Web Summit Qatar** | 7 h 4 min | *(Never finished)* | **52.0 mins** | — |
| **Titanic 1997** | 3 h 15 min | 4 h 20 min | **35.3 mins** | ~7x |
| **UDST Honors** | 2 h 23 min | 4 h 45 min | **22.6 mins** | ~12x |

*(Note: "Azure API Run" times represent the sum of PySceneDetect, Pre-Scan, Whisper, and AST parallel processes).*

---

## 2. Hallucination Filtering Success

During the run, the pipeline dynamically stripped out hundreds of Whisper hallucinations (music notes, emojis, looped babble, and false speech).

*   **Web Summit Qatar (7h):** Filtered out **765** hallucinations.
*   **Titanic 1997 (3h):** Filtered out **599** hallucinations.
*   **UDST Honors (2h):** Filtered out **395** hallucinations.

---

## 3. Bilingual Native Transcription

The API accurately passed back non-English dialogue without forcing translation:
*   **UDST Honors:** Accurately transcribed Arabic (RTL script) for names and dialogue during the graduation ceremony (e.g., `بشرة فاروق`).
*   **Watch Malala:** Preserved non-English phrases with native script accuracy.

---

## 4. The Zero-Cut Video Edge Case
> **Videos:** `.CCTV Dogs` and `.Statistical Learning`

You will notice these videos finished with **0.0s** for AST and 0 scenes mapped. 
*   **Why?** PySceneDetect relies on visual "cuts" (camera angle changes) to produce timestamps. Both of these videos are shot from a single, continuous, locked-off camera angle. Thus, PySceneDetect legitimately returned `0 scenes`.
*   **Is this a bug?** No, the audio pipeline did exactly what it was told: it transcribed the audio, found 0 scenes to map it to, and gracefully exited without crashing.
*   **The Fix for Production:** Before integrating into the main branch, a fallback must be added right after PySceneDetect runs in `main.py`: 
    > *If PySceneDetect returns 0 scenes, chop the video into static 10-second blocks mathematically so the downstream pipeline still has chunks to work with.*

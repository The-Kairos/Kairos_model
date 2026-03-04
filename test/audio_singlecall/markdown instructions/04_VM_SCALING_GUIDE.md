# 📚 Kairos Documentation & Scaling Guide

## 1. Modular Scaling: How the Engine Handles Workers
The Kairos pipeline is **modular by design**. It uses a "Stateless Process-Per-Worker" architecture, meaning:
- **Independence**: Each worker (`--workers N`) is a separate Python Process with its own private memory space.
- **Dynamic Workers**: You can pass `--workers 2` for a laptop or `--workers 16` for a 32-core VM. The code will scale automatically.
- **Memory Safety**: Whisper (High RAM) finishes and clears its memory before AST (Low RAM) starts, preventing crashes (OOM) on long videos.

---

## 2. RAM vs. Workers: High-Scale Formula
Total RAM needed = `[Base App (1GB)] + [N * Worker (1GB)] + [Video Buffer]`.

| Workers | CPU Strategy | Total RAM Est. | Recommended Hardware |
|---|---|---|---|
| **2 Workers** | 2 models at 100% | **4–5 GB** | 8GB Laptop |
| **4 Workers** | 4 models at 100% | **7–8 GB** | 16GB VM |
| **8 Workers** | 8 models at 100% | **11–13 GB** | 32GB VM |
| **16 Workers**| 16 models at 100%| **15–20 GB** | **Azure 188GB VM** |

**Note for Azure 188GB VM:** Since ASR (Transcription) is now offloaded to the **Azure OpenAI Whisper API**, local RAM usage is even lower. The bottleneck is purely CPU cores for AST and Vision tasks. You can safely run **20+ separate video jobs simultaneously** in Docker containers.

---

## 3. Resilience: Granular Checkpoints
Long sessions (like Titanic) are protected against VM restarts. The pipeline saves progress after **each major stage**:
1. `Scene Detection` (Saved)
2. `Whisper Transcription` (Saved)
3. `AST Sound Processing` (Saved)
If it crashes, the next run will detect `audio_checkpoint.json` and resume automatically.

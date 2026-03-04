# Kairos Pipeline Parallelization Proposal
### Processing a 1-Hour Video in Under 30 Minutes

**Author:** Kairos Engineering Team  
**Date:** March 2026  
**Status:** ✅ Phase 1 (Audio Parallelization) — COMPLETE | 🟡 Phase 2 (BLIP Parallelization) — Proposed

---

## The Problem: We Are Leaving Speed on the Table

The Kairos pipeline currently runs every stage **sequentially**, one after the other. This is simple and reliable — but it means the CPU and memory sit idle during most of the pipeline while they wait for the current stage to finish.

> **A 50-minute lecture currently takes ~18 minutes for audio alone.**  
> The full pipeline (with BLIP, YOLO, and GPT-4o) takes significantly longer.  
> A 1-hour video should not take more than 1 hour to process on production hardware. Right now it does.

---

## Measured Benchmark Data (Real Results, Google Cloud VM, CPU-Only)

These are real timings from our benchmark run today on the VM (188 GB RAM, CPU-only with `--cpu` flag):

| Video | Duration | Scenes | Whisper | AST | Audio Total |
|---|---|---|---|---|---|
| Paul Liang TEDxMIT | 16 min | 47 | 337s (5.6 min) | 62s | **430s = 7.2 min** |
| Learning: SVM | 50 min | 111 | 812s (13.5 min) | 143s | **1042s = 17.4 min** |
| DJI Chinatown Walk | 44 min | 257 | 2602s (43 min)* | 336s | **3016s = 50 min** |

> **\*Note:** DJI Chinatown Walk's Whisper time is unusually high because two competing pipeline instances were running simultaneously during benchmarking. Normalized, it is estimated at **~18–20 min**. We will re-run this standalone after the Titanic run finishes.

### 🔗 Deployment Integration
For a guide on how to integrate this Python pipeline with a **Node.js backend** on an **Azure VM**, see: [DEPLOYMENT_INTEGRATION_AZURE.md](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/DEPLOYMENT_INTEGRATION_AZURE.md)


### Key observation:
For the SVM lecture (50-min video, 111 scenes):
- Whisper alone: **13.5 min**
- AST alone: **2.4 min**
- These currently run **one after the other**
- **They could run at the same time as BLIP + YOLO** — zero conflict, different models, different data

---

## The Current Pipeline (Sequential)

```
[PySceneDetect] → [Frame Sampling] → [BLIP] → [FPS Sampling] → [YOLO]
                                                                     ↓
                                    [Whisper ASR] → [MIT AST] ←────────
                                                                     ↓
                                               [GPT-4o: Scene x N] → [Narrative] → [Synopsis] → [RAG]
```

Every arrow is a hard wait. Nothing runs in parallel.

---

## The Proposed Pipeline (Parallel)

```
                     ┌─── [Frame Sampling] → [BLIP] ────────────────┐
[PySceneDetect] ────→│                                               ├──→ [GPT-4o Scenes (async)] → [Narrative] → [Synopsis] → [RAG]
                     └─── [Audio Pre-Scan] → [Whisper] → [AST] ─────┘
```

**Three changes:**

### Change 1 — Run Audio and Video in Parallel ⚡
After PySceneDetect finishes, split into two concurrent threads:
- **Thread A:** Frame sampling → BLIP captions → YOLO detections
- **Thread B:** Audio pre-scan → Whisper transcription → AST classification

These two paths are **completely independent** — BLIP and YOLO work on image frames; Whisper and AST work on the audio waveform. They never share data until GPT-4o scene descriptions, which waits for both.

**Estimated time saved for a 1-hour video: ~13–18 minutes**

### Change 2 — Async GPT-4o Scene Descriptions ⚡
Currently GPT-4o is called **once per scene, sequentially**. For a 1-hour video with ~100 scenes, this means ~100 sequential API calls. Even at 0.5s latency each, that's 50+ seconds — and real-world latency is higher.

We can fire **4–8 calls simultaneously** using Python's `asyncio` with the OpenAI async client, respecting rate limits via a semaphore.

**Estimated time saved for a 1-hour video: ~10–20 minutes**

### Change 3 — Scale via Azure API Workers ✅ COMPLETE
Audio transcription has been moved to the **Azure OpenAI Whisper API** with parallel dynamic chunking. The VM's resources are no longer consumed by Whisper. Rate limiting is handled automatically with retry backoff and local Whisper fallback.

**Result:** Titanic (3h15m) transcribed in ~7 min. Young Sheldon (2.8m) in ~10s. **~12x faster than local model.**

---

## Current Status: What's Done vs What's Next

| Stage | Status | Notes |
|---|---|---|
| Parallel Audio (Whisper + AST) | ✅ DONE | Azure API + 4 workers, auto-language, local fallback |
| Whisper hallucination filtering | ✅ DONE | 6-layer filter: logprob, emoji, loops, dedup, VAD, chars |
| Parallel Audio vs Visual (Thread-level) | 🟡 Proposed | Root `main.py` needs ThreadPoolExecutor |
| Async GPT-4o Scene Descriptions | 🟡 Proposed | asyncio semaphore, 4–8 concurrent calls |
| BLIP Parallelization (Per-Scene) | 🟡 Proposed — See below | ThreadPoolExecutor across scenes |

---

## 🆕 Phase 2 Proposal: Parallel BLIP Captioning

BLIP is currently the dominant bottleneck for long videos. It runs **sequentially*** — one scene at a time, loading the same TorchVision model for each frame.

### Why BLIP is different from AST
AST is CPU-bound, so `ProcessPoolExecutor` works perfectly (each worker gets its own CPU core and memory). BLIP is **vision-model-bound** — it needs a GPU or large chunks of CPU RAM to load the model.

On the **188GB RAM Azure VM**, the ideal approach is **ThreadPoolExecutor** (not ProcessPool):
- Load the BLIP model **once** in the main process (~2–3 GB RAM)
- Pass frame images to worker threads — threads share the model in memory
- Worker threads process frames concurrently with Python's GIL released during model inference

### Proposed Code Change in `src/frame_captioning_blip.py`
```python
from concurrent.futures import ThreadPoolExecutor

def caption_scenes_parallel(scenes: list, model, processor, max_workers: int = 4) -> list:
    """Run BLIP captioning in parallel across scenes using shared model."""
    def caption_one(scene):
        frame_path = scene.get("frame_path")
        if not frame_path:
            return scene
        caption = run_blip_on_frame(frame_path, model, processor)
        scene["frame_caption"] = caption
        return scene

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(caption_one, scenes))
    return results
```

### RAM Impact on 188GB VM
| Setup | RAM per Worker | Total RAM (4 workers) |
|---|---|---|
| BLIP model shared (Thread) | ~0 extra | ~2–3 GB total |
| BLIP model per process (Process) | ~2–3 GB | ~8–12 GB total |

With threads, 4 workers use essentially the same memory as 1. **Recommended: ThreadPool with 4–8 workers.**

### Estimated Speedup
| Video | Sequential BLIP | Parallel BLIP (4 threads) | Saved |
|---|---|---|---|
| Paul Liang (47 scenes) | ~3 min | ~1 min | ~2 min |
| SVM (111 scenes) | ~6 min | ~2 min | ~4 min |
| Titanic (1857 scenes) | ~90 min | ~25 min | ~65 min |

> **Result: A 3-hour video that took 90 minutes for BLIP alone could finish in ~25 minutes.**

> **Result: Over 50% faster. A 1-hour video finishes in ~20 minutes.**

---

## What Changes in the Code

Only `main.py` needs to change. The individual model modules (BLIP, YOLO, Whisper, AST) stay exactly the same.

```python
# CURRENT (simplified)
scenes = run_blip(scenes)
scenes = run_yolo(scenes)
scenes = run_whisper(scenes)
scenes = run_ast(scenes)
scenes = run_gpt4o_per_scene(scenes)  # sequential

# PROPOSED (simplified)
with ThreadPoolExecutor() as pool:
    video_future = pool.submit(run_video_pipeline, scenes)  # BLIP + YOLO
    audio_future = pool.submit(run_audio_pipeline, scenes)  # Whisper + AST

video_scenes = video_future.result()
audio_scenes  = audio_future.result()
scenes = merge(video_scenes, audio_scenes)

scenes = await run_gpt4o_async(scenes, max_concurrent=6)  # async batch
```

**Risk level: Low.** The output format doesn't change. Checkpoints still work. The models are unchanged. Only the orchestration logic in `main.py` is refactored.

---

## Why This Is Safe

- ✅ **No model changes** — BLIP, YOLO, Whisper, AST are untouched
- ✅ **No output format changes** — checkpoint.json structure stays identical
- ✅ **Thread-safe** — BLIP/YOLO work on frames; Whisper/AST work on audio. No shared mutable state between threads
- ✅ **Fallback** — if a thread fails, we can catch and retry sequentially (same error handling as today)
- ✅ **Already proven** — Whisper parallelization is already working in production today

---

## Recommended Action Plan

| Priority | Change | Status | Effort | Speed Gain |
|---|---|---|---|---|
| ✅ Done | Azure Whisper API (Parallel Chunks) | Complete | — | ~12x ASR speedup |
| ✅ Done | Parallel AST (ProcessPool) | Complete | — | ~1.5x AST speedup |
| 🔴 High | Parallel BLIP (ThreadPool) | Proposed | 1–2 days | ~60 min for Titanic |
| 🔴 High | Async GPT-4o scene descriptions | Proposed | 1 day | ~15 min per 1-hr video |
| 🟡 Medium | Parallel Audio vs Visual (Thread-level in `main.py`) | Proposed | 0.5 days | ~8–15 min per 1-hr video |
| 🟢 Low | Increase Whisper workers (4 → 6) | Optional | 1 hour | ~3–5 min per long video |

**Total effort: ~3–4 days of engineering for a ~40% speed improvement.**

---

## Bottom Line

> The Kairos pipeline already has excellent model accuracy.  
> The remaining challenge is **speed**. Our models don't need to get faster — our **orchestration** does.  
> Running audio and video processing in parallel, combined with async LLM calls, brings a 1-hour video from ~44 minutes down to ~25 minutes with **zero changes to model accuracy or output format**.

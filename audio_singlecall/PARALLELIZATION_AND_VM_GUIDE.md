# Parallelization & VM Usage Guide

## TL;DR
> **The Google VM is used for its RAM, not its GPU.**  
> All models run on CPU — the VM just gives us 16 GB of clean, unshared memory to handle large videos without crashing.

---

## 1. How the Pipeline Runs (CPU-Only, Parallelized)

The audio pipeline has 4 stages per video:

| Stage | Model | How It Runs |
|---|---|---|
| Scene Detection | PySceneDetect | Single-threaded, fast |
| Audio Pre-Scan | Silero VAD | Single-threaded |
| Speech Transcription | Whisper (small) | **Multi-process**: audio split into 600s chunks, 2 workers |
| Sound Classification | MIT AST | **Multi-process**: per-scene, 2 workers |

### The `--parallel` Flag
When you pass `--parallel`, the pipeline:
1. **Splits the audio** into 600-second chunks with a 30-second overlap
2. **Spawns 2 child processes**, each loading Whisper independently and transcribing one chunk at a time
3. **Merges and deduplicates** the segments by timestamp after all chunks finish

This means a 50-minute video (`Learning: SVM`) gets split into 5 × 10-minute chunks processed 2-at-a-time — roughly **2.5× faster** than a single sequential call.

---

## 2. Why We Use the VM (It's About RAM, Not GPU)

```
--cpu flag → CUDA_VISIBLE_DEVICES="" → GPU is invisible to PyTorch
```

All computation happens on CPU cores. The VM is used because:

### RAM Requirements Per Video Category

| Video Type | RAM Needed | Run Locally? |
|---|---|---|
| Short (< 5 min) | ~2 GB | ✅ Yes, any machine |
| Medium (5–20 min) | ~3–5 GB | ✅ Yes, if ≥ 8 GB free |
| Long (20–60 min) | ~6–10 GB | ⚠️ Only if ≥ 10 GB free |
| Extra long (60+ min, e.g. Titanic 3h) | ~12–16 GB | ❌ VM required |

### RAM Per Component (2 Workers)
| Component | Per Worker |
|---|---|
| Whisper small model | ~500 MB |
| MIT AST model | ~300 MB |
| Audio buffer | ~170–400 MB |
| Python overhead | ~200 MB |
| **Total for 2 workers** | **~2.4–2.8 GB** |

For a 50-minute video, the **full audio buffer alone** takes ~750 MB, pushing total usage to ~4–5 GB during transcription.

---

## 3. Can I Run This Locally?

**Yes**, as long as you have enough free RAM. Rule:

```
Free RAM ≥ 5 GB  →  safe for most videos (up to ~60 min)
Free RAM ≥ 12 GB →  safe for Titanic-length (3+ hours)
```

Check your free RAM before running:
```bash
# Linux/Mac
free -h

# The "available" column is what matters, not "total"
```

### Video-by-Video Breakdown

| Video | Duration | RAM Needed | VM Required? |
|---|---|---|---|
| `Young Sheldon` | ~22 min | ~4 GB | No (≥ 8GB free) |
| `How to Make Pasta` | ~10 min | ~3 GB | No |
| `Argentina vs France` | ~15 min | ~3 GB | No |
| `Malala Nobel Speech` | ~20 min | ~3.5 GB | No |
| `AI Beyond Language (TEDxMIT)` | ~16 min | ~3 GB | No |
| `NY Times Square Walk` | ~11 min | ~3 GB | No |
| `DJI Osmo Chinatown Walk` | ~44 min | ~5 GB | ⚠️ 10 GB+ free |
| `Learning: SVM` | ~50 min | ~6 GB | ⚠️ 12 GB+ free |
| `UDST Honors Graduation` | varies | ~4–6 GB | ⚠️ depends |
| `Web Summit Qatar 2026` | varies | ~4–6 GB | ⚠️ depends |
| `Statistical Learning: K-Fold` | varies | ~4–6 GB | ⚠️ depends |
| `.CCTV Dogs` | varies | ~3 GB | No |
| `Titanic (1997)` | ~3 hours | ~14 GB | ✅ VM required |

---

## 4. The Command

**Always run from the project root** (`~/Kairos_model`):

```bash
cd ~/Kairos_model
python3 -m audio_singlecall.main --all --parallel --workers 2 --cpu --debug
```

> ⚠️ Do NOT run from inside `audio_singlecall/` — Python won't find the package.

### Flag Reference
| Flag | Purpose |
|---|---|
| `--all` | Process every `.mp4`/`.mkv` in `Videos/` (including hidden `.`-prefixed ones) |
| `--parallel` | Enable multi-process chunked transcription |
| `--workers 2` | Use 2 parallel workers (safe for 16 GB RAM) |
| `--cpu` | Force CPU-only, ignore GPU (for consistent benchmarks) |
| `--debug` | Print live progress to terminal |
| `--video "path"` | Process a single specific video instead |

---

## 5. Output Location

Results are saved in:
```
audio_singlecall/results/<video-name>/
    audio_results.json   # per-scene speech + sound labels
    timing.json          # benchmark timings for each stage
```

Already-processed videos are **automatically skipped** on re-runs.

# 🛠️ Manual Migration Plan: Audio Pipeline → Main Branch
### From `audio_singlecall/` (Testing) to `src/` + `main.py` (Production)

---

## Understanding the Two Worlds

### Current Testing Structure (`audio_singlecall/`)
This folder is our **isolated test harness**. It runs everything via a shell script with flags:
```bash
./run_pipeline.sh --all --parallel --api --workers 4 --cpu
```
It has its own `main.py`, its own `results/` folder, and uses `--flags` because we are testing and iterating quickly. **This is how we run it right now.**

### Target Production Structure (root `main.py` + `src/`)
When we migrate to the main branch, the app is invoked as:
```bash
python main.py --video /path/to/video.mp4
```
This root `main.py` orchestrates the **complete Kairos pipeline**: BLIP → Scene Detection → YOLO → ASR → AST → LLM Scene Description → RAG → Query. No shell script. No flags needed for deployment — everything is configured via code.

---

## Why We Use Flags in Testing But Not in Production

The testing shell script (`run_pipeline.sh`) uses CLI flags (`--api`, `--parallel`, `--workers`) to let us switch options quickly during development. When we migrate to the main branch:
- `--api` becomes `use_api=True` hardcoded in the `run_pipeline()` function call
- `--parallel` becomes `parallel=True` hardcoded
- `--workers 4` becomes `max_workers=4` hardcoded or read from a config file
- No human needs to type these — the backend calls `main.py` programmatically via Node.js

---

## Migration Plan (Do NOT Execute Yet — Plan Only)

### Step 1: Copy Core Modules into `src/`

| Source File (`audio_singlecall/`) | Destination (`src/`) | Purpose |
|:---|:---|:---|
| `audio_detector.py` | `src/audio_detector.py` | VAD pre-scan, language detection, speech masking |
| `whisper_singlecall.py` | `src/whisper_parallel.py` | Azure API chunking, fallback, hallucination filtering |
| `ast_processor.py` | `src/ast_parallel.py` | Parallel AST background sound classification |

> **Note:** `main.py` (audio_singlecall) and `evaluation.py` are **testing utilities** — they do NOT move to `src/`. They stay in `audio_singlecall/` as the test harness.

---

### Step 2: Replace `src/audio_speech.py`

The existing `src/audio_speech.py` uses the old sequential Whisper pipeline. After migration:
- **Delete** the old implementation from `src/audio_speech.py`
- **Import and wrap** the new parallel `extract_speech_singlecall` function from `src/whisper_parallel.py`

---

### Step 3: Replace `src/audio_natural.py`

The existing `src/audio_natural.py` uses the old sequential AST. After migration:
- **Delete** old implementation
- **Import and wrap** `extract_sounds_optimized` from `src/ast_parallel.py`

---

### Step 4: Update `src/log_utils.py`

Add logged wrappers for the three new functions so performance is tracked in the existing logging infrastructure:

```python
from src.audio_detector import scan_audio
from src.whisper_parallel import extract_speech_singlecall
from src.ast_parallel import extract_sounds_optimized

@log_step()
def scan_audio_log(*args, **kwargs):
    return scan_audio(*args, **kwargs)

@log_step()
def extract_speech_parallel_log(*args, **kwargs):
    return extract_speech_singlecall(*args, **kwargs)

@log_step()
def extract_sounds_parallel_log(*args, **kwargs):
    return extract_sounds_optimized(*args, **kwargs)
```

---

### Step 5: Update Root `main.py`

Replace the old sequential audio logic with a single parallel audio block. The three stages (Pre-Scan → Whisper → AST) run in sequence, but internally each is parallelized:

```python
# === OPTIMIZED PARALLEL AUDIO PIPELINE ===
if not audio_already_done(checkpoint):
    # 1. VAD Pre-Scan
    scan_result = scan_audio_log(video_path=video_path, scenes=scenes, debug=True)
    
    # 2. Azure Whisper API (Parallel Chunks, Auto Language, Local Fallback)
    checkpoint["scenes"] = extract_speech_parallel_log(
        scenes=checkpoint["scenes"],
        scan_result=scan_result,
        use_api=True,           # Azure OpenAI Whisper API
        language=None,          # Auto-detect per chunk (native script preserved)
        parallel=True,
        max_workers=4,
    )
    
    # 3. Parallel AST Sound Classification  
    checkpoint["scenes"] = extract_sounds_parallel_log(
        scenes=checkpoint["scenes"],
        scan_result=scan_result,
        max_workers=4,
        use_processes=True,
    )
    save_checkpoint(checkpoint)
```

> **No argparse changes needed.** The root `main.py` is called by Node.js with a video path, not by a human with flags.

---

### Step 6: Environment Variables

Ensure the production `.env` file on the Azure VM contains:
```
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=whisper-karios
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

The `load_dotenv()` call at the top of `src/whisper_parallel.py` handles loading these automatically. No code changes needed.

---

## Full Production Pipeline Flow (After Migration)

```
Node.js Backend
    └── Receives video upload
    └── Calls: python main.py --video /path/to/video.mp4
        ├── Scene Detection (PySceneDetect)
        ├── Frame Sampling (src/frame_sampling.py)
        ├── BLIP Captioning (src/frame_captioning_blip.py)
        ├── YOLO Object Detection (src/frame_obj_d_yolo.py)
        ├── Audio Pre-Scan (src/audio_detector.py)  ← NEW
        ├── Whisper ASR (src/whisper_parallel.py)   ← NEW
        ├── AST Sounds (src/ast_parallel.py)        ← NEW
        ├── LLM Scene Description (src/scene_description.py)
        ├── Synopsis (src/synopsis_systhesis.py)
        └── Store in Vector DB for RAG (src/rag_convo.py)
```

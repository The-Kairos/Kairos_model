# 🛠️ Manual Migration: Parallel Audio to Main Branch

Follow these exact steps to move the optimized parallel audio logic into your `model` branch's `src/` folder and update your root `main.py`.

---

### Step 1: Move Core Modules to `src/`
Copy these files from `audio_singlecall/` into your root `src/` directory and rename them as shown:

1.  **audio_detector.py** → `src/audio_detector.py`
2.  **whisper_singlecall.py** → `src/whisper_parallel.py`
3.  **ast_processor.py** → `src/ast_parallel.py`

---

### Step 2: Update `src/log_utils.py`
Add these logged wrappers to the end of `src/log_utils.py` so the new functions are tracked in your performance logs:

```python
# --- Add these imports at the top of src/log_utils.py ---
from src.audio_detector import scan_audio
from src.whisper_parallel import extract_speech_singlecall
from src.ast_parallel import extract_sounds_optimized

# --- Add these wrappers at the end of src/log_utils.py ---
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

### Step 3: Update Root `main.py`
We will replace the old sequential audio logic with the new parallel block.

#### 1. Update Imports
At the top of your root `main.py`, update the imports to include your new logged functions:
```python
from src.log_utils import (
    # ... existing imports ...,
    scan_audio_log,
    extract_speech_parallel_log,
    extract_sounds_parallel_log
)
```

#### 2. Replace Audio Logic (Lines ~267-292)
Delete the old `extract_sounds_log` and `extract_speech_log` blocks and replace them with this single optimized block:

```python
    # === OPTIMIZED AUDIO PIPELINE (Parallel) ===
    if "audio_natural" not in checkpoint["scenes"][-1].keys() or "audio_speech" not in checkpoint["scenes"][-1].keys():
        print_section("Running Optimized Audio Pipeline (Parallel)...")
        
        # 1. VAD Pre-Scan (Detects speech vs silence)
        scan_result, step['audio_prescan'] = scan_audio_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            target_sr=ast_target_sr,
            debug=True
        )
        
        # 2. Parallel Whisper (Now with Azure API integration)
        checkpoint["scenes"], step['asr_timings'] = extract_speech_parallel_log(
            scenes=checkpoint["scenes"],
            scan_result=scan_result,
            model_size="medium", # (Ignored if use_api=True)
            use_vad=asr_use_vad,
            language=None, # Set to 'ar' or 'en' for forced override, or None for Auto-Global Detection
            parallel=True, 
            use_api=True,  # Set to True to use Azure OpenAI Whisper API
            debug=True
        )
        
        # 3. Parallel AST (Sound classification)
        checkpoint["scenes"], step['ast_timings'] = extract_sounds_parallel_log(
            scenes=checkpoint["scenes"],
            scan_result=scan_result,
            target_sr=ast_target_sr,
            max_workers=4, # Recommendation for 188GB RAM VM
            use_processes=True,
            debug=True
        )
        
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
```

---

### Step 4: Update Argument Parser in root `main.py`
To control the new features from your terminal, add these arguments to the `argparse` section of your root `main.py`:

```python
    parser.add_argument("--parallel", action="store_true", help="Enable parallel audio processing")
    parser.add_argument("--use-api", action="store_true", default=True, help="Use Azure OpenAI Whisper API")
    parser.add_argument("--language", type=str, default=None, help="Force transcription language (e.g. 'en', 'ar')")
```

Then, ensure these are passed into your `run_pipeline` call within the `main` loop.

---

### ✅ Benefits Recap
- **Enterprise Stability**: Azure Whisper API provides higher accuracy and better handling of background noise.
- **Zero Local Constraints**: Offloading transcription to Azure frees up your VM's RAM for even faster BLIP/Vision processing.
- **Deduplication & Filters**: Even with the API, our custom filtering logic is maintained to ensure the highest quality text for your LLM.

# Kairos Pipeline: Setup & Rerun Guide

To ensure everyone is on the same version of the dependencies (especially the fix for NumPy/Numba) and to properly use the new GPT synthesis rules, follow these steps.

## 1. Clean Setup (One-time)
If your virtual environment is old or acting buggy, recreate it for a fresh start:

```powershell
# Remove old environment
Remove-Item -Recurse -Force venv

# Create new environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Install dependencies (Includes the NumPy fixes)
pip install -r requirements.txt
```

## 2. Environment Configuration (.env)
Make sure your `.env` file matches these critical settings:
```env
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_ENDPOINT=YOUR_ENDPOINT
AZURE_OPENAI_DEPLOYMENT_NAME=YOUR_MODEL_NAME
AZURE_OPENAI_API_VERSION=2024-02-15-preview # Must match this version for best results
```

## 3. Rerun Commands

### Full Pipeline Rerun (Fresh Start)
Use this if you want to start from absolute zero:
```powershell
venv\Scripts\python.exe main.py process --video "YourVideo.mp4" --redo scenes frame_captions yolo audio_speech audio_natural llm narrative synopsis rag
```

### Resume/Synthesize Only (Fastest)
Use this if the video has already been processed (i.e. if YOLO and ASR are done) and you just want to regenerate the **Story Narrative** and **Synopsis** with the new prompts:
```powershell
venv\Scripts\python.exe main.py process --video "YourVideo.mp4" --redo llm narrative synopsis rag
```

### Just the Final Synopsis (Instant)
Use this if you already have the `narrative_*.txt` file and just want to fix the JSON parsing/Timestamps:
```powershell
venv\Scripts\python.exe main.py process --video "YourVideo.mp4" --redo synopsis
```

---
**Recent Fixes Included:**
- **NumPy Compatibility**: Forced `numpy<2` to avoid Numba crashes.
- **Content Filter Bypass**: API calls now safely skip/fallback if Azure blocks content.
- **JSON Repair**: If GPT-4o returns invalid JSON, the script now automatically repairs it.
- **Accurate Timestamps**: Prompts have been refined to strictly enforce narrative timestamp extraction.

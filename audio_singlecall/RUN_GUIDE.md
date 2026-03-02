# How to Run the Optimized Audio Pipeline

This guide explains how to use the new modular audio processing pipeline from **within** the `audio_singlecall` folder.

## Prerequisites
1. Open your terminal (PowerShell or CMD).
2. Navigate to the `audio_singlecall` directory:
   ```cmd
   cd "c:\Users\tehre\OneDrive\Desktop\COMP4201-3 Capstone Project\Kairos Model\Kairos\audio_singlecall"
   ```

## 1. Automated Execution (Recommended)

### Step 1: Activate the environment
Choose the command for your terminal:
- **PowerShell**: `..\venv\Scripts\Activate.ps1`
- **CMD**: `..\venv\Scripts\activate.bat`

### Step 2: Run the script
Once you see `(venv)` in your prompt, run:

**Using PowerShell:**
```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

**Using CMD:**
```cmd
run_all.bat
```

---

## 2. Manual Execution (Standard Python Commands)

If you prefer to run the modules manually, you **must** set the `PYTHONPATH` to the parent directory so Python recognizes the package structure.

### PowerShell Manual:
```powershell
$env:PYTHONPATH = ".."
..\venv\Scripts\python.exe -m audio_singlecall.main --all
..\venv\Scripts\python.exe -m audio_singlecall.evaluation
```

### CMD Manual:
```cmd
set PYTHONPATH=..
..\venv\Scripts\python.exe -m audio_singlecall.main --all
..\venv\Scripts\python.exe -m audio_singlecall.evaluation
```

---

## 3. Frequently Asked Questions

### Do I need to activate the `venv`?
No. The scripts and commands above use the absolute path (`..\venv\Scripts\python.exe`), which automatically uses the virtual environment's packages without needing to run `activate`. 

However, if you **want** to activate it manually:
- **PowerShell**: `..\venv\Scripts\Activate.ps1`
- **CMD**: `..\venv\Scripts\activate.bat`

### Where are the outputs?
All results are saved inside `audio_singlecall/results/<video_name>/`.
- `audio_results.json`: Main data (speech + natural sounds).
- `timing.json`: Speed metrics.
- `evaluation.json`: Accuracy comparison.

### Why did I get a "ModuleNotFoundError"?
This happens if you run the script from inside the folder without setting `PYTHONPATH=..`. The automated scripts above (`run_all.ps1` and `run_all.bat`) handle this for you automatically.

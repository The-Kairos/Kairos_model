# Optimized Pipeline Commands

Based on your system resources (**16GB RAM**, **12 logical cores**) and the **Whisper Small** model, use these commands for maximum efficiency without crashing.

## 1. Run Remaining Videos (Skip Complete)
This command will process all remaining videos in the `Videos/` folder. It uses **2 workers** to stay safely within your available 2.3GB RAM.

```powershell
.\venv\Scripts\python -m audio_singlecall.main --all --parallel --workers 2
```

> [!TIP]
> **Why 2 workers?** Each worker consumes ~1.2GB. With 2.3GB available RAM on your laptop, 2 workers is the safest "high-speed" choice. 4 workers would likely trigger OOM.

## 2. Evaluate All Results
Once the run is complete, use this to generate the final comparison metrics for the entire catalog:

```powershell
.\venv\Scripts\python -m audio_singlecall.evaluation --all
```

## 3. Individual Video Run
If you want to run a specific file (e.g., Titanic) individually:

```powershell
.\venv\Scripts\python -m audio_singlecall.main --video "Videos/.Titanic.1997.mkv" --parallel --workers 2
```

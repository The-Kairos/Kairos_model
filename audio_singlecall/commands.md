# Audio Pipeline Execution Commands

Use these commands to run and evaluate the audio pipeline.

## 1. Run Pipeline on All Videos
This command runs the audio-only pipeline (Scene Detection -> Audio Pre-Scan -> Whisper -> AST) on every video in the `Videos/` folder. It will **automatically skip** any videos that have already been processed.

```powershell
.\venv\Scripts\python -m audio_singlecall.main --all --parallel --workers 4
```

### Parameters:
- `--all`: Process everything in the `Videos/` directory.
- `--parallel`: Uses multiple CPU cores for Whisper and AST (high speed).
- `--workers 4`: Number of concurrent processes (adjust based on your RAM/CPU).

## 2. Evaluate All Results
Once processing is finished, run this to compare the new results against the original pipeline results:

```powershell
.\venv\Scripts\python -m audio_singlecall.evaluation --all
```

## 3. Run/Evaluate a Single Video
If you want to focus on just one file:

```powershell
# Run
.\venv\Scripts\python -m audio_singlecall.main --video "Videos\Titanic.mkv" --parallel

# Evaluate
.\venv\Scripts\python -m audio_singlecall.evaluation --video "Videos\Titanic.mkv"
```

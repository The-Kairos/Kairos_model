@echo off
echo Starting High-Parallel Audio Pipeline (CPU Optimized)...

REM Set PYTHONPATH to parent directory so we can run as a module from inside this folder
set PYTHONPATH=..

REM Run the pipeline with --parallel and --all
REM Using ..\venv to reach the virtual environment in the root directory
..\venv\Scripts\python.exe -m audio_singlecall.main --all --parallel --workers 4

echo.
echo Generating Evaluation Metrics...
..\venv\Scripts\python.exe -m audio_singlecall.evaluation --all

echo.
echo Done! Check results in: audio_singlecall/results/
echo Summary report: audio_singlecall/BENCHMARK_COMPARISON.md
pause

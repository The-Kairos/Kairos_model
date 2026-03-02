@echo off
REM Kairos Optimized Audio Pipeline - Process All Videos
REM This script is designed to be run from WITHIN the 'audio_singlecall' folder.

echo.
echo Starting Optimized Audio Pipeline...

REM Set PYTHONPATH to parent directory
set PYTHONPATH=..

echo Processing all videos...
"..\venv\Scripts\python.exe" -m audio_singlecall.main --all

echo.
echo Generating Evaluation Metrics...
"..\venv\Scripts\python.exe" -m audio_singlecall.evaluation

echo.
echo Done! Check results in: audio_singlecall/results/
pause

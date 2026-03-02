# Kairos Optimized Audio Pipeline - Process All Videos
# This script is designed to be run from WITHIN the 'audio_singlecall' folder.
# Usage: powershell -ExecutionPolicy Bypass -File .\run_all.ps1

Write-Host "`nStarting Optimized Audio Pipeline..." -ForegroundColor Cyan

# Set PYTHONPATH to parent directory so 'audio_singlecall' is recognized as a package
$env:PYTHONPATH = ".."

# 1. Run the Pipeline on all videos
Write-Host "Processing all videos..." -ForegroundColor Gray
& "..\venv\Scripts\python.exe" -m audio_singlecall.main --all

# 2. Run Evaluation
Write-Host "`nGenerating Evaluation Metrics..." -ForegroundColor Gray
& "..\venv\Scripts\python.exe" -m audio_singlecall.evaluation

Write-Host "`nDone! Check results in: audio_singlecall/results/" -ForegroundColor Green

# Resuming Pipeline After VM Shutdown

If your VM shuts down or the process is interrupted, you don't have to restart from scratch. The **Granular Checkpointing** system handles this automatically.

### 1. Fix Permissions (One-time)
If you see "Permission denied", run this first:
```bash
chmod +x ~/Kairos_model/audio_singlecall/run_pipeline.sh
```

### 2. Resume Commands

#### Option A: Resume All Remaining Videos
```bash
cd ~/Kairos_model/audio_singlecall
./run_pipeline.sh --all --parallel --workers 4 --cpu --debug
```

#### Option B: Resume a Specific Video (e.g., UDST Honors)
```bash
cd ~/Kairos_model/audio_singlecall
./run_pipeline.sh --video ".UDST honors graduation" --parallel --workers 4 --cpu --debug
```

---

### How Checkpointing Works
When you re-run the command:
1.  The pipeline checks for `audio_checkpoint.json` in the video's results folder.
2.  It will see which stages are marked as **"COMPLETE"**.
3.  **Example for UDST**: If it finished the Scan and Whisper before the shutdown, it will say `[1/4] Scene Detection: SKIPPED` and `[2/4] Whisper Transcription: SKIPPED` and go straight to AST.

> [!IMPORTANT]
> Always run the script from the `~/Kairos_model/audio_singlecall` directory to ensure the Python module paths are resolved correctly.

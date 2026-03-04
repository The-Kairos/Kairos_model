# Integrated Pipeline Guide (main_test.py)

We have created `main_test.py` which combines the **Full Kairos Logic** (BLIP, YOLO, LLM, RAG) with the **New Parallel Audio Pipeline**.

### 🚀 How to Run the Full Pipeline
Always run from the project root:
```bash
cd ~/Kairos_model
python3 main_test.py process --video "Videos/.UDST honors graduation.mp4" --parallel --api --workers 4 --cpu
```

### 🛠️ What `main_test.py` Does:
1.  **Scene Detection**: Reuses parameters (`threshold=27`).
2.  **Vision (BLIP/YOLO)**: Runs as before, saving results to `_processed/`.
3.  **Parallel Audio**: Instead of sequential AST/ASR, it triggers the `audio_singlecall` logic.
    -   It merges the parallel results back into the main scene list.
    -   It respects existing checkpoints in `_processed/` OR `audio_singlecall/results/`.
4.  **LLM/RAG**: Continues the process using the high-quality parallel audio results.

---

### 🐢 Why is UDST Honors taking "three million years"?
The UDST Honors video is **142 minutes (2.4 hours)** long. 

1.  **CPU Processing**: Whisper transcription on CPU is much slower than GPU.
2.  **Workers**: You are currently using **2 workers**.
    -   The script split the audio into **15 chunks**.
    -   With 2 workers, it processes only 2 chunks at a time.
    -   **Recommendation**: Since your VM has massive RAM (188GB), you can safely increase workers to **4 or 6** to speed this up significantly.

**To speed it up**:
```bash
python3 main_test.py process --video ".UDST honors graduation" --parallel --workers 4 --cpu
```

> [!TIP]
> **Checking Progress**: You can check the `audio_singlecall/results/.UDST honors graduation/audio_checkpoint.json` to see exactly which chunks are done.

# 🔄 Fresh Run Protocol: High-Quality Benchmarking

Since we upgraded from the `small` to the `medium` Whisper model and added **Global Language Detection**, the processing times and transcription accuracy have changed. We need to reset the data and run everything from scratch to get accurate final benchmarks.

---

### Step 1: Clean the Slate
To ensure a true "from scratch" run without using old checkpoints or cached results, delete the existing results folder:

```bash
cd ~/Kairos_model/audio_singlecall
rm -rf results/*
```
*Note: This removes all old JSONs and checkpoints for all 13 videos.*

---

### Step 2: Run the High-Quality Pipeline
Run the optimized pipeline for all videos. We will use **4 workers** and the **medium** model (now the default).

```bash
./run_pipeline.sh --all --parallel --api --workers 4 --cpu
```

---

### Step 3: Monitor Multilingual Output
Since the **UDST Honors** video is currently being processed, you can check its progress to see the Arabic names being stored in native script.

**To verify while running**:
```bash
# In another terminal
tail -f results/.UDST\ honors\ graduation/audio_results.json
```

---

### Step 4: Final Benchmark Update
Once the run is complete, we will update the **Performance Report**. 
- The "New ASR" timings will be slightly higher than the `small` model results, but still significantly faster than the legacy sequential pipeline.
- We will document these as the "Final High-Quality Benchmarks."

---

### ✅ Success Criteria
1.  **Arabic Integrity**: Names like `دانيا أحمد زياد` are stored as shown, not as phonetic approximations.
2.  **No Hallucinations**: No repetitive "I feel you will be bad" loops in noisy sections.
3.  **Speed**: Benchmarks remain within the optimized targets (e.g., Titanic in < 60 mins).

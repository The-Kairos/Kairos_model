# Kairos Smart Sync & Idempotency Logic

This document explains how the Kairos pipeline avoids redundant processing and ensures data consistency across re-runs.

---

## 1. Deterministic ID Generation
The system ensures that a single video file always maps to the same **Chat ID** in MongoDB, even if you run it from different environments or at different times.

- **Logic**: The `StorageManager` (`src/storage_utils.py`) takes the base filename (e.g., `Young_Sheldon.mp4`), cleans it, and generates a 24-character hex hash.
- **Why?**: This prevents "Chat Duplication." If we used random IDs, running the same video twice would create two different records in your database, splitting your conversation history.
- **Location**: `src/storage_utils.py` inside the `__init__` method.

---

## 2. Local Checkpointing (`_processed/`)
To save on GPU costs and time, the pipeline is "Idempotent"—it checks for existing results before starting any AI task.

- **Logic**: Each time a stage (YOLO, BLIP, Whisper) completes, it saves its state into a `checkpoint.json` file inside the video's folder in `_processed/`.
- **How it skips**: When you run the process again, `main.py` loads this JSON. If it sees that a stage is already marked as finished, it skips that code entirely and moves to the next step.
- **Location**: `main.py` within the main loop, specifically where it handles `redo_steps`.

---

## 3. The "Smart Sync" Mechanism
Even when processing is skipped locally, the system **always** synchronizes with MongoDB.

- **Sync Flow**:
    1. Pipeline starts and detects existing local data.
    2. Processing is skipped (0ms GPU time used).
    3. The `StorageManager` is initialized with the deterministic ID.
    4. A final "Sync" call is made to ensure the **Synopsis**, **Chunks**, and **Embeddings** in MongoDB match the local files.
- **Benefit**: This ensures your Web App is always up-to-date even if the video was processed weeks ago on a different machine.

---

## 4. How to Force a Re-run
If you change your prompts, models, or settings and **want** to repeat the processing, you can bypass this logic using the `--redo` flag:

```bash
# Redo everything from scratch
python3 main.py process "Videos/MyVideo.mp4" --redo scenes

# Redo only the final synopsis and RAG embeddings
python3 main.py process "Videos/MyVideo.mp4" --redo synopsis
```

> [!TIP]
> **Deleting Cache**: You can also force a re-run by simply deleting the video's folder in `_processed/`.

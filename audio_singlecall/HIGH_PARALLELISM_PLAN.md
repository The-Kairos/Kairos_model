# 🚀 High Parallelism Deployment Plan (CPU Optimized)

This plan outlines the implementation of **Dynamic Audio Chunking** and **Multi-Process Parallelism** to achieve sub-realtime performance on extremely long videos (Titanic, Web Summit, etc.) using only CPU resources.

## 1. Technical Strategy: Bypassing the GIL
Python's Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecodes at once. Since Whisper and AST are compute-intensive, standard threading is insufficient.
- **Solution**: Use `concurrent.futures.ProcessPoolExecutor`.
- **Mechanism**: Each CPU core runs a completely separate Python process with its own memory space, allowing 100% utilization of multi-core processors.

## 2. Dynamic Audio Chunking (Whisper)
For videos exceeding **15 minutes**, the pipeline will automatically pivot to chunked processing:

### A. Logic
- **Chunk Size**: 600 seconds (10 minutes).
- **Threshold (15 min)**: Why 15 minutes? 
    - **Overhead**: Loading the Whisper model (500MB+) into a new process takes ~15-20s. 
    - **Efficiency**: For a 5-minute video, transcription takes ~2 mins; parallelizing it into two 2.5m chunks would save only 1 min but add 20s overhead (net gain < 45s).
    - **Long Videos**: At 15+ minutes, transcription takes 5+ minutes. Parallelizing into 10-minute chunks provides a massive CPU speedup that far outweighs the model load time.
    - **Memory**: For a 3-hour video (Titanic), a single audio buffer can exceed 2GB. 10-minute chunks take only ~40MB each, preventing OOM.
- **Overlap**: 30 seconds (to ensure word continuity).
- **Parallelism**: $N$ chunks processed simultaneously across $N$ CPU cores.

### B. Merge & Deduplication Strategy
1. **Offsetting**: Add `chunk_start_time` to all segment timestamps returned by Whisper.
2. **Midpoint Filtering**: For any segment falling into an overlap zone $[T_{overlap\_start}, T_{overlap\_end}]$, only keep it if its `(start + end) / 2` falls within the primary half of the overlap. This prevents duplicate text in the final output.

## 3. Multi-Process AST
- **Batching**: Instead of processing 1 scene per thread, we will distribute scenes across multiple processes.
- **Resource Guard**: Model loading consumes ~450MB per process. The system will auto-detect available RAM and limit the number of worker processes to prevent swapping/crashing.

## 4. Deployment Roadmap
1. **Main Orchestrator**: Add `--parallel` and `--chunks` flags to `main.py`.
2. **Chunker Utility**: Implement `audio_utils.py` to handle PCM buffer slicing without re-encoding (zero quality loss).
3. **Merging Engine**: Implement the deduplication logic in `whisper_singlecall.py`.
4. **Benchmarking**: Run a "Titanic" (3h) test to verify the speedup factor.

## 5. Technology Stack
- **Core**: Python `multiprocessing` (Standard Library).
- **Audio Logic**: `NumPy` + `PyAV` (In-memory slicing).
- **Inference**: `openai-whisper` (current) -> `faster-whisper` (future upgrade).

---
**Goal**: Reduce the processing time of a 2-hour video from 1.5 hours to **~15-20 minutes** on a standard 8-to-16-core CPU.

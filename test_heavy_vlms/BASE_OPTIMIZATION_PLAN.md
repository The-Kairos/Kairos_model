# Base Processing Optimization & Deployment Plan

To scale the benchmarking pipeline for videos of 1-2 hours, we must move away from the "One-at-a-time" sequential model to a **Concurrent & Batched Architecture**.

## 📊 Current vs. Projected Performance (75 Scene Video)

| Module | Current (Sequential) | Optimized (Parallel/Batched) | Gain |
| :--- | :--- | :--- | :--- |
| **Scene Detection** | ~1.5s | ~1.5s | — |
| **ASR (Whisper)** | ~688s | ~60s (Chunked API) | **11x** |
| **AST (Sounds)** | ~90s | ~15s (Batched GPU) | **6x** |
| **YOLO (Tracking)** | ~9s | ~5s (Batched Inference) | **1.8x** |
| **Total Pipeline** | **~788s** | **~85s** | **9.2x Speedup** |

---

## 🛠️ Step 1: Technical Optimization Plan

### A. ASR: Dynamic Parallel Chunking
Instead of a fixed number, we use **Dynamic Chunking** based on the total video duration.
- **Strategy**: Aim for chunks of **~15 minutes** each. 
  - *Formula*: `num_chunks = max(1, math.ceil(total_duration_minutes / 15))`
  - *Example*: 10m video = 1 chunk; 60m video = 4 chunks; 120m video = 8 chunks.
- **Rate Limit Management**: 
  - Azure Whisper default is **~3 RPM (Requests Per Minute)**.
  - **The Plan**: Our code will use a `Semaphore` or `concurrency_limit` in the `ThreadPoolExecutor` to never send more than 3 requests simultaneously. This guarantees zero `429` errors.
- **Temporal Alignment**: Segment-level timestamps in `verbose_json` are matched back to the original `base_data.json` scene timestamps.

### B. AST (Sounds): Batched GPU Inference
**Current**: We loop through 75 scenes, and for each scene, we load the audio slice, send it to the GPU, wait for the label, and repeat.
**Optimized (The "Batch")**:
1. Pre-extract all audio slices for the whole video into a single NumPy array/tensor.
2. Send the **entire stack** (Batch) to the GPU at once.
- **Why it's faster**: GPUs are designed to do thousands of calculations in parallel. Processing 32 audio slices takes almost the same amount of time as processing 1. This is where the **6x speedup** comes from.

### C. YOLO (Objects): High-Throughput Tracking
**Current**: YOLO processes frames in a stream.
**Optimized**:
- Increase the `batch_size` parameter in the YOLO model. instead of `model.predict(frame)`, we use `model.predict(list_of_32_frames)`.
- **Benefit**: Saturates the GPU memory. Instead of the GPU waiting for the CPU to "feed" it frames, the GPU stays 100% active, reducing the time-per-frame significantly.

### D. Pipeline Concurrency (DAG Execution)
Currently, stages run 1 → 2 → 3.
**New Flow**:
1. Run **Scene Detection** (CPU).
2. **Launch Simultaneously**:
   - `Process A`: Extract Audio -> ASR Parallel API Calls (I/O).
   - `Process B`: Extract Frames -> YOLO Batched Tracking (GPU).
   - `Process C`: Extract Full Audio -> AST Batched Inference (GPU).

### 3. Data Persistence (Azure Cosmos DB)
Since the pipeline generates structured JSON, Cosmos DB (NoSQL) is the ideal destination.

- **Storage Strategy**:
  - **Video Document**: A single document per video containing metadata (name, duration, path).
  - **Scene Documents**: Each scene (with its VLM, ASR, AST, and YOLO data) is stored as an item in a `scenes` container.
- **Worker Workflow**:
  - Once the Python worker finishes the fusion (`[3/3]`), it performs a **Batch Upsert** to Cosmos DB.
  - Using the `azure-cosmos` Python SDK, the worker pushes the final result directly to the cloud.
- **Indexing**: 
  - Partition key: `video_id`.
  - Enables the Node.js frontend to fetch all scenes for a 1-hour video in a single, ultra-fast query: `SELECT * FROM c WHERE c.video_id = 'my_video_123'`.

---

## 🏗️ Step 3: Backend Architecture & Open Source Stack

The proposed stack is **100% Free and Open Source**, making it easy to integrate into your Node.js website.

| Component | Technology | License | Role |
| :--- | :--- | :--- | :--- |
| **Task Queue** | **BullMQ** | MIT | Manages the background jobs from Node.js. |
| **Data Store** | **Redis** | BSD | Stores the queue and temporary state. |
| **Worker Engine**| **BullMQ-Python** | MIT | Allows Python to consume jobs from the Node.js queue. |
| **API Wrapper** | **FastAPI** | MIT | (Optional) If you prefer REST over Queues. |

### How it runs in your Web App:
1. **Producer (Node.js)**: Your website receives a video. It adds a "job" to BullMQ: `jobQueue.add('vlm-process', { videoPath: '...' })`.
2. **Broker (Redis)**: Redis holds the job until a worker is free.
3. **Consumer (Python)**: The optimized Python script (using the BullMQ-Python worker) sees the job, runs the parallelized pipeline (YOLO + ASR + VLM), and writes the final JSON.
4. **Listener (Node.js)**: Node.js watches for the "completed" event and notifies the user via WebSockets or an API response.

---

## ⚠️ System Limitations & Risks

1. **VRAM Overlap**: If running ASR (Cloud) and YOLO (local GPU) at once, VRAM is fine. But if running AST + YOLO concurrently, you need at least **8GB VRAM** to stay safe with 720p/1080p frames.
2. **API Rate Limits**: Azure OpenAI has TPM (Tokens Per Minute) limits. 10 parallel chunks might trigger a `429 Too Many Requests` error.
3. **FFmpeg Overhead**: Extracting hundreds of frames is Disk I/O intensive. A high-speed NVMe SSD is recommended for production.

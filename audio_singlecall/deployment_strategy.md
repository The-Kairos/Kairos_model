# Backend Deployment Strategy: Audio Pipeline

This guide explains how the High-Parallelism Audio Pipeline should be integrated into a production backend (e.g., FastAPI, Node.js, or Go) to handle large-scale video processing.

## 1. Asynchronous Task Queue (The Core)
In a production backend, you **never** run the pipeline directly in the API request handler. Instead, use a task queue like **Celery (Python)**, **Redis Queue (RQ)**, or **Bull (Node.js)**.

- **Workflow**: 
    1. User uploads video.
    2. API saves video to Cloud Storage (S3/GCS).
    3. API triggers a "background task" with the video ID.
    4. Worker nodes pick up the task and run `audio_singlecall.main`.

## 2. Resource Management (Preventing OOM)
Deployment environments often have limited RAM per container. 
- **Memory Check**: Use the `psutil` logic we implemented to skip or downscale parallel workers if the server is under heavy load.
- **Worker Isolation**: Instead of full Kubernetes (K8s), you can use **Azure Container Apps (ACA)** or **Azure Container Instances (ACI)**.
    - **Azure Container Apps (ACA)**: Best for this project. It scales to zero when no videos are being processed (saving money) and automatically triggers containers based on Azure Queue Storage or Service Bus messages.
    - **Azure Container Instances (ACI)**: Great for long-running, isolated "Titanic" jobs. You can spin up a specific ACI for one movie, and Azure deletes it once the process finishes.
- **Concurrency**: Limit the number of concurrent videos processed per machine. For Titanic-sized videos, process only 1 at a time per node to avoid OOM.

## 3. Storage Optimization (Cleanup)
- **Temporary Files**: Use a `/tmp` directory or a mounted volume for `.clips`, `.frames`, and `.fps`.
- **Auto-Cleanup**: Implement a `finally` block in the worker code to delete these folders immediately after the JSON results and narratives are uploaded to the persistent database/bucket.
- **Statelessness**: The worker should be stateless. Input comes from Cloud Storage; output goes to the DB/Storage; local disk is wiped.

## 4. Scalability (Parallelism)
- **Node-Level**: This pipeline uses `ProcessPoolExecutor` to utilize all cores on a single high-compute VM.
- **Horizontal Scaling**: Use **Azure Container Apps** with KEDA (Kubernetes-based Event Driven Autoscaler). It sounds like K8s, but it's "serverless"—you don't manage the cluster. It will spin up 1 worker for every 1 video in the queue.

## 6. Integration with Azure & Node.js
- **Node.js Frontend**: Your Node.js API should use the **Azure Storage SDK** to generate a "SAS Token" (temporary access URL) for the video and pass it to the Python worker.
- **Azure DevOps**: Your CI/CD pipeline should build the Docker image and push it to **Azure Container Registry (ACR)**. ACA will then pull the latest image automatically.
- **Communication**: Use **Azure Service Bus** or **Azure Storage Queues**. Node.js pushes the message, and the Python worker reads it.

## 7. Resource Calculation: Your Setup
Based on our analysis, your laptop has **16GB Total RAM** with approximately **2.3GB Available** during runs.

| Component | RAM Estimate (MB) |
| --- | --- |
| Whisper (small) | ~500 MB |
| AST Model | ~300 MB |
| Audio Buffers (per worker) | ~170 MB |
| Overhead (Tensors, Python) | ~200 MB |
| **Total Per Worker** | **~1.2 GB** |

### Safe Worker Recommendation:
- **On Laptop (Available 2.3GB)**: Use **`--workers 2`**. Even with 16GB total, your background apps (Chrome, OS) are consuming ~13GB. 
- **On Azure (Standard D4s_v5 - 16GB)**: Use **`--workers 4`**. Since the VM will be "clean" (no background apps), you'll have ~14GB available, making 4 workers very safe.

### Key Rule for Titanic (3-hours):
For massive videos, always stick to **`--workers 2`** or fewer to avoid the multiplication of the 170MB+ audio buffers across processes during the AST phase.

## 8. Transition to VM (Google Cloud)
To run benchmarks on your Google VM while keeping results compatible with production:

### SSH & Setup
1. **Connect**: `ssh <user>@<vm-ip>`
2. **Environment**: Use the provided `Dockerfile` to ensure the exact same libraries.
   ```bash
   docker build -t kairos-audio .
   docker run -v $(pwd)/Videos:/app/Videos kairos-audio
   ```
3. **CPU Benchmarking**: To simulate a CPU-only server on a GPU machine, use the `--cpu` flag:
   ```bash
   python -m audio_singlecall.main --all --parallel --workers 4 --cpu
   ```

## 9. Transition to Production (Azure)
### Azure Blob Storage Integration
In deployment, you won't use a local `Videos/` folder.
- **Node.js**: Receives video -> Uploads to Azure Blob.
- **Python Worker**: 
    1. Downloads video from Blob to `/tmp`.
    2. Runs the pipeline.
    3. Uploads `audio_results.json` back to Azure Blob.
- **Scaling**: Use **Azure Container Apps** to spin up one container per message in the queue.

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

## 5. Handling Long Videos (The 3-Hour Case)
- **Timeouts**: Ensure the task runner has a high enough timeout (e.g., 2 hours for AST/Whisper on a 3-hour movie).
- **Heartbeats**: The worker should send "progress updates" to the backend (e.g., "AST 40% complete") so the UI doesn't time out.
- **Resume Capability**: Our `checkpoint.json` logic is crucial here. If a worker crashes or gets preempted (Spot instances), the new worker can resume from the last successful step instead of restarting from minute 0.

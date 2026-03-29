# Kairos VM Migration: Project History (Start to Finish)

This document serves as the official log for the migration of the Kairos video processing pipeline from a local environment to a GPU-accelerated Virtual Machine (VM) with MongoDB integration.

---

## Phase 1: Infrastructure & API
**Goal**: Move processing from local scripts to a server-based API.
- **Flask Server**: Implemented `server/app.py` to handle remote video processing requests.
- **Job Orchestration**: Integrated a `ThreadPoolExecutor` to queue tasks, preventing GPU out-of-memory (OOM) errors by serializing video runs.
- **Real-Time Streaming**: Implemented **Server-Sent Events (SSE)** to push progress updates (e.g., "YOLO: 45%") directly to the web application.

---

## Phase 2: MongoDB Atlas Integration
**Goal**: Synchronize all results to a persistent, globally accessible database.
- **StorageManager**: Developed `src/storage_utils.py` as a unified interface for local and remote storage.
- **Data Collections**:
    - `chats`: Stores pipeline state (`processing`, `ready`), synopsis, and summary metadata.
    - `chat_chunks`: Stores scene-by-scene metadata and **OpenAI Vector Embeddings** for semantic search.
- **Auto-Sync Engine**: Enabled the pipeline to automatically detect `.env` credentials and perform "dual-write" (saving to both the local `_processed/` folder and MongoDB simultaneously).

---

## Phase 3: CLI & UX Streamlining
**Goal**: Make the pipeline "human-friendly" for developers and the production server.
- **Simplified Commands**: Updated `main.py` so a video can be processed with a single argument: `python3 main.py process <video_path>`.
- **Deterministic IDs**: Implemented 24-character native hex IDs generated from the video name. This ensures that re-running a video updates the **same** record instead of cluttering the database with duplicates.
- **"Raw" Run Support**: Enabled the system to automatically "upsert" missing chat documents during manual command-line tests.

---

## Phase 4: Final Refinements & Fixes
**Goal**: Ensure seamless production integration.
- **Environment Parity**: Fixed an issue where the Flask server was not loading the `.env` file, ensuring its background tasks always have the correct database connection.
- **Log Readability**: Updated the server to preserve the original filename of uploaded videos (e.g., `Young_Sheldon.mp4`) instead of renaming everything to `input.mp4`.
- **Production Roadmap**: Identified and documented the final Nginx requirements (turning off `proxy_buffering`) for live deployment.

---

## Status: SUCCESS
The Kairos VM is now a fully integrated hub for AI processing. It correctly handles file uploads, performs deep ML analysis, and live-syncs all results to your MongoDB Atlas cluster.

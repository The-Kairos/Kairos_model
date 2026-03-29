# Kairos VM & MongoDB Architecture Overview

This document explains the integrated flow between the Kairos video processing pipeline, the Flask API server, and the MongoDB Atlas backend.

---

## 1. System Architecture

The Kairos system is designed for high-performance AI processing on a GPU VM, with real-time feedback to a central database.

### The Stack:
- **Application**: Flask (Server) + Python (Orchestrator)
- **Database**: MongoDB Atlas (Source of Truth)
- **AI Tools**: YOLO (Object), BLIP (Caption), Whisper (Speech), AST (Sounds), GPT-4o (Summary)

---

## 2. Smart Storage & "Auto-Sync"

The `StorageManager` (`src/storage_utils.py`) has been upgraded to be "Self-Configuring."

- **Auto-Detection**: The pipeline automatically looks for your `MONGODB_URI` in your `.env`. If found, it enables "Remote Mode" without any extra CLI flags.
- **Deterministic Native IDs**: If no `chat-id` is provided (e.g., during a raw command-line run), the system generates a **valid 24-character hex ID** based on the video name.
  - *Benefit*: Re-running the same video updates the **same** record in MongoDB instead of creating duplicates.
- **Auto-Upsert**: The system intelligently creates the parent "Chat" document on the fly if it doesn't already exist.
- **Connection Stability**: The `StorageManager` now handles `mongodb+srv://` URIs by ensuring `dnspython` is present and validates connections with a `ping` command upon initialization.
- **Multi-Video Support**: Initialization logic has been refactored to correctly isolate `chat_id` and `is_remote` state for each video when processing multiple files in a single batch.

---

## 3. Real-Time Progressive Updates

The pipeline now features **"Catch-up Progress Reporting"**. 

- **State Persistence**: If you restart a partially completed video, the system detects existing checkpoints and immediately syncs the current "furthest" stage to MongoDB. 
  - *Example*: If BLIP captioning was already done, the progress bar in your Next.js frontend will instantly jump to 40% instead of staying at 0% until the next new step finishes.
- **Granular Stages**: Stages are mapped from internal step names to user-friendly titles (e.g., `get_scene_list` -> `scene_detection`).

## 3. Streamlined CLI (Human-Friendly)

We have simplified the commands to make testing as easy as possible.

**Run the ALL-IN-ONE process:**
```bash
# No flags required! It auto-detects DB and creates IDs.
python3 main.py process "Videos/Young Sheldon - First Day of High School.mp4"
```

**Force a Full Re-run:**
```bash
# Ignores cached results and re-processes everything
python3 main.py process "Videos/Young Sheldon.mp4" --redo scenes
```

---

## 4. Production API & SSE

The **Flask Server** (`server/app.py`) handles requests from your Next.js frontend.

### Real-Time Streaming (SSE)
We use **Server-Sent Events (SSE)** via the `/jobs/<runId>/stream` endpoint.

## 6. Future Steps: Production Deployment

While the AI and Database core is 100% complete, the final web server (Nginx) is not yet installed on this VM. Follow these steps for production deployment:

### 1. Install Nginx
```bash
sudo apt update && sudo apt install nginx
```

### 2. Configure for Real-Time Streaming
Add this block to your Nginx site configuration (e.g., `/etc/nginx/sites-available/default`) to support **Server-Sent Events (SSE)**:

```nginx
location /jobs/ {
    proxy_pass http://localhost:8000;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
    proxy_buffering off; # Essential for the progress bar!
}
```

### 3. Run with Gunicorn (Background)
Instead of running with `python3 server/app.py`, run with Gunicorn to handle multiple concurrent statuses:
```bash
gunicorn -w 2 --threads 4 -b 0.0.0.0:8000 server.app:app
```

---

## 5. Security Note
Following your request, **API Key authentication has been removed** for simpler internal use. Security should now be managed at the firewall/IP level within your VM environment.

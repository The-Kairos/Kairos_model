# Kairos VM & MongoDB Integration: Testing Guide

This guide explains how to verify the MongoDB storage integration and the VM Flask API.

---

## 1. Environment Setup

Before running tests, ensure your local environment variables are set. You can do this by creating a `.env` file in the root directory or by exporting them in your terminal.

```bash
# Required for API testing
export KAIROS_VM_API_KEY="your-secret-key"
# Required for MongoDB testing (use a mock or real URI)
export MONGODB_URI="mongodb://localhost:27017"
# Ensure the root directory is in the Python path
export PYTHONPATH=$PYTHONPATH:.
```

---

## 2. Test Folders

Tests are grouped by scope:

- `test/unit`: fast tests with mocks and no external services
- `test/smoke`: small real-connection checks for providers or assets
- `test/integration`: multi-component or API workflow checks

## 3. Running Individual Tests

### A. Unit Tests
Verifies local logic without calling external services.

```bash
python test/unit/test_storage.py
python test/unit/test_pyscenedetect.py
```

### B. Flask API Server Test
Start the server in one terminal:
```bash
python3 server/app.py
```

Then, in a second terminal, check the health:
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/health
```

---

### C. Smoke Tests
These touch real provider credentials or real local assets.

```bash
python test/smoke/gpt4o_connection.py
python test/smoke/gemini_connection.py
```

### D. Integration Test Client
This exercises the VM API workflow end-to-end against a running server.

```bash
python test/integration/kairos_client.py --video Videos/your_video.mp4
```

## 4. Video Processing Workflows

### Phase 1: Local Only (Default)
Run a video process without MongoDB synchronization.
```bash
python3 main.py process --video "path/to/video.mp4"
```

### Phase 2: Remote MongoDB Sync
Run a video process and sync results to a specific MongoDB Chat ID.
```bash
python3 main.py process \
  --video "path/to/video.mp4" \
  --chat-id "65f4d1a2b3c4d5e6f7a8b9c0" \
  --mongo-uri "mongodb+srv://..."
```

---

## 5. End-to-End API Integration
To simulate how the Next.js app will trigger the VM:

```bash
curl -X POST http://localhost:8000/process \
  -H "X-API-Key: your-secret-key" \
  -F "video=@test/test_video.mp4" \
  -F "videoId=test_vid_1" \
  -F "chatId=65f4d1a2b3c4d5e6f7a8b9c0"
```

### Watching Progress (SSE)
To see real-time progress updates (Server-Sent Events):
```bash
curl -N -H "X-API-Key: your-secret-key" http://localhost:8000/jobs/<RUN_ID_FROM_PREVIOUS_STEP>/stream
```

---

## 6. Troubleshooting

- **ModuleNotFoundError: No module named 'src'**: Run commands from the project root. The unit tests now add the repo root to `sys.path` automatically.
- **Connection Refused**: Ensure the Flask server is running on the correct port (default 8000).
- **Unauthorized (401)**: Double-check that your `X-API-Key` matches the `KAIROS_VM_API_KEY` on the server.

# Kairos VM API & Pipeline — Complete Reference

## 1. API Endpoints

### GET /health (or /api/health)
Health check.
```json
// Response 200
{ "status": "ok", "active_jobs": 0, "gpu_available": true }
```

### POST /process
Start video processing. Accepts JSON or multipart.
```json
// Request (JSON)
{
  "video_url": "string (required)",
  "videoId": "string (required)",
  "chatId": "string (required)",
  "videoFilename": "string (optional, default 'input.mp4')",
  "jobId": "string (optional, used as runId if provided)"
}

// Response 202
{ "runId": "uuid-string" }

// Response 400
{ "error": "Missing required fields" }
```

### GET /jobs/{run_id}
Get job status.
```json
// Response 200
{
  "status": "pending|running|completed|failed",
  "stage": "string",
  "percent": 0-100,
  "runId": "string",
  "chatId": "string",
  "videoPath": "string",
  "error": "string (only if failed)"
}

// Response 404
{ "error": "Job not found" }
```

### GET /jobs/{run_id}/stream
SSE stream of job progress. See Section 4 for event format.

### POST /query
RAG query on processed video.
```json
// Request
{
  "videoId": "string (required)",
  "chatId": "string (required)",
  "query": "string (required)",
  "topK": 5  // optional, integer, min 1
}

// Response 200
{
  "answer": "string",
  "clips": [
    {
      "startTimeSec": 0.0,
      "endTimeSec": 10.5,
      "startTimecode": "00:00:00.000",
      "endTimecode": "00:00:10.500",
      "context": "scene description text",
      "score": 0.8742
    }
  ]
}

// Response 400
{ "error": "Missing videoId, chatId, or query" }
// Response 404
{ "error": "No embedded chat_chunks found..." }
// Response 500
{ "error": "exception message" }
```

---

## 2. MongoDB Collections

**Database name:** `MONGODB_DB_NAME` env var, default `"kairos"`
**Connection:** `MONGODB_URI` env var (MongoDB Atlas SRV string)

### chats collection

Updated during processing via `update_pipeline_state()`, finalized by `save_final_results()`.

```json
{
  "_id": "ObjectId (from chatId)",

  "pipeline": {
    "state": "processing | ready",
    "percent": 0-100,
    "lastStage": "stage name string",
    "updatedAt": "ISO UTC string",
    "lastError": "string | null"
  },

  "summary": {
    "title": "Video Summary: {video_stem}",
    "videoName": "video_stem",
    "synopsis": {
      "summary": "string",
      "video_highlights": [{ "start": "HH:MM:SS", "end": "HH:MM:SS", "highlight": "string" }],
      "video_timeline": [{ "timestamp": "HH:MM:SS", "event": "string" }],
      "suggested_clips": [{ "timestamp": "HH:MM:SS.mmm", "start": "HH:MM:SS.mmm", "end": "HH:MM:SS.mmm", "description": "string" }]
    },
    "generatedAt": "ISO UTC string"
  },

  "messages": [],
  "messageCount": 0,
  "deletedAt": null,
  "createdAt": "ISO UTC string",
  "lastMessageAt": null,
  "updatedAt": "ISO UTC string"
}
```

### chat_chunks collection

Inserted by `save_final_results()` after pipeline completes. Old chunks deleted first via `delete_many({"chatId": oid})`.

**Scene chunk:**
```json
{
  "_id": "ObjectId (auto)",
  "chatId": "ObjectId",
  "chunkIndex": 0,
  "sceneIndex": 0,
  "type": "scene",
  "startSec": 0.0,
  "endSec": 10.5,
  "startTimecode": "00:00:00.000",
  "endTimecode": "00:00:10.500",
  "context": "scene description text",
  "captions": ["frame caption 1", "frame caption 2", "frame caption 3"],
  "objects": ["person", "car", "dog"],
  "audioSpeech": "transcribed speech text",
  "audioNatural": "Music (conf=0.85), Speech (conf=0.72)",
  "embedding": [0.012, -0.034, ...],
  "createdAt": "ISO UTC string"
}
```

**Meta chunk** (one per synopsis section):
```json
{
  "_id": "ObjectId (auto)",
  "chatId": "ObjectId",
  "chunkIndex": 23,
  "sceneIndex": null,
  "type": "summary | highlights | timeline | suggested_clips | questions",
  "context": "payload text",
  "embedding": [0.012, -0.034, ...],
  "createdAt": "ISO UTC string",
  "<type>": "payload text"
}
```
The `<type>` field is a dynamic key matching the `type` value (e.g., `"summary": "text"` when type is `"summary"`).

---

## 3. Pipeline Stages (in order)

| Step Name | SSE Stage ID | Percent | What It Does |
|-----------|-------------|---------|-------------|
| get_scene_list | scene_detection | 7% (SSE) / 10% (MongoDB) | PySceneDetect — detect scene boundaries |
| save_clips | clip_extraction | 15% / 20% | Extract clip files per scene (debug only) |
| sample_frames | frame_sampling | 18% / 30% | Sample 3 frames per scene, resize to 320px |
| caption_frames | frame_captioning | 20% / 40% | BLIP captions on sampled frames |
| sample_fps | motion_sampling | 30% / 45% | Sample frames at fixed FPS for YOLO |
| detect_object_yolo | object_detection | 33% / 50% | YOLOv8 object detection + ByteTrack |
| audio_scan | audio_prescan | 40% / 60% | Silero VAD — detect speech/music presence |
| asr_timings | speech_transcription | 50% / 65% | Whisper speech-to-text |
| ast_timings | sound_analysis | 60% / 75% | MIT AST audio classification |
| describe_scenes | scene_description | 73% / 85% | GPT-4o/Gemini scene descriptions |
| summarize_scenes | narrative_synthesis | 82% / 90% | GPT-4o narrative summaries |
| synthesize_synopsis | synopsis_generation | 91% / 95% | GPT-4o synopsis, highlights, clips, Q&A |
| make_embedding | embedding | 100% / 100% | Gemini embed contexts + KMeans clustering |

**Note:** SSE percent (from `STAGE_PROGRESS` in app.py) and MongoDB percent (from `STEP_STAGE_MAP` in main.py) use different mappings. The SSE values are what the frontend sees.

**Parallel execution:** BLIP, YOLO, and Audio branches run in parallel via ThreadPoolExecutor. LLM stages run sequentially after.

---

## 4. SSE Event Format

Each event is sent as `data: {JSON}\n\n`.

**Progress event:**
```json
{
  "status": "running",
  "stage": "scene_detection",
  "percent": 7,
  "runId": "uuid",
  "chatId": "string",
  "videoPath": "/path/to/video.mp4"
}
```

**Completion event:**
```json
{
  "status": "completed",
  "stage": "embedding",
  "percent": 100,
  "runId": "uuid",
  "chatId": "string",
  "videoPath": "/path/to/video.mp4"
}
```
No synopsis data is included in SSE — the frontend reads it from MongoDB after completion.

**Failure event:**
```json
{
  "status": "failed",
  "stage": "narrative_synthesis",
  "percent": 82,
  "runId": "uuid",
  "chatId": "string",
  "videoPath": "/path/to/video.mp4",
  "error": "exception message"
}
```

**All possible stage values:**
`precheck`, `downloading`, `scene_detection`, `clip_extraction`, `frame_sampling`, `frame_captioning`, `motion_sampling`, `object_detection`, `audio_prescan`, `speech_transcription`, `sound_analysis`, `scene_description`, `narrative_synthesis`, `synopsis_generation`, `embedding`

**All possible status values:**
`pending`, `running`, `completed`, `failed`

---

## 5. Synopsis Output Schema

```json
{
  "chat_name": "string (3-5 words)",
  "summary": "string (single paragraph)",
  "video_highlights": [
    {
      "start": "HH:MM:SS or 'Not explicitly stated'",
      "end": "HH:MM:SS or 'Not explicitly stated'",
      "highlight": "string (one sentence)"
    }
  ],
  "video_timeline": [
    {
      "timestamp": "HH:MM:SS or 'Not explicitly stated'",
      "event": "string (3-5 words)"
    }
  ],
  "suggested_clips": [
    {
      "timestamp": "HH:MM:SS.mmm",
      "start": "HH:MM:SS.mmm (same as timestamp)",
      "end": "HH:MM:SS.mmm (where clip content ends)",
      "description": "string"
    }
  ],
  "questions": [
    {
      "question": "string",
      "answer": "string"
    }
  ]
}
```

- `video_highlights`: 4-6 items
- `video_timeline`: 4-6 items
- `suggested_clips`: 4-6 items
- `questions`: 22 required + ~15 generated = ~37 total

---

## 6. RAG Query Flow

When `/query` is called:
1. `load_chat_chunks_from_mongo(chat_id)` — loads all chunks from `chat_chunks` collection for that chatId
2. `rank_chat_chunks(question, chunks, top_k)`:
   - Embeds the question using **Gemini `gemini-embedding-001`** (768 dimensions)
   - Computes **cosine similarity** between question embedding and each chunk's stored embedding
   - Separates results into `scene_matches` (type=scene) and `meta_matches` (other types)
   - Returns top_k scene matches + top meta matches
3. `create_answer(question, matches)` — sends question + top context to **Gemini `gemini-2.5-pro`** for answer generation
4. Returns `{ "answer": "...", "clips": [...] }`

**Clip payload shape** (from `_clip_payload()`):
```json
{
  "startTimeSec": float,
  "endTimeSec": float,
  "startTimecode": "HH:MM:SS.mmm",
  "endTimecode": "HH:MM:SS.mmm",
  "context": "scene description",
  "score": 0.8742
}
```

---

## 7. Dependencies

### ML Models
| Model | ID | Source | File |
|-------|----|--------|------|
| BLIP | `Salesforce/blip-image-captioning-base` | HuggingFace | frame_captioning_blip.py |
| YOLOv8 | `yolov8s` (small) | Ultralytics | frame_obj_d_yolo.py |
| Whisper | `medium` (default) | OpenAI | audio_whisper_parallel.py |
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` | HuggingFace | audio_MIT_ast_parallel.py |
| Silero VAD | `snakers4/silero-vad` | PyTorch Hub | audio_detector.py |
| Gemini Embedding | `gemini-embedding-001` | Google | rag_convo.py |
| Gemini Generation | `gemini-2.5-pro` | Google | rag_convo.py |
| GPT-4o | `gpt-4o-kairos` (Azure) | Azure OpenAI | synopsis_synthesis.py |

### Key Python Packages
torch 2.9.1, transformers 4.57.1, ultralytics 8.3.230, openai-whisper, opencv-python, scenedetect, flask, flask-cors, pymongo, google-genai 1.50.1, openai, pillow, librosa, av, numpy, scikit-learn

---

## 8. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MONGODB_URI` | MongoDB connection string | None (required) |
| `MONGODB_DB_NAME` | Database name | `"kairos"` |
| `GEMINI_API_KEY` | Google Gemini API key | None (required for embeddings/RAG) |
| `GPT_ENDPOINT` | Azure OpenAI endpoint | None (required for synopsis) |
| `GPT_DEPLOYMENT` | Azure OpenAI deployment name | None |
| `GPT_KEY` | Azure OpenAI API key | None |
| `GPT_VERSION` | Azure OpenAI API version | None |
| `AZURE_OPENAI_KEY` | Azure Whisper API key | None |
| `AZURE_OPENAI_ENDPOINT` | Azure Whisper endpoint | None |
| `AZURE_OPENAI_DEPLOYMENT` | Azure Whisper deployment | None |
| `HF_TOKEN` | HuggingFace token | None |
| `PORT` | Flask server port | `8000` |
| `KAIROS_EXECUTION_MODE` | `"parallel"` or `"semi_parallel"` | `"parallel"` |
| `KAIROS_JOB_ROOT` | Temp job directory root | None (uses CWD) |
| `KAIROS_MODEL_CWD` | Working directory | `os.getcwd()` |
| `KAIROS_LOW_MEM` | Low memory mode | `"AUTO"` |
| `MAX_KAIROS_WORKERS` | Max concurrent pipeline workers | `2` |
| `HF_HUB_OFFLINE` | Disable HuggingFace downloads | `"1"` (set by server) |
| `TRANSFORMERS_OFFLINE` | Disable Transformers downloads | `"1"` (set by server) |

---

## 9. File Outputs

Directory: `_processed/{chatId}/`

| File | Description |
|------|-------------|
| `checkpoint.json` | Full pipeline state: scenes (with captions, detections, audio, descriptions), narratives, synopsis, rag_embedding metadata, step timings, benchmarks |
| `synopsis.json` | Synopsis object (same as checkpoint.synopsis) |
| `synopsis.md` | Markdown rendering with timecode links |
| `rag_embedding.json` | `{ provider, model, context_count, embedding_dim, contexts: [str], embeddings: [[float]], kmeans_clusters: {...} }` |

Temp job files: `.tmp/kairos/jobs/{runId}/` (video file during processing)

---

## 10. What Must NOT Change (External Contract)

These are the interfaces the web app depends on. Changing any of these breaks the frontend.

### Endpoint Paths & Methods
- `POST /process` — request/response schema
- `GET /jobs/{runId}` — response fields: `status`, `stage`, `percent`, `runId`, `chatId`, `error`
- `GET /jobs/{runId}/stream` — SSE event format (all fields listed in Section 4)
- `POST /query` — request fields: `videoId`, `chatId`, `query`, `topK`; response fields: `answer`, `clips`
- `GET /health`

### SSE Event Contract
- Field names: `status`, `stage`, `percent`, `runId`, `chatId`, `videoPath`, `error`
- Status values: `pending`, `running`, `completed`, `failed`
- Stage name strings (the frontend maps these to UI labels)

### MongoDB Schemas
- **chats collection:**
  - `pipeline.state` (`"processing"` / `"ready"`)
  - `pipeline.percent`, `pipeline.lastStage`, `pipeline.lastError`
  - `summary.synopsis` nested object shape (summary, video_highlights, video_timeline, suggested_clips)
  - `summary.title`, `summary.videoName`, `summary.generatedAt`

- **chat_chunks collection:**
  - Scene chunk fields: `chatId`, `type`, `sceneIndex`, `startSec`, `endSec`, `startTimecode`, `endTimecode`, `context`, `captions`, `objects`, `audioSpeech`, `audioNatural`, `embedding`
  - Meta chunk fields: `chatId`, `type`, `context`, `embedding`, plus dynamic `<type>` key

### Query Response Shape
- `clips[].startTimeSec`, `clips[].endTimeSec`, `clips[].startTimecode`, `clips[].endTimecode`, `clips[].context`, `clips[].score`

### Synopsis Schema
- All field names in Section 5 including the new `start` and `end` on `suggested_clips`
- `questions` array with `question` and `answer` fields
- `video_highlights` with `start`, `end`, `highlight`
- `video_timeline` with `timestamp`, `event`

### Database & Collection Names
- Database: `kairos` (from `MONGODB_DB_NAME`)
- Collections: `chats`, `chat_chunks`

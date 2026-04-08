import os
import sys
import uuid
import json
import time
import shutil
import subprocess
import tempfile
import requests as http_requests
from flask import Flask, request, jsonify, Response
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from werkzeug.utils import secure_filename

# Ensure repo root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.path_utils import load_kairos_env
from src.rag_convo import query_chat

# Load project environment variables from .env
load_kairos_env(override=True)

# Use locally cached HuggingFace models — skip network checks that timeout
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

app = Flask(__name__)

# Load configuration from environment
MONGODB_URI = os.getenv("MONGODB_URI")
MODEL_CWD = os.getenv("KAIROS_MODEL_CWD", os.getcwd())

# One concurrent job for GPU safety
executor = ThreadPoolExecutor(max_workers=1)

# In-memory job tracking for SSE
jobs = {}

STAGE_PROGRESS = {
    "precheck": 5,
    "downloading": 6,
    "scene_detection": 7,
    "clip_extraction": 15,
    "frame_sampling": 18,
    "frame_captioning": 20,
    "motion_sampling": 30,
    "object_detection": 33,
    "audio_prescan": 40,
    "speech_transcription": 50,
    "sound_analysis": 60,
    "scene_description": 73,
    "narrative_synthesis": 82,
    "synopsis_generation": 91,
    "embedding": 100,
}

# Ordered list: first matching phrase wins.  Longer/more-specific phrases
# come before shorter ones so "running gpt4o scene descriptions" is not
# accidentally swallowed by a substring match on a shorter phrase.
STAGE_PHRASES = [
    ("scene_detection", ["running pyscenedetect"]),
    ("frame_captioning", ["running blip"]),
    ("object_detection", ["running yolov8"]),
    ("audio_prescan", ["running audio pre-scan"]),
    ("speech_transcription", ["running whisper"]),
    ("sound_analysis", ["running mit ast"]),
    ("scene_description", ["running gpt4o scene descriptions"]),
    ("narrative_synthesis", ["running gpt4o summary narrative"]),
    ("synopsis_generation", ["running gpt4o synopsis generation"]),
    ("embedding", ["make_embedding", "rag_embedding", "inserted ", "pipeline marked as ready"]),
]


def _job_root_candidates():
    configured = os.getenv("KAIROS_JOB_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        Path(tempfile.gettempdir()) / "kairos" / "jobs",
        Path("/var/tmp") / "kairos" / "jobs",
        Path(MODEL_CWD) / ".tmp" / "kairos" / "jobs",
    ])
    return candidates


def _prepare_job_dir(run_id: str) -> Path:
    last_error = None
    for root in _job_root_candidates():
        try:
            root.mkdir(parents=True, exist_ok=True)
            run_dir = root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Unable to create job directory in any configured temp root: {last_error}")


def _active_job_count():
    return sum(1 for job in jobs.values() if job.get("status") in {"pending", "running"})


def _update_job(run_id, **updates):
    job = jobs.get(run_id)
    if not job:
        return
    job.update(updates)
    # Push status changes (completed/failed) to the event log so SSE picks them up
    if "status" in updates:
        job.setdefault("_events", []).append({
            "stage": job.get("stage", ""),
            "percent": job.get("percent", 0),
            "status": updates["status"],
            "error": job.get("error"),
        })


def _set_job_stage(run_id, stage):
    job = jobs.get(run_id)
    if not job:
        return

    percent = STAGE_PROGRESS.get(stage, job.get("percent", 0))

    job["stage"] = stage
    job["percent"] = percent

    # Append to event log so no stage update is lost between SSE polls
    job.setdefault("_events", []).append({
        "stage": stage,
        "percent": percent,
    })


def _extract_stage_from_line(line):
    lowered = line.strip().lower()
    if not lowered:
        return None

    for stage, phrases in STAGE_PHRASES:
        if any(phrase in lowered for phrase in phrases):
            return stage

    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "active_jobs": _active_job_count(),
        "gpu_available": True,
    })

def _download_video_from_url(url: str, dest_path: Path):
    """Stream-download a video from a SAS URL to a local file."""
    with http_requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


@app.route('/process', methods=['POST'])
def process_video():
    # --- JSON path: video_url from frontend ---
    if request.is_json:
        data = request.get_json()
        video_url = data.get("video_url")
        video_id = data.get("videoId")
        chat_id = data.get("chatId")
        raw_filename = data.get("videoFilename") or "input.mp4"

        if not video_url or not video_id or not chat_id:
            return jsonify({"error": "Missing video_url, videoId, or chatId"}), 400

        run_id = data.get("jobId") or str(uuid.uuid4())
        try:
            temp_dir = _prepare_job_dir(run_id)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        filename = secure_filename(raw_filename) or "input.mp4"
        video_path = temp_dir / filename

        # Initialize job state immediately so SSE can start
        jobs[run_id] = {
            "status": "pending",
            "stage": "precheck",
            "percent": STAGE_PROGRESS["precheck"],
            "runId": run_id,
            "chatId": chat_id,
            "videoPath": str(video_path),
            "_events": [{"stage": "precheck", "percent": STAGE_PROGRESS["precheck"]}],
        }

        # Download happens in background task so the frontend gets SSE updates
        executor.submit(run_pipeline_task, run_id, str(video_path), chat_id, video_url=video_url)
        return jsonify({"runId": run_id}), 202

    # --- Multipart path: file upload (legacy/fallback) ---
    video_file = request.files.get('video')
    video_id = request.form.get('videoId')
    chat_id = request.form.get('chatId')

    if not video_file or not video_id or not chat_id:
        return jsonify({"error": "Missing video, videoId, or chatId"}), 400

    run_id = str(uuid.uuid4())
    try:
        temp_dir = _prepare_job_dir(run_id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    filename = secure_filename(video_file.filename) or "input.mp4"
    video_path = temp_dir / filename
    video_file.save(str(video_path))

    # Initialize job state
    jobs[run_id] = {
        "status": "pending",
        "stage": "precheck",
        "percent": STAGE_PROGRESS["precheck"],
        "runId": run_id,
        "chatId": chat_id,
        "videoPath": str(video_path),
        "_events": [{"stage": "precheck", "percent": STAGE_PROGRESS["precheck"]}],
    }

    # Start background processing
    executor.submit(run_pipeline_task, run_id, str(video_path), chat_id)

    return jsonify({"runId": run_id}), 202

@app.route('/jobs/<run_id>', methods=['GET'])
def get_job_status(run_id):
    job = jobs.get(run_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({k: v for k, v in job.items() if not k.startswith("_")})

@app.route('/jobs/<run_id>/stream', methods=['GET'])
def stream_job_progress(run_id):
    def _job_snapshot(job, overrides=None):
        """Build a JSON-safe snapshot, excluding internal fields."""
        snap = {k: v for k, v in job.items() if not k.startswith("_")}
        if overrides:
            snap.update(overrides)
        return snap

    def generate():
        if run_id not in jobs:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        cursor = 0

        while True:
            job = jobs.get(run_id)
            if not job:
                break

            events = job.get("_events", [])

            # Drain every event the SSE client hasn't seen yet
            while cursor < len(events):
                evt = events[cursor]
                cursor += 1
                snapshot = _job_snapshot(job, evt)
                print(f"[SSE SEND] cursor={cursor}, stage={snapshot.get('stage')}, percent={snapshot.get('percent')}, status={snapshot.get('status')}", flush=True)
                yield f"data: {json.dumps(snapshot)}\n\n"

            if job["status"] in ("completed", "failed"):
                # Send one final snapshot to guarantee the client sees the terminal state
                final = _job_snapshot(job)
                print(f"[SSE SEND] FINAL stage={final.get('stage')}, percent={final.get('percent')}, status={final.get('status')}", flush=True)
                yield f"data: {json.dumps(final)}\n\n"
                break

            time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    if not data or 'videoId' not in data or 'chatId' not in data or 'query' not in data:
        return jsonify({"error": "Missing videoId, chatId, or query"}), 400

    try:
        top_k = data.get("topK", 5)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return jsonify({"error": "topK must be a number"}), 400
        top_k = max(top_k, 1)

        result = query_chat(
            chat_id=data["chatId"],
            question=data["query"],
            top_k=top_k,
            mongo_uri=MONGODB_URI,
            video_id=data["videoId"],
        )
        return jsonify({
            "answer": result["answer"],
            "clips": result["clips"],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _clear_previous_run(video_path, chat_id):
    """Wipe processed data for this specific chat so the pipeline runs fresh."""
    if not chat_id:
        return

    # 1. Delete the local _processed/<chatId> folder (isolated per user/chat)
    processed_dir = Path(MODEL_CWD) / "_processed" / chat_id
    if processed_dir.is_dir():
        shutil.rmtree(processed_dir, ignore_errors=True)
        print(f"[cleanup] Deleted processed folder: {processed_dir}", flush=True)

    # 2. Delete old log file for this chat
    log_file = Path(MODEL_CWD) / "logs" / f"_processed/{chat_id}.json"
    if log_file.is_file():
        log_file.unlink(missing_ok=True)

    # 3. Clear old MongoDB chunks/state for this chatId
    if MONGODB_URI and chat_id:
        try:
            from pymongo import MongoClient
            from bson import ObjectId
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db_name = os.getenv("MONGODB_DB_NAME", "kairos")
            db = client[db_name]
            try:
                oid = ObjectId(chat_id)
            except Exception:
                oid = chat_id
            db.chat_chunks.delete_many({"chatId": oid})
            db.chats.update_one(
                {"_id": oid},
                {"$set": {
                    "pipeline.state": "processing",
                    "pipeline.percent": 0,
                    "pipeline.lastStage": "precheck",
                    "pipeline.lastError": None,
                }},
            )
            client.close()
            print(f"[cleanup] Cleared MongoDB state for chat {chat_id}", flush=True)
        except Exception as e:
            print(f"[cleanup] WARNING: MongoDB cleanup failed: {e}", flush=True)


def run_pipeline_task(run_id, video_path, chat_id, video_url=None):
    _update_job(run_id, status="running")
    _set_job_stage(run_id, "precheck")

    # If a URL was provided, download the video first
    if video_url:
        _set_job_stage(run_id, "downloading")
        print(f"[pipeline:{run_id}] Downloading video from blob URL...", flush=True)
        try:
            _download_video_from_url(video_url, Path(video_path))
            print(f"[pipeline:{run_id}] Download complete: {video_path}", flush=True)
        except Exception as e:
            _update_job(run_id, status="failed", error=f"Video download failed: {e}")
            return

    # Wipe previous results for this video so the pipeline runs from scratch
    _clear_previous_run(video_path, chat_id)

    def _stage_callback(stage, percent):
        """Called by the pipeline whenever it enters a new stage."""
        _set_job_stage(run_id, stage)
        print(f"[pipeline:{run_id}] stage={stage} percent={percent}", flush=True)

    try:
        # Import here so the heavy libraries are only loaded once (on first
        # request) and reused for every subsequent request in this process.
        from main import run_pipeline

        run_pipeline(
            video_path=video_path,
            chat_id=chat_id,
            mongo_uri=MONGODB_URI,
            execution_mode=os.getenv("KAIROS_EXECUTION_MODE", "parallel"),
            debug=False,
            quiet=True,
            stage_callback=_stage_callback,
        )

        _update_job(run_id, status="completed")
        _set_job_stage(run_id, "embedding")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[pipeline:{run_id}] FAILED:\n{tb}", flush=True)
        _update_job(run_id, status="failed", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8000))

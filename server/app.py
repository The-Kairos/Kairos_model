import os
import sys
import uuid
import json
import time
import subprocess
import tempfile
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
    "scene_detection": 10,
    "clip_extraction": 20,
    "frame_sampling": 30,
    "frame_captioning": 40,
    "motion_sampling": 45,
    "object_detection": 50,
    "audio_prescan": 60,
    "speech_transcription": 65,
    "sound_analysis": 75,
    "scene_description": 85,
    "narrative_synthesis": 90,
    "synopsis_generation": 95,
    "embedding": 100,
}

STAGE_PHRASES = [
    ("scene_detection", ["running pyscenedetect"]),
    ("clip_extraction", ["saving clips in:"]),
    ("frame_sampling", ["saving sampled frames in:"]),
    ("frame_captioning", ["running blip"]),
    ("motion_sampling", ["saving sampled fps in:"]),
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


def _set_job_stage(run_id, stage):
    job = jobs.get(run_id)
    if not job:
        return

    percent = STAGE_PROGRESS.get(stage, job.get("percent", 0))
    if percent < job.get("percent", 0):
        percent = job["percent"]

    job["stage"] = stage
    job["percent"] = percent


def _extract_stage_from_line(line):
    lowered = line.strip().lower()
    if not lowered:
        return None

    for stage in STAGE_PROGRESS:
        if stage in lowered:
            return stage

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

@app.route('/process', methods=['POST'])
def process_video():
    video_file = request.files.get('video')
    video_id = request.form.get('videoId')
    chat_id = request.form.get('chatId')

    if not video_file or not video_id or not chat_id:
        return jsonify({"error": "Missing video, videoId, or chatId"}), 400

    # Save original video to temp location
    run_id = str(uuid.uuid4())
    try:
        temp_dir = _prepare_job_dir(run_id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    
    # Use original filename (sanitized) instead of 'input.mp4'
    raw_filename = secure_filename(video_file.filename) or "input.mp4"
    video_path = temp_dir / raw_filename
    video_file.save(str(video_path))

    # Initialize job state
    jobs[run_id] = {
        "status": "pending",
        "stage": "precheck",
        "percent": STAGE_PROGRESS["precheck"],
        "runId": run_id,
        "chatId": chat_id,
        "videoPath": str(video_path),
    }

    # Start background processing
    executor.submit(run_pipeline_task, run_id, str(video_path), chat_id)

    return jsonify({"runId": run_id}), 202

@app.route('/jobs/<run_id>', methods=['GET'])
def get_job_status(run_id):
    job = jobs.get(run_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/jobs/<run_id>/stream', methods=['GET'])
def stream_job_progress(run_id):
    def generate():
        if run_id not in jobs:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        last_stage = None
        last_percent = None
        last_status = None
        last_error = None
        
        while True:
            job = jobs.get(run_id)
            if not job:
                break
                
            # Only send if something changed
            if (
                job['stage'] != last_stage
                or job['percent'] != last_percent
                or job['status'] != last_status
                or job.get('error') != last_error
            ):
                last_stage = job['stage']
                last_percent = job['percent']
                last_status = job['status']
                last_error = job.get('error')
                yield f"data: {json.dumps(job)}\n\n"
            
            if job['status'] in ('completed', 'failed'):
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

def run_pipeline_task(run_id, video_path, chat_id):
    _update_job(run_id, status="running")
    _set_job_stage(run_id, "precheck")
    
    cmd = [
        sys.executable, "main.py", "process",
        "--video", video_path,
        "--chat-id", chat_id
    ]
    if MONGODB_URI:
        cmd.extend(["--mongo-uri", MONGODB_URI])

    # Ensure subprocess has the correct environment (MONGODB_URI, PYTHONPATH)
    sub_env = os.environ.copy()
    existing_pythonpath = sub_env.get("PYTHONPATH", "")
    sub_env["PYTHONPATH"] = f"{existing_pythonpath}:{MODEL_CWD}" if existing_pythonpath else MODEL_CWD
    if MONGODB_URI:
        sub_env["MONGODB_URI"] = MONGODB_URI

    error_lines = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=MODEL_CWD,
            env=sub_env,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue

            print(f"[pipeline:{run_id}] {line}", flush=True)

            stage = _extract_stage_from_line(line)
            if stage:
                _set_job_stage(run_id, stage)

            if "error" in line.lower() or "traceback" in line.lower():
                error_lines.append(line)
                error_lines = error_lines[-20:]

        process.wait()

        if process.returncode == 0:
            _update_job(run_id, status="completed")
            _set_job_stage(run_id, "embedding")
        else:
            _update_job(
                run_id,
                status="failed",
                error="\n".join(error_lines) if error_lines else f"Process exited with code {process.returncode}",
            )

    except Exception as e:
        _update_job(run_id, status="failed", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8000))

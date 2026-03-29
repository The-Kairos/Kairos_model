import os
import uuid
import json
import time
import subprocess
from flask import Flask, request, jsonify, Response
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from werkzeug.utils import secure_filename
from src.path_utils import load_kairos_env

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

@app.before_request
def before_request():
    pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "active_jobs": sum(1 for j in jobs.values() if j['status'] == 'running'),
        "gpu_available": True # Assuming GPU is available on this VM
    })

@app.route('/process', methods=['POST'])
def process_video():
    video_file = request.files.get('video')
    video_id = request.form.get('videoId')
    chat_id = request.form.get('chatId')
    job_id = str(uuid.uuid4())

    if not video_file or not video_id or not chat_id:
        return jsonify({"error": "Missing video, videoId, or chatId"}), 400

    # Save original video to temp location
    run_id = str(uuid.uuid4())
    temp_dir = Path(f"/tmp/kairos/jobs/{run_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original filename (sanitized) instead of 'input.mp4'
    raw_filename = secure_filename(video_file.filename) or "input.mp4"
    video_path = temp_dir / raw_filename
    video_file.save(str(video_path))

    # Initialize job state
    jobs[run_id] = {
        "status": "pending",
        "stage": "initializing",
        "percent": 0,
        "runId": run_id,
        "chatId": chat_id,
        "videoPath": str(video_path)
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
        last_stage = None
        last_percent = None
        
        while True:
            job = jobs.get(run_id)
            if not job:
                break
                
            # Only send if something changed
            if job['stage'] != last_stage or job['percent'] != last_percent:
                last_stage = job['stage']
                last_percent = job['percent']
                yield f"data: {json.dumps(job)}\n\n"
            
            if job['status'] in ('completed', 'failed'):
                # Send one last event to signal end
                yield f"data: {json.dumps(job)}\n\n"
                break
                
            time.sleep(1)
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    if not data or 'videoId' not in data or 'query' not in data:
        return jsonify({"error": "Missing videoId or query"}), 400
    
    # We can invoke the 'ask_rag' logic directly or via subprocess
    # Since we have integrated RAG logic in the model repo, we can call it here.
    # For now, let's keep it simple and return a mock/placeholder or call the utility.
    try:
        # Example call to rag utility (assuming we have a helper for this)
        # return handle_query(data)
        return jsonify({"answer": "Query received. RAG integration pending in app.py.", "clips": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_pipeline_task(run_id, video_path, chat_id):
    jobs[run_id]["status"] = "running"
    
    cmd = [
        "python", "main.py", "process",
        "--video", video_path,
        "--chat-id", chat_id
    ]
    if MONGODB_URI:
        cmd.extend(["--mongo-uri", MONGODB_URI])

    # Ensure subprocess has the correct environment (MONGODB_URI, PYTHONPATH)
    sub_env = os.environ.copy()
    sub_env["PYTHONPATH"] = sub_env.get("PYTHONPATH", "") + f":{MODEL_CWD}"
    if MONGODB_URI:
        sub_env["MONGODB_URI"] = MONGODB_URI

    try:
        # Start subprocess and capture output line-by-line to extract stage updates
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=MODEL_CWD,
            env=sub_env,
            universal_newlines=True
        )

        # Parse stdout for stage/percent updates (if model logs them in a specific format)
        # Note: Since main.py now updates MongoDB directly, we can also trust that.
        # But for SSE, we might want to watch stdout/stderr.
        for line in process.stdout:
            line = line.strip()
            # Simple parsing of stage updates (matching our StorageManager.update_pipeline_state calls)
            # This is a bit brittle, but works for MVP if the model prints its stages.
            # In our current main.py, it says "Running PysceneDetect...", "Running BLIP...", etc.
            if "Running" in line:
                jobs[run_id]["stage"] = line.replace("Running ", "").replace("...", "")
            
        process.wait()

        if process.returncode == 0:
            jobs[run_id]["status"] = "completed"
            jobs[run_id]["stage"] = "done"
            jobs[run_id]["percent"] = 100
        else:
            jobs[run_id]["status"] = "failed"
            jobs[run_id]["error"] = process.stderr.read()

    except Exception as e:
        jobs[run_id]["status"] = "failed"
        jobs[run_id]["error"] = str(e)
    finally:
        # We might want to keep job state for a while before cleaning up
        # shutil.rmtree(Path(video_path).parent, ignore_errors=True)
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8000))

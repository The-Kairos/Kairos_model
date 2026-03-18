"""
Light VLM benchmark: same pipeline as test_heavy_vlms but with light VLMs.
Uses the same Videos folder and benchmarks; writes per-video results and a summary table.
Run: python test_light_vlms/main_test.py  (from project root)
"""
import os

# Set threading env vars before any numpy/torch imports so they use all CPU cores
_CPU_COUNT = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_COUNT))

import sys
import time
import json
import argparse
import torch
import cv2

# PyTorch: use all CPU threads and enable cudnn benchmark for faster GPU convs
torch.set_num_threads(_CPU_COUNT)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root before ANY other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Pipeline imports (same as heavy)
from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_from_clip, sample_frames
from src.audio_utils import extract_scene_audio_ffmpeg
from src.audio_speech import extract_speech_asr_api
from src.audio_natural import extract_sounds
from src.frame_obj_d_yolo import detect_object_yolo
from src.scene_description import describe_scenes
from src.debug_utils import clear_frames
try:
    from src.system_metrics import get_system_usage
except ImportError:
    from test_light_vlms.benchmark_utils import get_system_usage

# Light VLM registry (same interface: load_vlm_model, caption_image)
def get_vlm_module(vlm_name):
    if vlm_name == "siglip":
        import test_light_vlms.test_siglip as vlm
    elif vlm_name == "mobilevlm":
        import test_light_vlms.test_mobilevlm as vlm
    elif vlm_name == "tinyllava":
        import test_light_vlms.test_tinyllava as vlm
    elif vlm_name == "blip2":
        import test_light_vlms.test_blip2 as vlm
    else:
        raise ValueError(f"Unknown light VLM: {vlm_name}")
    return vlm

# Same Videos dir as heavy VLMs for comparable benchmarks
# All outputs go under vlms_light/results/
VLMS_LIGHT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_ROOT / "Videos"
RESULTS_DIR = VLMS_LIGHT_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "light_vlm_metrics.json"
SUMMARY_TABLE_PATH = RESULTS_DIR / "summary_table.md"

def run_pipeline_with_vlm(video_path, vlm_name, results_dir, gcloud_json):
    """Run full pipeline with the given light VLM. Same steps as heavy."""
    video_name = Path(video_path).stem
    output_dir = results_dir / vlm_name / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> PROCESSING: {video_name} | Light VLM: {vlm_name}")

    pipeline_metrics = {}
    t_start = time.time()

    # Shared pre-processing cache (scene cuts, audio, AST, YOLO) per video,
    # stored at project root so all light VLM runs can reuse it.
    cache_dir = PROJECT_ROOT / "cache_preproc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_name}_preproc.json"

    if cache_path.exists():
        print("  [Shared] Loading cached scenes + audio + YOLO...")
        with open(cache_path, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        pipeline_metrics["scene_count"] = len(scenes)
    else:
        # 1. Scene Detection
        print("  [Step 1/6] Scene Detection...")
        scenes = get_scene_list(str(video_path))
        pipeline_metrics["scene_count"] = len(scenes)

        # 2. Audio (ASR & AST)
        print("  [Step 2/6] Audio Processing (ASR + Local AST)...")
        for scene in scenes:
            idx = scene["scene_index"]
            start, end = scene["start_seconds"], scene["end_seconds"]
            wav_path = audio_dir / f"scene_{idx:02d}.wav"
            extract_scene_audio_ffmpeg(str(video_path), str(wav_path), start, end)
            speech, _ = extract_speech_asr_api(str(wav_path), enable_logs=False)
            scene["audio_speech"] = speech
        extract_sounds(str(video_path), scenes, debug=False)

        # 3. YOLO
        print("  [Step 3/6] YOLO Object Detection...")
        from src.frame_sampling import sample_fps
        scenes = sample_fps(str(video_path), scenes, fps=1.0, new_size=320, store_meta=True)
        scenes = detect_object_yolo(scenes, model_size="model/yolov8s.pt", summary_key="yolo_detections")

        # Save a lightweight, JSON-serializable version of scenes for reuse.
        cached_scenes = clear_frames(scenes)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached_scenes, f, indent=2)


    # 4. Light VLM Captioning
    print(f"  [Step 4/6] Light VLM Captioning ({vlm_name})...")
    vlm_mod = get_vlm_module(vlm_name)
    model, processor = vlm_mod.load_vlm_model()

    for scene in scenes:
        mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
        frames = sample_from_clip(str(video_path), scene["scene_index"], mid, mid + 0.1, num_frames=1, new_size=336)
        if frames:
            pil_img = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            caption = vlm_mod.caption_image(model, processor, pil_img)
            scene["frame_captions"] = [caption]
        else:
            scene["frame_captions"] = ["None"]

    del model
    del processor
    torch.cuda.empty_cache()

    # 5. Scene Description (LLM Fusion)
    print("  [Step 5/6] Scene Description (LLM Fusion)...")
    from google import genai
    gemini_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(vertexai=True, api_key=gemini_key)
    scenes = describe_scenes(
        scenes,
        client,
        FLIP_key="frame_captions",
        ASR_key="audio_speech",
        AST_key="audio_natural",
        model="gemini-2.5-flash"
    )

    # 6. Save Results
    print("  [Step 6/6] Saving Results...")
    pipeline_metrics["total_duration_sec"] = time.time() - t_start
    pipeline_metrics["system_usage"] = get_system_usage()

    result_data = {
        "vlm": vlm_name,
        "video": video_name,
        "metrics": pipeline_metrics,
        "scenes": [
            {
                "idx": s["scene_index"],
                "start": s["start_seconds"],
                "end": s["end_seconds"],
                "vlm_caption": s["frame_captions"][0],
                "fusion_description": s.get("llm_scene_description", ""),
            }
            for s in scenes
        ],
    }

    with open(output_dir / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return pipeline_metrics


def build_summary_table(metrics_path, table_path):
    """Build a Markdown summary table from light_vlm_metrics.json."""
    if not metrics_path.exists():
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    def _fmt(val):
        return "" if val in (None, "") else f"{val:.1f}" if isinstance(val, (int, float)) else str(val)

    rows = []
    for vlm, videos in all_metrics.items():
        for video, m in videos.items():
            duration = m.get("total_duration_sec", 0)
            scenes = m.get("scene_count", 0)
            sys_use = m.get("system_usage") or {}
            gpu_mb = sys_use.get("gpu_memory_mb", "")
            cpu_pct = sys_use.get("cpu_percent", "")
            mem_mb = sys_use.get("memory_mb", "")
            rows.append((
                vlm, video,
                f"{duration:.1f}" if isinstance(duration, (int, float)) else str(duration),
                scenes,
                _fmt(gpu_mb),
                _fmt(cpu_pct),
                _fmt(mem_mb),
            ))

    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Light VLM Benchmark Summary\n\n")
        f.write("## All runs (VLM × Video)\n\n")
        f.write("| VLM | Video | Duration (s) | Scenes | GPU (MB) | CPU (%) | Memory (MB) |\n")
        f.write("|-----|-------|--------------|--------|----------|---------|-------------|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")

        # By-video comparison: for each video, compare all VLMs side by side
        videos_seen = set()
        for vlm, videos in all_metrics.items():
            for video in videos:
                videos_seen.add(video)

        if videos_seen:
            f.write("\n## By video (compare VLMs)\n\n")
            for video in sorted(videos_seen):
                f.write(f"### {video}\n\n")
                vlm_rows = []
                for vlm in all_metrics:
                    m = (all_metrics.get(vlm) or {}).get(video) or {}
                    duration = m.get("total_duration_sec", 0)
                    scenes = m.get("scene_count", 0)
                    sys_use = m.get("system_usage") or {}
                    gpu_mb = sys_use.get("gpu_memory_mb", "")
                    vlm_rows.append((
                        vlm,
                        f"{duration:.1f}" if isinstance(duration, (int, float)) else str(duration),
                        scenes,
                        _fmt(gpu_mb),
                    ))
                f.write("| VLM | Duration (s) | Scenes | GPU (MB) |\n")
                f.write("|-----|--------------|--------|----------|\n")
                for r in vlm_rows:
                    f.write("| " + " | ".join(str(x) for x in r) + " |\n")
                f.write("\n")

        f.write("\n*Same videos and pipeline as test_heavy_vlms for comparison.*\n")


def parse_args():
    """CLI options to select which VLMs and videos to run."""
    parser = argparse.ArgumentParser(description="Run light VLM benchmarks.")
    parser.add_argument(
        "--vlms",
        type=str,
        default="",
        help=(
            "Comma-separated VLM names to run (default: all). "
            "Available: siglip,mobilevlm,tinyllava,blip2"
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        default="",
        help="Comma-separated video stems or filenames to include (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    if args.videos:
        allowed = {x.strip() for x in args.videos.split(",") if x.strip()}
        videos = [v for v in videos if v.stem in allowed or v.name in allowed]

    if not videos:
        print("No videos in Videos/. Add .mp4 files to run benchmarks.")
        sys.exit(0)

    VLMS = ["siglip", "mobilevlm", "tinyllava", "blip2"]
    if args.vlms:
        requested = [x.strip() for x in args.vlms.split(",") if x.strip()]
        VLMS = [v for v in VLMS if v in requested]

    if not VLMS:
        print("No VLMs selected. Use --vlms to choose from: siglip,mobilevlm,tinyllava,blip2")
        sys.exit(0)

    GCLOUD_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # If a previous metrics file exists, load it so we can reuse metrics
    # for already-processed (vlm, video) pairs that we skip.
    existing_metrics = {}
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}

    all_metrics = {}

    for vlm in VLMS:
        all_metrics[vlm] = {}
        for video in videos:
            video_name = video.name
            video_stem = video.stem
            result_dir = RESULTS_DIR / vlm / video_stem
            result_json = result_dir / "pipeline_results.json"

            # If results already exist for this (vlm, video), reuse them instead of recomputing.
            if result_json.exists():
                print(f"SKIP: {vlm} on {video_name} (results already exist)")
                try:
                    with open(result_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    metrics = data.get("metrics", {})
                except Exception:
                    metrics = (existing_metrics.get(vlm, {}) or {}).get(video_name, {})
                all_metrics[vlm][video_name] = metrics
                continue

            try:
                metrics = run_pipeline_with_vlm(video, vlm, RESULTS_DIR, GCLOUD_JSON)
                all_metrics[vlm][video_name] = metrics
            except Exception as e:
                print(f"FAILED: {vlm} on {video_name} | Error: {e}")
                all_metrics[vlm][video_name] = {"error": str(e)}

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    build_summary_table(SUMMARY_PATH, SUMMARY_TABLE_PATH)
    print("\n\nLight VLM benchmarking complete.")
    print("  All outputs in vlms_light/results/")
    print("  Per-video:  results/<vlm>/<video>/pipeline_results.json")
    print("  Metrics:    results/light_vlm_metrics.json")
    print("  Summary:    results/summary_table.md")

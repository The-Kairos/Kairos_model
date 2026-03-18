"""
Light VLM benchmark: same pipeline as test_heavy_vlms but with light VLMs.
Uses the same Videos folder and benchmarks; writes per-video results and a summary table.
Run: python test_light_vlms/main_test.py  (from project root)
"""
import os
import sys

# Configure CPU threading BEFORE importing torch/numpy (they read these at import time)
_n_cpu = os.cpu_count() or 4
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, str(_n_cpu))

import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import cv2
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root before ANY other imports
# Script is at test/vlms_light/main_test.py -> need 3 parents to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
VLMS_LIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VLMS_LIGHT_DIR))

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
    from benchmark_utils import get_system_usage

# PyTorch: use all CPU threads, enable cuDNN auto-tuning for faster conv layers
torch.set_num_threads(_n_cpu)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Light VLM registry (same interface: load_vlm_model, caption_image)
def get_vlm_module(vlm_name):
    if vlm_name == "blip2":
        import test_blip2 as vlm
    elif vlm_name == "siglip":
        import test_siglip as vlm
    elif vlm_name == "mobilevlm":
        import test_mobilevlm as vlm
    elif vlm_name == "tinyllava":
        import test_tinyllava as vlm
    else:
        raise ValueError(f"Unknown light VLM: {vlm_name}")
    return vlm

# Paths: all results stored under vlms_light/results/ for each (vlm, video) pair
VIDEOS_DIR = PROJECT_ROOT / "Videos"
VLMS_LIGHT_DIR = Path(__file__).parent  # vlms_light folder
RESULTS_DIR = VLMS_LIGHT_DIR / "results"
CACHE_DIR = VLMS_LIGHT_DIR / "cache"
SUMMARY_PATH = VLMS_LIGHT_DIR / "light_vlm_metrics.json"
SUMMARY_TABLE_PATH = RESULTS_DIR / "summary_table.md"
SUMMARY_PIVOT_PATH = RESULTS_DIR / "summary_pivot.md"  # Videos × VLMs comparison
BY_VIDEO_DIR = RESULTS_DIR / "by_video"  # Per-video aggregated metrics


def _preproc_one_video(video_path_str):
    """Worker: pre-process one video. Used with ProcessPoolExecutor."""
    from src.frame_sampling import sample_fps
    video = Path(video_path_str)
    video_name = video.stem
    cache_path = CACHE_DIR / f"{video_name}_preproc.json"
    if cache_path.exists():
        return video_name, True  # already cached

    cache_audio_dir = CACHE_DIR / "audio"
    video_audio_dir = cache_audio_dir / video_name
    video_audio_dir.mkdir(parents=True, exist_ok=True)

    scenes = get_scene_list(str(video))
    for scene in scenes:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        wav_path = video_audio_dir / f"scene_{idx:02d}.wav"
        extract_scene_audio_ffmpeg(str(video), str(wav_path), start, end)
        speech, _ = extract_speech_asr_api(str(wav_path), enable_logs=False)
        scene["audio_speech"] = speech
    extract_sounds(str(video), scenes, debug=False)
    scenes = sample_fps(str(video), scenes, fps=1.0, new_size=320, store_meta=True)
    scenes = detect_object_yolo(scenes, model_size="model/yolov8s.pt", summary_key="yolo_detections")

    cached_scenes = clear_frames(scenes)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_scenes, f, indent=2)
    return video_name, False


def ensure_preproc_cache(videos, workers=1):
    """
    Pre-compute scene detection, audio (ASR + AST), and YOLO for all videos.
    Saves to vlms_light/cache/ so the same pre-processing is reused across all light VLMs.
    workers: parallel workers for cache building (default 1; >1 uses more GPU for YOLO).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "audio").mkdir(parents=True, exist_ok=True)

    to_process = [v for v in videos if not (CACHE_DIR / f"{Path(v).stem}_preproc.json").exists()]
    if not to_process:
        print("[CACHE] All videos already cached.")
        return

    if workers <= 1:
        for video in to_process:
            video_name = Path(video).stem
            print(f"\n[CACHE] Pre-processing {video_name} (scene detection + audio + YOLO)...")
            name, skipped = _preproc_one_video(str(video))
            if not skipped:
                print(f"  Saved cache: {name}_preproc.json")
    else:
        print(f"\n[CACHE] Pre-processing {len(to_process)} video(s) with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_preproc_one_video, str(v)): v for v in to_process}
            for fut in as_completed(futures):
                name, skipped = fut.result()
                if not skipped:
                    print(f"  Saved cache: {name}_preproc.json")
    print("")


def run_pipeline_with_vlm(video_path, vlm_name, results_dir, gcloud_json):
    """Run full pipeline with the given light VLM. Uses cached pre-processing."""
    video_name = Path(video_path).stem
    output_dir = results_dir / vlm_name / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> PROCESSING: {video_name} | Light VLM: {vlm_name}")

    pipeline_metrics = {}
    t_start = time.time()

    cache_path = CACHE_DIR / f"{video_name}_preproc.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pre-processing cache not found for {video_name}. "
            "Run with the same videos to ensure cache is built first."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    pipeline_metrics["scene_count"] = len(scenes)
    print("  [Cache] Loaded scenes + audio + YOLO from cache")

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


def _fmt_metric(m, key, fmt=None, default="-"):
    """Extract and format a metric, handling errors."""
    if isinstance(m, dict) and "error" in m:
        return "FAILED"
    val = m.get(key) if isinstance(m, dict) else None
    if val is None:
        return default
    if fmt == "float":
        return f"{float(val):.1f}"
    if fmt == "int":
        return str(int(val))
    return str(val)


def build_summary_table(metrics_path, table_path):
    """Build a Markdown summary table with all saved metrics (duration, scenes, GPU, CPU, memory)."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Light VLM Benchmark Summary\n\nNo metrics yet. Run the pipeline to populate.\n")
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    rows = []
    for vlm, videos in all_metrics.items():
        for video, m in videos.items():
            duration = _fmt_metric(m, "total_duration_sec", "float") if "error" not in (m or {}) else "FAILED"
            scenes = _fmt_metric(m, "scene_count", "int") if "error" not in (m or {}) else "-"
            sys_use = (m or {}).get("system_usage") or {}
            gpu_mb = _fmt_metric(sys_use, "gpu_memory_mb", default="-")
            cpu_pct = _fmt_metric(sys_use, "cpu_percent", default="-")
            mem_mb = _fmt_metric(sys_use, "memory_mb", "float") if sys_use.get("memory_mb") is not None else "-"
            rows.append((vlm, video, duration, scenes, gpu_mb, cpu_pct, mem_mb))

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Light VLM Benchmark Summary\n\n")
        f.write("| VLM | Video | Duration (s) | Scenes | GPU (MB) | CPU (%) | Memory (MB) |\n")
        f.write("|-----|-------|--------------|--------|----------|---------|-------------|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write("\n*Same videos and pipeline as test_heavy_vlms for comparison.*\n")


def build_pivot_table(metrics_path, pivot_path, vlms_list, videos_list):
    """Build a pivot table: rows=videos, columns=VLMs, cells=duration for easy comparison."""
    pivot_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        with open(pivot_path, "w", encoding="utf-8") as f:
            f.write("# Light VLM Pivot Comparison (Duration in seconds)\n\nNo metrics yet.\n")
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    header = "| Video | " + " | ".join(vlms_list) + " |\n"
    sep = "|-------|" + "|".join(["--------"] * len(vlms_list)) + "|\n"
    lines = [header, sep]
    for video in videos_list:
        cells = [video]
        for vlm in vlms_list:
            m = (all_metrics.get(vlm) or {}).get(video)
            if m and "error" in m:
                cells.append("FAILED")
            elif m:
                d = m.get("total_duration_sec")
                cells.append(f"{d:.1f}" if d is not None else "-")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |\n")
    with open(pivot_path, "w", encoding="utf-8") as f:
        f.write("# Light VLM Pivot Comparison (Duration in seconds)\n\n")
        f.write("Videos as rows, VLMs as columns. Compare models side-by-side per video.\n\n")
        f.writelines(lines)


def write_per_video_aggregates(all_metrics, by_video_dir, videos_list):
    """Write per-video aggregated JSON: results/by_video/<video>/all_metrics.json for each video."""
    by_video_dir.mkdir(parents=True, exist_ok=True)
    for video in videos_list:
        video_stem = Path(video).stem if hasattr(video, "stem") else Path(str(video)).stem
        video_name = video if isinstance(video, str) else getattr(video, "name", str(video))
        agg = {}
        for vlm, videos in all_metrics.items():
            if video_name in videos:
                agg[vlm] = videos[video_name]
        if agg:
            out_dir = by_video_dir / video_stem
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "all_metrics.json", "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2)


def parse_args():
    """CLI options to select which VLMs and videos to run."""
    parser = argparse.ArgumentParser(description="Run light VLM benchmarks.")
    parser.add_argument(
        "--vlms",
        type=str,
        default="",
        help=(
            "Comma-separated VLM names to run (default: all). "
            "Available: blip2,siglip,mobilevlm,tinyllava"
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        default="",
        help="Comma-separated video stems or filenames to include (default: all).",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=1,
        help="Parallel workers for cache pre-processing (default: 1). Use 2+ for multiple videos; each worker loads YOLO so requires GPU memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    print(f"[Resources] CPU threads: {_n_cpu} (OMP/MKL/etc)")
    if torch.cuda.is_available():
        print(f"[Resources] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Resources] No GPU detected, using CPU")

    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    if args.videos:
        allowed = {x.strip() for x in args.videos.split(",") if x.strip()}
        videos = [v for v in videos if v.stem in allowed or v.name in allowed]

    if not videos:
        print("No videos in Videos/. Add .mp4 files to run benchmarks.")
        sys.exit(0)

    VLMS = ["blip2", "siglip", "mobilevlm", "tinyllava"]
    if args.vlms:
        requested = [x.strip() for x in args.vlms.split(",") if x.strip()]
        VLMS = [v for v in VLMS if v in requested]

    # Filter out VLMs that fail to import (e.g. MobileVLM when repo not installed)
    _available = []
    for v in VLMS:
        try:
            get_vlm_module(v)
            _available.append(v)
        except Exception as e:
            print(f"[SKIP] {v}: not available ({e})")
    VLMS = _available

    if not VLMS:
        print("No VLMs selected. Use --vlms to choose from: blip2,siglip,mobilevlm,tinyllava")
        sys.exit(0)

    # Skip VLMs that fail to import (e.g. MobileVLM when repo not installed)
    _available = []
    for v in VLMS:
        try:
            get_vlm_module(v)
            _available.append(v)
        except Exception as e:
            print(f"[SKIP VLM] {v}: {e}")
    if not _available:
        print("No VLMs could be loaded. Install dependencies (protobuf, MobileVLM, etc.) and retry.")
        sys.exit(1)
    VLMS = _available
    print(f"[VLMs] Running: {', '.join(VLMS)}")

    GCLOUD_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    print(f"[Resources] CPU threads: {_n_cpu} | GPU: {'yes' if torch.cuda.is_available() else 'no'} | Cache workers: {getattr(args, 'cache_workers', 1)}")

    # Pre-compute shared cache (scene detection + audio + YOLO) for all videos.
    # Reused across all light VLMs so we don't redo it when testing different models.
    ensure_preproc_cache(videos, workers=getattr(args, "cache_workers", 1))

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
    build_pivot_table(SUMMARY_PATH, SUMMARY_PIVOT_PATH, VLMS, [v.name for v in videos])
    write_per_video_aggregates(all_metrics, BY_VIDEO_DIR, videos)

    print("\n\nLight VLM benchmarking complete.")
    print("  Per-video results:    vlms_light/results/<vlm>/<video>/pipeline_results.json")
    print("  Per-video aggregates: vlms_light/results/by_video/<video>/all_metrics.json")
    print("  Metrics summary:      vlms_light/light_vlm_metrics.json")
    print("  Summary table:        vlms_light/results/summary_table.md")
    print("  Pivot comparison:     vlms_light/results/summary_pivot.md")

"""
SceneWalk Benchmark Runner for Kairos.

Downloads SceneWalk videos (YouTube), runs the Kairos pipeline, extracts
scene descriptions with timestamps, and evaluates with SODA (temporal matching
+ caption scoring) plus BERTScore and ROUGE-L on matched pairs.

Usage:
    python test/benchmarks/run_scenewalk_benchmark.py --max-videos 5
    python test/benchmarks/run_scenewalk_benchmark.py --max-videos 5 --min-duration 300
    python test/benchmarks/run_scenewalk_benchmark.py --skip-pipeline   # metrics only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Isolation: add project root and src/ to path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARKS_DIR_FOR_IMPORTS = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR_FOR_IMPORTS))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Override env vars AFTER load_dotenv ──
os.environ["KAIROS_LOW_MEM"] = "TRUE"
os.environ["KAIROS_BLIP_BATCH_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("MONGODB_URI", None)
os.environ.pop("MONGODB_DB_NAME", None)

os.chdir(PROJECT_ROOT)

from dataload.scenewalk_loader import (
    load_scenewalk_dataset,
    group_segments_by_video,
    download_youtube_video,
    save_manifest,
    load_manifest,
)


def _load_metrics():
    from metrics.soda_metric import compute_soda, rouge_l_f1
    from metrics.bertscore_metric import compute_bertscore
    from metrics.rouge_metric import compute_rouge_l
    return compute_soda, rouge_l_f1, compute_bertscore, compute_rouge_l


BENCHMARKS_DIR = Path(__file__).resolve().parent
CACHE_DIR = BENCHMARKS_DIR / "cache"
VIDEO_CACHE = CACHE_DIR / "videos"
RESULTS_DIR = BENCHMARKS_DIR / "results"
MANIFEST_PATH = CACHE_DIR / "scenewalk_manifest.json"


def prepare_scenewalk_videos(max_videos=5, min_duration_sec=120, min_segments=5):
    """Find SceneWalk videos, download from YouTube. Returns list of video entries."""
    if MANIFEST_PATH.exists():
        videos = load_manifest(MANIFEST_PATH)
        print(f"[SceneWalk] Loaded cached manifest with {len(videos)} videos")
        all_present = all(
            Path(v.get("local_video_path", "")).exists()
            for v in videos
        )
        if all_present and len(videos) >= max_videos:
            return videos[:max_videos]
        print("[SceneWalk] Some videos missing or need more — re-scanning")

    print(f"[SceneWalk] Loading dataset from HuggingFace (streaming)...")
    ds = load_scenewalk_dataset()
    videos = group_segments_by_video(
        ds, max_videos=max_videos * 3,
        min_segments=min_segments, min_duration_sec=min_duration_sec,
    )
    print(f"[SceneWalk] Attempting to download {len(videos)} candidate videos...")

    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i, video in enumerate(videos):
        if len(downloaded) >= max_videos:
            break
        vid_id = video["video_id"]
        local_path = VIDEO_CACHE / f"{vid_id}.mp4"
        dur_min = video["total_duration_sec"] / 60
        print(f"[SceneWalk] [{i+1}/{len(videos)}] Downloading {vid_id} "
              f"({dur_min:.0f}min, {video['num_segments']} segments)...")
        ok = download_youtube_video(video["url"], local_path)
        if ok:
            video["local_video_path"] = str(local_path)
            downloaded.append(video)
        else:
            print(f"  [SKIP] Unavailable: {vid_id}")

    save_manifest(downloaded, MANIFEST_PATH)
    print(f"[SceneWalk] {len(downloaded)} videos ready")
    return downloaded


def _clear_gpu():
    try:
        from src.frame_captioning_blip import unload_blip
        unload_blip()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass


def _ensure_low_mem():
    os.environ["KAIROS_LOW_MEM"] = "TRUE"
    os.environ["KAIROS_BLIP_BATCH_SIZE"] = "1"
    os.environ.pop("MONGODB_URI", None)
    os.environ.pop("MONGODB_DB_NAME", None)
    try:
        import main as main_mod
        main_mod.LOW_MEM_MODE = True
        main_mod.blip_batch_size = 1
    except Exception:
        pass
    os.environ["KAIROS_LOW_MEM"] = "TRUE"
    os.environ.pop("MONGODB_URI", None)
    os.environ.pop("MONGODB_DB_NAME", None)


_yolo_patch_installed = False


def _install_yolo_batch_patch():
    global _yolo_patch_installed
    if _yolo_patch_installed:
        return
    import src.frame_obj_d_yolo as yolo_mod

    _original = yolo_mod.run_yolo_track_on_frames
    CHUNK = 200

    def _batched(model, frames, conf=0.25, iou=0.45,
                 tracker="bytetrack.yaml", device=None):
        if len(frames) <= CHUNK:
            return _original(model, frames, conf=conf, iou=iou,
                             tracker=tracker, device=device)
        import torch, gc
        all_results = []
        for start in range(0, len(frames), CHUNK):
            chunk = frames[start:start + CHUNK]
            gen = _original(model, chunk, conf=conf, iou=iou,
                            tracker=tracker, device=device)
            if gen is not None:
                all_results.extend(list(gen))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return iter(all_results) if all_results else None

    yolo_mod.run_yolo_track_on_frames = _batched
    _yolo_patch_installed = True


def run_kairos_on_video(video_path, output_dir):
    """Run the Kairos pipeline on a single video. Returns the checkpoint dict."""
    _clear_gpu()
    _ensure_low_mem()
    _install_yolo_batch_patch()

    try:
        from main import run_pipeline
        run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            execution_mode="sequential",
            quiet=True,
        )
    except Exception as e:
        print(f"  [WARN] Pipeline error (may be non-fatal): {type(e).__name__}: {e}")
    finally:
        _clear_gpu()

    checkpoint_path = Path(output_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scene_segments(checkpoint):
    """Extract Kairos scene descriptions with timestamps as SODA-format segments."""
    scenes = checkpoint.get("scenes", [])
    segments = []
    for scene in scenes:
        text = scene.get("llm_scene_description", "")
        if not text:
            continue
        segments.append({
            "start": scene.get("start_seconds", 0),
            "end": scene.get("end_seconds", 0),
            "text": text,
        })
    return segments


def _ref_segments_from_entry(video_entry):
    """Convert SceneWalk ground truth to SODA-format segments."""
    segments = []
    for seg in video_entry.get("segments", []):
        segments.append({
            "start": seg["start_sec"],
            "end": seg["end_sec"],
            "text": seg["caption"],
        })
    return segments


def run_benchmark(videos, skip_pipeline=False):
    """Run full benchmark: pipeline + metrics."""
    all_pred_segments = []
    all_ref_segments = []
    video_results = []

    sw_output_root = CACHE_DIR / "scenewalk_outputs"
    sw_output_root.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos):
        vid_id = video.get("video_id", "unknown")
        video_path = video.get("local_video_path", "")
        output_dir = str(sw_output_root / f"video_{i:03d}")
        dur_min = video.get("total_duration_sec", 0) / 60
        n_segs = video.get("num_segments", 0)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(videos)}] {vid_id} | {dur_min:.1f} min | {n_segs} GT segments")
        print(f"{'='*60}")

        checkpoint_path = Path(output_dir) / "checkpoint.json"
        if skip_pipeline and checkpoint_path.exists():
            print(f"  [CACHED] Using existing pipeline output")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        elif skip_pipeline:
            print(f"  [SKIP] No cached output and --skip-pipeline set")
            continue
        else:
            print(f"  Running Kairos pipeline...")
            t0 = time.perf_counter()
            checkpoint = run_kairos_on_video(video_path, output_dir)
            elapsed = time.perf_counter() - t0
            print(f"  Pipeline completed in {elapsed:.1f}s")
            if checkpoint is None:
                print(f"  [ERROR] Pipeline produced no checkpoint")
                continue

        pred_segs = extract_scene_segments(checkpoint)
        ref_segs = _ref_segments_from_entry(video)

        if not pred_segs:
            print(f"  [WARN] No scene descriptions found in checkpoint")
            continue

        all_pred_segments.append(pred_segs)
        all_ref_segments.append(ref_segs)

        video_results.append({
            "index": i,
            "video_id": vid_id,
            "duration_min": round(dur_min, 1),
            "num_kairos_scenes": len(pred_segs),
            "num_gt_segments": len(ref_segs),
        })

    if not all_pred_segments:
        print("\n[ERROR] No predictions to evaluate")
        return None

    print(f"\n{'='*60}")
    print(f"Computing metrics on {len(all_pred_segments)} videos...")
    print(f"{'='*60}")

    compute_soda, rouge_l_scorer, compute_bertscore, compute_rouge_l = _load_metrics()

    # --- SODA with ROUGE-L scorer ---
    print("  Computing SODA (ROUGE-L scorer)...")
    soda_results = []
    for j, (preds, refs) in enumerate(zip(all_pred_segments, all_ref_segments)):
        result = compute_soda(preds, refs, scorer_fn=rouge_l_scorer, iou_threshold=0.3)
        soda_results.append(result)
        video_results[j]["soda_f1"] = result["f1"]
        video_results[j]["soda_precision"] = result["precision"]
        video_results[j]["soda_recall"] = result["recall"]
        video_results[j]["soda_matched"] = result["num_matched"]

    n = len(soda_results)
    mean_soda_f1 = sum(r["f1"] for r in soda_results) / n
    mean_soda_p = sum(r["precision"] for r in soda_results) / n
    mean_soda_r = sum(r["recall"] for r in soda_results) / n
    print(f"    Mean SODA F1: {mean_soda_f1:.4f}")

    # --- BERTScore and ROUGE-L on matched pairs (from best SODA alignment) ---
    matched_preds = []
    matched_refs = []
    for j, sr in enumerate(soda_results):
        for m in sr["matches"]:
            pi = m["pred_idx"]
            ri = m["ref_idx"]
            matched_preds.append(all_pred_segments[j][pi]["text"])
            matched_refs.append(all_ref_segments[j][ri]["text"])

    bertscore_agg = {"mean_f1": 0, "mean_precision": 0, "mean_recall": 0}
    rouge_agg = {"mean_f1": 0, "mean_precision": 0, "mean_recall": 0}

    if matched_preds:
        _clear_gpu()
        print(f"  Computing BERTScore on {len(matched_preds)} matched pairs...")
        bertscore_agg = compute_bertscore(matched_preds, matched_refs)
        print(f"    Mean F1: {bertscore_agg['mean_f1']:.4f}")

        print(f"  Computing ROUGE-L on matched pairs...")
        rouge_agg = compute_rouge_l(matched_preds, matched_refs)
        print(f"    Mean F1: {rouge_agg['mean_f1']:.4f}")

    results = {
        "dataset": "SceneWalk",
        "num_videos": len(all_pred_segments),
        "aggregate": {
            "soda_f1": mean_soda_f1,
            "soda_precision": mean_soda_p,
            "soda_recall": mean_soda_r,
            "matched_bertscore_f1": bertscore_agg["mean_f1"],
            "matched_bertscore_precision": bertscore_agg["mean_precision"],
            "matched_bertscore_recall": bertscore_agg["mean_recall"],
            "matched_rouge_l_f1": rouge_agg["mean_f1"],
            "total_matched_pairs": len(matched_preds),
        },
        "per_video": video_results,
    }

    return results


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"scenewalk_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return out_path


def print_summary(results):
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"  SceneWalk Benchmark Results ({results['num_videos']} videos)")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'Score':>10}")
    print(f"  {'-'*45}")
    print(f"  {'SODA F1 (ROUGE-L scorer)':<35} {agg['soda_f1']:>10.4f}")
    print(f"  {'SODA Precision':<35} {agg['soda_precision']:>10.4f}")
    print(f"  {'SODA Recall':<35} {agg['soda_recall']:>10.4f}")
    print(f"  {'Matched BERTScore F1':<35} {agg['matched_bertscore_f1']:>10.4f}")
    print(f"  {'Matched ROUGE-L F1':<35} {agg['matched_rouge_l_f1']:>10.4f}")
    print(f"  {'Total matched pairs':<35} {agg['total_matched_pairs']:>10d}")
    print(f"{'='*60}")


def generate_reports(results):
    """Generate MD benchmark report and comparison files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        print("  [WARN] No manifest found, skipping report generation")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    agg = results["aggregate"]
    sw_output_root = CACHE_DIR / "scenewalk_outputs"

    # ── Metric report ──
    lines = [
        "# SceneWalk Benchmark Report — Kairos\n",
        f"**Dataset:** SceneWalk (IVLLab/SceneWalk)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Videos:** {results['num_videos']}\n",
        "---\n",
        "## Aggregate Metrics\n",
        "| Metric                    |   Score |",
        "|---------------------------|---------|",
    ]
    for key, label in [
        ("soda_f1", "SODA F1 (ROUGE-L scorer)"), ("soda_precision", "SODA Precision"),
        ("soda_recall", "SODA Recall"), ("matched_bertscore_f1", "Matched BERTScore F1"),
        ("matched_bertscore_precision", "Matched BERTScore Precision"),
        ("matched_bertscore_recall", "Matched BERTScore Recall"),
        ("matched_rouge_l_f1", "Matched ROUGE-L F1"),
    ]:
        if key in agg:
            lines.append(f"| {label:<25} | {agg[key]:>7.4f} |")
    lines.append(f"| {'Total Matched Pairs':<25} | {agg.get('total_matched_pairs',0):>7d} |")

    lines += ["\n---\n", "## Per-Video Breakdown\n",
              "| # | Video ID | Duration | Kairos Scenes | GT Segments | Matched | SODA F1 |",
              "|---|----------|----------|---------------|-------------|---------|---------|"]
    for v in results["per_video"]:
        lines.append(f"| {v['index']+1} | {v['video_id']} | {v['duration_min']:.0f} min | "
                     f"{v['num_kairos_scenes']} | {v['num_gt_segments']} | "
                     f"{v.get('soda_matched',0)} | {v.get('soda_f1',0):.4f} |")

    with open(RESULTS_DIR / "scenewalk_benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {RESULTS_DIR / 'scenewalk_benchmark_report.md'}")

    # ── Comparison (GT vs Kairos) ──
    comparisons = []
    for i, v in enumerate(results["per_video"]):
        vid_id = v["video_id"]
        entry = manifest[i] if i < len(manifest) else {}
        checkpoint_path = sw_output_root / f"video_{i:03d}/checkpoint.json"

        kairos_scenes = []
        kairos_synopsis = ""
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            syn = cp.get("synopsis", {})
            if isinstance(syn, dict):
                kairos_synopsis = syn.get("summary", "")
            for scene in cp.get("scenes", []):
                desc = scene.get("llm_scene_description", "")
                if desc:
                    kairos_scenes.append({
                        "start_sec": scene.get("start_seconds", 0),
                        "end_sec": scene.get("end_seconds", 0),
                        "kairos_description": desc,
                    })

        gt_segments = []
        for seg in entry.get("segments", []):
            gt_segments.append({
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "ground_truth_caption": seg["caption"],
            })

        comparisons.append({
            "video_number": i + 1,
            "video_id": vid_id,
            "video_url": f"https://www.youtube.com/watch?v={vid_id}",
            "duration_min": v.get("duration_min", 0),
            "num_gt_segments": len(gt_segments),
            "num_kairos_scenes": len(kairos_scenes),
            "kairos_full_synopsis": kairos_synopsis,
            "ground_truth_segments": gt_segments,
            "kairos_scene_descriptions": kairos_scenes,
        })

    with open(RESULTS_DIR / "scenewalk_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# SceneWalk Benchmark — Ground Truth vs Kairos Comparison\n",
        f"**Dataset:** SceneWalk (IVLLab/SceneWalk)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Videos:** {len(comparisons)}\n",
        "---\n",
    ]
    for c in comparisons:
        md_lines += [
            f"## Video {c['video_number']}: {c['video_id']}\n",
            f"- **URL:** {c['video_url']}",
            f"- **Duration:** {c['duration_min']} min",
            f"- **Ground truth segments:** {c['num_gt_segments']}",
            f"- **Kairos scenes detected:** {c['num_kairos_scenes']}\n",
            "### Kairos Full Synopsis\n",
            f"> {c['kairos_full_synopsis']}\n",
            "### Scene-by-Scene Comparison\n",
        ]
        gt = c["ground_truth_segments"]
        kairos = c["kairos_scene_descriptions"]
        shown = 0
        for ki, ks in enumerate(kairos):
            k_start, k_end = ks["start_sec"], ks["end_sec"]
            overlapping = [gs for gs in gt
                           if max(0, min(k_end, gs["end_sec"]) - max(k_start, gs["start_sec"])) > 0]
            ts = f"{int(k_start//60):02d}:{int(k_start%60):02d} – {int(k_end//60):02d}:{int(k_end%60):02d}"
            md_lines.append(f"#### Scene {ki+1} [{ts}]\n")
            desc = ks["kairos_description"]
            if len(desc) > 500:
                desc = desc[:500] + "..."
            md_lines += [f"**Kairos:**", f"> {desc}\n"]
            if overlapping:
                md_lines.append(f"**Ground Truth ({len(overlapping)} overlapping):**")
                for gs in overlapping[:3]:
                    gt_ts = f"{int(gs['start_sec']//60):02d}:{int(gs['start_sec']%60):02d}–{int(gs['end_sec']//60):02d}:{int(gs['end_sec']%60):02d}"
                    cap = gs["ground_truth_caption"]
                    if len(cap) > 300:
                        cap = cap[:300] + "..."
                    md_lines.append(f"> [{gt_ts}] {cap}\n")
            else:
                md_lines.append("**Ground Truth:** *No overlapping segments*\n")
            shown += 1
            if shown >= 10 and len(kairos) > 12:
                md_lines += ["---\n", f"*... {len(kairos)-shown} more scenes (see scenewalk_comparison.json) ...*\n"]
                break
        md_lines.append("---\n")

    with open(RESULTS_DIR / "scenewalk_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Comparison saved to {RESULTS_DIR / 'scenewalk_comparison.md'}")


def main():
    parser = argparse.ArgumentParser(description="Run SceneWalk benchmark for Kairos")
    parser.add_argument("--max-videos", type=int, default=5,
                        help="Number of SceneWalk videos to benchmark (default: 5)")
    parser.add_argument("--min-duration", type=int, default=120,
                        help="Minimum video duration in seconds (default: 120)")
    parser.add_argument("--min-segments", type=int, default=5,
                        help="Minimum ground truth segments per video (default: 5)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline execution, use cached outputs only")
    args = parser.parse_args()

    print(f"[SceneWalk Benchmark] Starting with max_videos={args.max_videos}, "
          f"min_duration={args.min_duration}s, min_segments={args.min_segments}")

    videos = prepare_scenewalk_videos(
        max_videos=args.max_videos,
        min_duration_sec=args.min_duration,
        min_segments=args.min_segments,
    )
    if not videos:
        print("[ERROR] No SceneWalk videos available")
        sys.exit(1)

    results = run_benchmark(videos, skip_pipeline=args.skip_pipeline)
    if results is None:
        sys.exit(1)

    print_summary(results)
    save_results(results)
    generate_reports(results)


if __name__ == "__main__":
    main()

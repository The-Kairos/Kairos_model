"""
TIB Benchmark Runner for Kairos.

Downloads TIB videos, runs the Kairos pipeline, extracts the generated synopsis,
and compares it against the human-written abstract using BERTScore, ROUGE-L, and BLEU.

Usage:
    python test/benchmarks/run_tib_benchmark.py --max-videos 5
    python test/benchmarks/run_tib_benchmark.py --max-videos 5 --language en --min-duration 1800
    python test/benchmarks/run_tib_benchmark.py --skip-pipeline   # metrics only (reuse cached outputs)
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

# ── Override env vars AFTER load_dotenv (it sets KAIROS_LOW_MEM=AUTO from .env) ──
# LOW_MEM_MODE ensures BLIP unloads after captioning, freeing ~7 GiB GPU before YOLO.
os.environ["KAIROS_LOW_MEM"] = "TRUE"
os.environ["KAIROS_BLIP_BATCH_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("MONGODB_URI", None)
os.environ.pop("MONGODB_DB_NAME", None)

os.chdir(PROJECT_ROOT)

from dataload.tib_loader import (
    load_tib_dataset,
    filter_usable_entries,
    download_video,
    make_video_filename,
    save_manifest,
    load_manifest,
)


def _load_metrics():
    """Lazy-load metric functions to avoid heavy imports at startup."""
    from metrics.bertscore_metric import compute_bertscore
    from metrics.rouge_metric import compute_rouge_l
    from metrics.bleu_metric import compute_bleu
    return compute_bertscore, compute_rouge_l, compute_bleu

BENCHMARKS_DIR = Path(__file__).resolve().parent
CACHE_DIR = BENCHMARKS_DIR / "cache"
VIDEO_CACHE = CACHE_DIR / "videos"
RESULTS_DIR = BENCHMARKS_DIR / "results"
MANIFEST_PATH = CACHE_DIR / "tib_manifest.json"


def prepare_tib_videos(max_videos=5, language=None, min_duration_sec=0):
    """Download TIB dataset entries and their videos. Returns list of entries with local paths."""
    if MANIFEST_PATH.exists():
        entries = load_manifest(MANIFEST_PATH)
        print(f"[TIB] Loaded cached manifest with {len(entries)} entries")
        if min_duration_sec:
            entries = [e for e in entries if e.get("estimated_duration_sec", 0) >= min_duration_sec]
        all_present = all(
            Path(e.get("local_video_path", "")).exists()
            for e in entries
        )
        if all_present and len(entries) >= max_videos:
            return entries[:max_videos]
        print("[TIB] Some videos missing or need more — re-downloading")

    print(f"[TIB] Loading dataset from HuggingFace (streaming)...")
    ds = load_tib_dataset()
    entries = list(filter_usable_entries(ds, max_entries=max_videos, language=language,
                                         min_duration_sec=min_duration_sec))
    print(f"[TIB] Found {len(entries)} usable entries")

    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i, entry in enumerate(entries):
        fname = make_video_filename(entry)
        local_path = VIDEO_CACHE / fname
        dur = entry.get("estimated_duration_sec", 0)
        dur_str = f" ({dur/60:.0f}min)" if dur else ""
        print(f"[TIB] Downloading video {i+1}/{len(entries)}: {entry['title'][:60]}{dur_str}...")
        ok = download_video(entry["video_url"], local_path)
        if ok:
            entry["local_video_path"] = str(local_path)
            downloaded.append(entry)
        else:
            print(f"  [SKIP] Failed to download: {entry['title'][:60]}")

    save_manifest(downloaded, MANIFEST_PATH)
    print(f"[TIB] {len(downloaded)} videos ready")
    return downloaded


def _clear_gpu():
    """Free GPU memory between pipeline runs — unload all cached models."""
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
    """Force LOW_MEM_MODE=True in main module, regardless of .env or load_kairos_env().
    Also re-apply env overrides since main.py's import chain calls load_kairos_env(override=True)
    which resets everything from .env."""
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
    # Re-set AFTER import — load_kairos_env(override=True) in main.py resets env vars from .env
    os.environ["KAIROS_LOW_MEM"] = "TRUE"
    os.environ.pop("MONGODB_URI", None)
    os.environ.pop("MONGODB_DB_NAME", None)


_yolo_patch_installed = False


def _install_yolo_batch_patch():
    """Monkeypatch YOLO tracking to process frames in small batches, avoiding GPU OOM.

    ultralytics model.track(frames, stream=True) preprocesses ALL frames into a single
    batch tensor before yielding results.  For long scenes (600s at 4 FPS = 2400 frames),
    the intermediate conv2d allocation exceeds 22 GiB.  This patch splits frames into
    chunks of CHUNK so each batch stays under ~2.5 GiB."""
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


def extract_synopsis_text(checkpoint):
    """Extract the generated synopsis summary text from a Kairos checkpoint."""
    synopsis = checkpoint.get("synopsis", {})
    if isinstance(synopsis, dict):
        return synopsis.get("summary", "")
    return ""


def run_benchmark(entries, skip_pipeline=False):
    """
    Run full benchmark: pipeline + metrics.
    Returns a results dict with per-video and aggregate scores.
    """
    predictions = []
    references = []
    video_results = []

    tib_output_root = CACHE_DIR / "tib_outputs"
    tib_output_root.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        title = entry.get("title", "unknown")[:60]
        video_path = entry.get("local_video_path", "")
        ground_truth = entry.get("abstract", "")
        output_dir = str(tib_output_root / f"video_{i:03d}")

        duration_sec = entry.get("estimated_duration_sec", 0)
        dur_str = f" | {duration_sec/60:.1f} min" if duration_sec else ""
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(entries)}] {title}{dur_str}")
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

        generated = extract_synopsis_text(checkpoint)
        if not generated:
            print(f"  [WARN] No synopsis text found in checkpoint")
            continue

        predictions.append(generated)
        references.append(ground_truth)
        duration_sec = entry.get("estimated_duration_sec", 0)
        video_results.append({
            "index": i,
            "title": title,
            "doi": entry.get("doi", ""),
            "duration_sec": duration_sec,
            "duration_min": round(duration_sec / 60, 1),
            "ground_truth_length": len(ground_truth.split()),
            "generated_length": len(generated.split()),
            "ground_truth_preview": ground_truth[:200],
            "generated_preview": generated[:200],
        })

    if not predictions:
        print("\n[ERROR] No predictions to evaluate")
        return None

    print(f"\n{'='*60}")
    print(f"Computing metrics on {len(predictions)} videos...")
    print(f"{'='*60}")

    compute_bertscore, compute_rouge_l, compute_bleu = _load_metrics()

    print("  Computing BERTScore...")
    bertscore = compute_bertscore(predictions, references)
    print(f"    Mean F1: {bertscore['mean_f1']:.4f}")

    print("  Computing ROUGE-L...")
    rouge = compute_rouge_l(predictions, references)
    print(f"    Mean F1: {rouge['mean_f1']:.4f}")

    print("  Computing BLEU...")
    bleu = compute_bleu(predictions, references)
    print(f"    Mean BLEU-1: {bleu['mean_bleu_1']:.4f}")
    print(f"    Mean BLEU-4: {bleu['mean_bleu_4']:.4f}")

    for j, vr in enumerate(video_results):
        vr["bertscore_f1"] = bertscore["f1"][j]
        vr["rouge_l_f1"] = rouge["f1"][j]
        vr["bleu_1"] = bleu["bleu_1"][j]
        vr["bleu_4"] = bleu["bleu_4"][j]

    results = {
        "dataset": "TIB",
        "num_videos": len(predictions),
        "aggregate": {
            "bertscore_f1": bertscore["mean_f1"],
            "bertscore_precision": bertscore["mean_precision"],
            "bertscore_recall": bertscore["mean_recall"],
            "rouge_l_f1": rouge["mean_f1"],
            "rouge_l_precision": rouge["mean_precision"],
            "rouge_l_recall": rouge["mean_recall"],
            "bleu_1": bleu["mean_bleu_1"],
            "bleu_2": bleu["mean_bleu_2"],
            "bleu_3": bleu["mean_bleu_3"],
            "bleu_4": bleu["mean_bleu_4"],
        },
        "per_video": video_results,
    }

    return results


def save_results(results):
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"tib_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return out_path


def print_summary(results):
    """Print a formatted summary table."""
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"  TIB Benchmark Results ({results['num_videos']} videos)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*35}")
    print(f"  {'BERTScore F1':<25} {agg['bertscore_f1']:>10.4f}")
    print(f"  {'BERTScore Precision':<25} {agg['bertscore_precision']:>10.4f}")
    print(f"  {'BERTScore Recall':<25} {agg['bertscore_recall']:>10.4f}")
    print(f"  {'ROUGE-L F1':<25} {agg['rouge_l_f1']:>10.4f}")
    print(f"  {'BLEU-1':<25} {agg['bleu_1']:>10.4f}")
    print(f"  {'BLEU-2':<25} {agg['bleu_2']:>10.4f}")
    print(f"  {'BLEU-3':<25} {agg['bleu_3']:>10.4f}")
    print(f"  {'BLEU-4':<25} {agg['bleu_4']:>10.4f}")
    print(f"{'='*60}")


def generate_reports(results):
    """Generate MD benchmark report and comparison files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "tib_manifest.json"
    if not manifest_path.exists():
        print("  [WARN] No manifest found, skipping report generation")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # ── Metric report ──
    agg = results["aggregate"]
    lines = [
        "# TIB Benchmark Report — Kairos\n",
        f"**Dataset:** TIB AV-Portal (academic presentations)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Videos:** {results['num_videos']}\n",
        "---\n",
        "## Aggregate Metrics\n",
        "| Metric              |   Score |",
        "|---------------------|---------|",
    ]
    for key, label in [
        ("bertscore_f1", "BERTScore F1"), ("bertscore_precision", "BERTScore Precision"),
        ("bertscore_recall", "BERTScore Recall"), ("rouge_l_f1", "ROUGE-L F1"),
        ("rouge_l_precision", "ROUGE-L Precision"), ("rouge_l_recall", "ROUGE-L Recall"),
        ("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2"), ("bleu_3", "BLEU-3"), ("bleu_4", "BLEU-4"),
    ]:
        if key in agg:
            lines.append(f"| {label:<19} | {agg[key]:>7.4f} |")

    lines += ["\n---\n", "## Per-Video Breakdown\n",
              "| # | Title | Duration | BERTScore F1 | ROUGE-L F1 | BLEU-1 |",
              "|---|-------|----------|-------------|-----------|--------|"]
    for v in results["per_video"]:
        lines.append(f"| {v['index']+1} | {v['title'][:45]} | {v['duration_min']:.0f} min | "
                     f"{v.get('bertscore_f1',0):.4f} | {v.get('rouge_l_f1',0):.4f} | {v.get('bleu_1',0):.4f} |")

    with open(RESULTS_DIR / "tib_benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {RESULTS_DIR / 'tib_benchmark_report.md'}")

    # ── Comparison (GT vs Kairos) ──
    comparisons = []
    for i, v in enumerate(results["per_video"]):
        idx = v["index"]
        entry = manifest[idx] if idx < len(manifest) else {}
        checkpoint_path = CACHE_DIR / f"tib_outputs/video_{idx:03d}/checkpoint.json"

        kairos_synopsis = ""
        num_scenes = 0
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            num_scenes = len(cp.get("scenes", []))
            syn = cp.get("synopsis", {})
            if isinstance(syn, dict):
                kairos_synopsis = syn.get("summary", "")
            elif isinstance(syn, str):
                kairos_synopsis = syn

        comparisons.append({
            "video_number": i + 1,
            "title": entry.get("title", v.get("title", "")),
            "doi": entry.get("doi", v.get("doi", "")),
            "video_url": entry.get("video_url", ""),
            "duration_min": v.get("duration_min", 0),
            "num_kairos_scenes": num_scenes,
            "ground_truth_abstract": entry.get("abstract", ""),
            "kairos_synopsis": kairos_synopsis,
        })

    with open(RESULTS_DIR / "tib_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# TIB Benchmark — Ground Truth vs Kairos Comparison\n",
        f"**Dataset:** TIB AV-Portal (academic presentations)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Videos:** {len(comparisons)}\n",
        "---\n",
    ]
    for c in comparisons:
        md_lines += [
            f"## Video {c['video_number']}: {c['title']}\n",
            f"- **DOI:** `{c['doi']}`",
        ]
        if c["video_url"]:
            md_lines.append(f"- **URL:** {c['video_url']}")
        md_lines += [
            f"- **Duration:** {c['duration_min']} min",
            f"- **Kairos scenes detected:** {c['num_kairos_scenes']}\n",
            "### Ground Truth (Human Abstract)\n",
            f"> {c['ground_truth_abstract']}\n",
            "### Kairos Synopsis\n",
            f"> {c['kairos_synopsis']}\n",
            "---\n",
        ]

    with open(RESULTS_DIR / "tib_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Comparison saved to {RESULTS_DIR / 'tib_comparison.md'}")


def main():
    parser = argparse.ArgumentParser(description="Run TIB benchmark for Kairos")
    parser.add_argument("--max-videos", type=int, default=5,
                        help="Number of TIB videos to benchmark (default: 5)")
    parser.add_argument("--language", type=str, default=None,
                        help="Filter to specific language (e.g. 'en')")
    parser.add_argument("--min-duration", type=int, default=0,
                        help="Minimum video duration in seconds (e.g. 1800 for 30+ min)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline execution, use cached outputs only")
    args = parser.parse_args()

    if args.min_duration:
        print(f"[TIB Benchmark] Starting with max_videos={args.max_videos}, min_duration={args.min_duration}s ({args.min_duration//60}min)")
    else:
        print(f"[TIB Benchmark] Starting with max_videos={args.max_videos}")

    entries = prepare_tib_videos(max_videos=args.max_videos, language=args.language,
                                 min_duration_sec=args.min_duration)
    if not entries:
        print("[ERROR] No TIB entries available")
        sys.exit(1)

    results = run_benchmark(entries, skip_pipeline=args.skip_pipeline)
    if results is None:
        sys.exit(1)

    print_summary(results)
    save_results(results)
    generate_reports(results)


if __name__ == "__main__":
    main()

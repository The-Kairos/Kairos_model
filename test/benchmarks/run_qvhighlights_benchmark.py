"""
QVHighlights Clip Retrieval Benchmark Runner for Kairos.

Downloads QVHighlights videos and annotations, runs the Kairos pipeline,
retrieves clips via embedding cosine similarity, and evaluates with
standard moment retrieval metrics (R@K at IoU thresholds, mIoU).

Kairos operates zero-shot: no training on QVHighlights data. It segments
the video into scenes, generates multimodal descriptions, embeds them,
and retrieves the best-matching scene for each query.

Usage:
    python test/benchmarks/run_qvhighlights_benchmark.py --max-videos 5
    python test/benchmarks/run_qvhighlights_benchmark.py --max-videos 50 --merge-adjacent
    python test/benchmarks/run_qvhighlights_benchmark.py --skip-pipeline
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Isolation: add project root and src/ to path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ["KAIROS_LOW_MEM"] = "TRUE"
os.environ["KAIROS_BLIP_BATCH_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("MONGODB_URI", None)
os.environ.pop("MONGODB_DB_NAME", None)

os.chdir(PROJECT_ROOT)

from dataload.qvhighlights_loader import (
    prepare_qvhighlights,
    group_queries_by_video,
    load_annotations,
    download_qvhighlights_annotations,
    save_manifest,
)

CACHE_DIR = BENCHMARKS_DIR / "cache"
VIDEO_CACHE = CACHE_DIR / "qvh_videos"
RESULTS_DIR = BENCHMARKS_DIR / "results"


# ── Pipeline helpers (adapted from run_scenewalk_benchmark.py) ──

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
        print(f"  [WARN] Pipeline error: {type(e).__name__}: {e}")
    finally:
        _clear_gpu()

    checkpoint_path = Path(output_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Clip retrieval logic ──

def _load_retrieval_components():
    """Lazy-load RAG components to avoid import overhead until needed."""
    from src.rag_convo import (
        load_rag_embeddings,
        embed_question,
        format_scene_embedding,
        _embedding_values,
        _cosine_similarity,
        _to_vector,
        compute_kmeans_clusters,
    )
    return {
        "load_rag_embeddings": load_rag_embeddings,
        "embed_question": embed_question,
        "format_scene_embedding": format_scene_embedding,
        "_embedding_values": _embedding_values,
        "_cosine_similarity": _cosine_similarity,
        "_to_vector": _to_vector,
        "compute_kmeans_clusters": compute_kmeans_clusters,
    }


def retrieve_clips_for_query(query_text, checkpoint, rag_data, top_k=5,
                              rag_fns=None):
    """Retrieve top-K scene clips for a query using embedding cosine similarity.

    Args:
        query_text: Natural language query
        checkpoint: Kairos checkpoint dict (has "scenes")
        rag_data: Loaded rag_embedding.json dict (has "contexts", "embeddings")
        top_k: Number of clips to retrieve
        rag_fns: Dict of lazy-loaded RAG functions

    Returns:
        List of {"start": float, "end": float, "score": float, "scene_idx": int}
    """
    if rag_fns is None:
        rag_fns = _load_retrieval_components()

    scenes = checkpoint.get("scenes", [])
    num_scenes = len(scenes)
    if num_scenes == 0:
        return []

    all_contexts = rag_data.get("contexts", [])
    all_embeddings = rag_data.get("embeddings", [])

    scene_embeddings = all_embeddings[:num_scenes]
    if not scene_embeddings:
        return []

    query_embedding = rag_fns["embed_question"](query_text)
    if isinstance(query_embedding, list) and len(query_embedding) > 0:
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
    query_vec = rag_fns["_to_vector"](
        rag_fns["_embedding_values"](query_embedding)
    )
    if query_vec is None:
        return []

    scores = []
    for i, emb in enumerate(scene_embeddings):
        emb_vec = rag_fns["_to_vector"](
            rag_fns["_embedding_values"](emb)
        )
        if emb_vec is None:
            scores.append(0.0)
            continue
        score = rag_fns["_cosine_similarity"](query_vec, emb_vec)
        scores.append(float(score))

    top_indices = np.argsort(scores)[-top_k:][::-1]

    clips = []
    for idx in top_indices:
        idx = int(idx)
        if idx >= num_scenes:
            continue
        scene = scenes[idx]
        clips.append({
            "start": scene.get("start_seconds", 0),
            "end": scene.get("end_seconds", 0),
            "score": scores[idx],
            "scene_idx": idx,
        })

    return clips


def merge_adjacent_clips(clips, gap_threshold=2.0):
    """Merge clips that are adjacent or overlapping.

    Adjacent scenes within gap_threshold seconds are merged into one clip.
    The merged clip's score is the max of its constituents.
    """
    if len(clips) <= 1:
        return clips

    sorted_clips = sorted(clips, key=lambda c: c["start"])
    merged = [sorted_clips[0].copy()]

    for clip in sorted_clips[1:]:
        if clip["start"] - merged[-1]["end"] <= gap_threshold:
            merged[-1]["end"] = max(merged[-1]["end"], clip["end"])
            merged[-1]["score"] = max(merged[-1]["score"], clip["score"])
        else:
            merged.append(clip.copy())

    merged.sort(key=lambda c: c["score"], reverse=True)
    return merged


# ── Benchmark orchestration ──

def run_benchmark(video_entries, annotations_by_vid, split="val",
                  skip_pipeline=False, top_k=5, merge_adjacent=False,
                  merge_gap_sec=2.0, output_cache_name="qvhighlights_outputs"):
    """Run the full QVHighlights benchmark.

    Returns results dict with aggregate metrics and per-video/per-query details.
    """
    from metrics.moment_retrieval_metric import compute_moment_retrieval

    output_root = CACHE_DIR / output_cache_name
    output_root.mkdir(parents=True, exist_ok=True)

    rag_fns = _load_retrieval_components()

    all_predictions = []
    all_ground_truths = []
    video_results = []
    query_details = []
    total_queries = 0
    total_queries_evaluated = 0

    for i, entry in enumerate(video_entries):
        vid = entry["vid"]
        video_path = entry["video_path"]
        output_dir = str(output_root / f"video_{i:03d}")
        queries = annotations_by_vid.get(vid, entry.get("queries", []))
        num_queries = len(queries)
        total_queries += num_queries

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(video_entries)}] {vid} | {num_queries} queries")
        print(f"{'='*60}")

        # ── Run or load pipeline ──
        checkpoint_path = Path(output_dir) / "checkpoint.json"
        rag_path = Path(output_dir) / "rag_embedding.json"

        if skip_pipeline and checkpoint_path.exists() and rag_path.exists():
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

        if not rag_path.exists():
            print(f"  [ERROR] No rag_embedding.json produced")
            continue

        rag_data = rag_fns["load_rag_embeddings"](str(rag_path))
        num_scenes = len(checkpoint.get("scenes", []))
        num_embeddings = len(rag_data.get("embeddings", []))
        print(f"  Scenes: {num_scenes} | Embeddings: {num_embeddings} | Queries: {num_queries}")

        if num_scenes == 0 or num_embeddings == 0:
            print(f"  [WARN] No scenes or embeddings, skipping")
            continue

        # ── Retrieve clips for each query ──
        video_predictions = []
        video_ground_truths = []

        for qi, q in enumerate(queries):
            query_text = q["query"]
            gt_windows = q.get("relevant_windows", [])

            if not gt_windows:
                continue

            clips = retrieve_clips_for_query(
                query_text, checkpoint, rag_data,
                top_k=top_k, rag_fns=rag_fns,
            )

            if merge_adjacent and len(clips) > 1:
                clips = merge_adjacent_clips(clips, gap_threshold=merge_gap_sec)

            video_predictions.append(clips)
            video_ground_truths.append(gt_windows)

            if clips:
                top_clip = clips[0]
                from metrics.soda_metric import temporal_iou
                best_iou = max(
                    temporal_iou(top_clip["start"], top_clip["end"], gt[0], gt[1])
                    for gt in gt_windows
                )
            else:
                best_iou = 0.0

            query_details.append({
                "vid": vid,
                "qid": q.get("qid"),
                "query": query_text,
                "gt_windows": gt_windows,
                "top1_start": clips[0]["start"] if clips else None,
                "top1_end": clips[0]["end"] if clips else None,
                "top1_score": clips[0]["score"] if clips else None,
                "top1_iou": best_iou,
                "num_clips_returned": len(clips),
            })
            total_queries_evaluated += 1

        all_predictions.extend(video_predictions)
        all_ground_truths.extend(video_ground_truths)

        if video_predictions:
            video_metrics = compute_moment_retrieval(video_predictions, video_ground_truths)
        else:
            video_metrics = {}

        video_results.append({
            "index": i,
            "vid": vid,
            "num_scenes": num_scenes,
            "num_queries": len(video_predictions),
            "metrics": video_metrics,
        })

        r1_05 = video_metrics.get("R@1_IoU=0.5", 0)
        miou = video_metrics.get("mIoU", 0)
        print(f"  R@1 IoU=0.5: {r1_05:.1f}% | mIoU: {miou:.3f} | queries: {len(video_predictions)}")

    # ── Aggregate metrics ──
    if not all_predictions:
        print("\n[ERROR] No predictions to evaluate")
        return None

    print(f"\n{'='*60}")
    print(f"Computing aggregate metrics on {total_queries_evaluated} queries from {len(video_results)} videos...")
    print(f"{'='*60}")

    aggregate = compute_moment_retrieval(all_predictions, all_ground_truths)

    results = {
        "dataset": "QVHighlights",
        "split": split,
        "num_videos": len(video_results),
        "num_queries": total_queries_evaluated,
        "top_k": top_k,
        "merge_adjacent": merge_adjacent,
        "merge_gap_sec": merge_gap_sec if merge_adjacent else None,
        "aggregate": aggregate,
        "per_video": video_results,
        "query_details_sample": query_details[:100],
    }

    return results


# ── Output ──

def print_summary(results):
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"  QVHighlights Benchmark Results")
    print(f"  {results['num_videos']} videos | {results['num_queries']} queries | top_k={results['top_k']}")
    if results.get("merge_adjacent"):
        print(f"  Scene merging: ON (gap={results['merge_gap_sec']}s)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*37}")

    for key in sorted(agg.keys()):
        if key.startswith("R@"):
            print(f"  {key:<25} {agg[key]:>9.1f}%")
    print(f"  {'mIoU':<25} {agg.get('mIoU', 0):>10.4f}")
    print(f"{'='*60}")


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merge_tag = "_merged" if results.get("merge_adjacent") else ""
    out_path = RESULTS_DIR / f"qvhighlights_results_{timestamp}{merge_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return out_path


def generate_report(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = results["aggregate"]

    lines = [
        "# QVHighlights Clip Retrieval Benchmark — Kairos\n",
        f"**Dataset:** QVHighlights (Lei et al., NeurIPS 2021)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Split:** {results.get('split', 'val')}",
        f"**Videos:** {results['num_videos']}",
        f"**Queries:** {results['num_queries']}",
        f"**Top-K:** {results['top_k']}",
        f"**Scene Merging:** {'Yes (gap=' + str(results.get('merge_gap_sec', 2.0)) + 's)' if results.get('merge_adjacent') else 'No'}\n",
        "---\n",
        "## Aggregate Metrics\n",
        "| Metric | Kairos (Zero-Shot) | Moment-DETR (Supervised) | QD-DETR (Supervised) | UniVTG (Supervised) |",
        "|--------|--------------------|--------------------------|----------------------|---------------------|",
    ]

    baselines = {
        "R@1_IoU=0.5": {"moment_detr": "52.89", "qd_detr": "62.40", "univtg": "58.86"},
        "R@1_IoU=0.7": {"moment_detr": "33.02", "qd_detr": "44.98", "univtg": "40.86"},
    }

    for key in ["R@1_IoU=0.3", "R@1_IoU=0.5", "R@1_IoU=0.7",
                "R@5_IoU=0.3", "R@5_IoU=0.5", "R@5_IoU=0.7"]:
        val = agg.get(key, 0)
        bl = baselines.get(key, {})
        lines.append(
            f"| {key} | **{val:.1f}%** | {bl.get('moment_detr', '—')} | "
            f"{bl.get('qd_detr', '—')} | {bl.get('univtg', '—')} |"
        )
    lines.append(f"| mIoU | **{agg.get('mIoU', 0):.4f}** | — | — | — |")

    lines += [
        "\n*Note: Kairos operates zero-shot (no training on QVHighlights). "
        "All baselines are trained on the QVHighlights training split with moment-level supervision.*\n",
        "---\n",
        "## Per-Video Breakdown\n",
        "| # | Video ID | Scenes | Queries | R@1 IoU=0.5 | mIoU |",
        "|---|----------|--------|---------|-------------|------|",
    ]

    for v in results["per_video"]:
        vm = v.get("metrics", {})
        lines.append(
            f"| {v['index']+1} | {v['vid'][:30]}... | {v['num_scenes']} | "
            f"{v['num_queries']} | {vm.get('R@1_IoU=0.5', 0):.1f}% | "
            f"{vm.get('mIoU', 0):.3f} |"
        )

    sample_details = results.get("query_details_sample", [])
    if sample_details:
        lines += [
            "\n---\n",
            "## Sample Query Results (first 20)\n",
            "| Query | GT Window | Top-1 Clip | IoU | Score |",
            "|-------|-----------|------------|-----|-------|",
        ]
        for qd in sample_details[:20]:
            gt_str = ", ".join(
                f"[{w[0]:.1f}-{w[1]:.1f}]" for w in qd.get("gt_windows", [])
            )
            t1s = qd.get("top1_start")
            t1e = qd.get("top1_end")
            clip_str = f"[{t1s:.1f}-{t1e:.1f}]" if t1s is not None else "—"
            query_short = qd["query"][:60] + ("..." if len(qd["query"]) > 60 else "")
            lines.append(
                f"| {query_short} | {gt_str} | {clip_str} | "
                f"{qd.get('top1_iou', 0):.3f} | {qd.get('top1_score', 0):.4f} |"
            )

    report_path = RESULTS_DIR / "qvhighlights_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {report_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Run QVHighlights clip retrieval benchmark for Kairos"
    )
    parser.add_argument("--max-videos", type=int, default=10,
                        help="Number of videos to benchmark (default: 10)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="QVHighlights split (default: val)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of clips to retrieve per query (default: 5)")
    parser.add_argument("--merge-adjacent", action="store_true",
                        help="Merge adjacent retrieved scenes into longer clips")
    parser.add_argument("--merge-gap-sec", type=float, default=2.0,
                        help="Max gap between scenes to merge (default: 2.0)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline, use cached checkpoint/embeddings only")
    parser.add_argument("--output-cache-name", default="qvhighlights_outputs",
                        help="Cache directory name under test/benchmarks/cache")
    args = parser.parse_args()

    print(f"[QVHighlights Benchmark] max_videos={args.max_videos}, "
          f"split={args.split}, top_k={args.top_k}, "
          f"merge_adjacent={args.merge_adjacent}")

    # ── Prepare data ──
    ann_path = download_qvhighlights_annotations(CACHE_DIR, split=args.split)
    if ann_path is None:
        print("[ERROR] Could not download annotations")
        sys.exit(1)

    annotations = load_annotations(ann_path)
    annotations_by_vid = group_queries_by_video(annotations)

    video_entries = prepare_qvhighlights(
        cache_dir=CACHE_DIR,
        video_dir=VIDEO_CACHE,
        split=args.split,
        max_videos=args.max_videos,
    )

    if not video_entries:
        print("[ERROR] No videos available")
        sys.exit(1)

    # ── Run benchmark ──
    results = run_benchmark(
        video_entries,
        annotations_by_vid,
        split=args.split,
        skip_pipeline=args.skip_pipeline,
        top_k=args.top_k,
        merge_adjacent=args.merge_adjacent,
        merge_gap_sec=args.merge_gap_sec,
        output_cache_name=args.output_cache_name,
    )

    if results is None:
        sys.exit(1)

    print_summary(results)
    save_results(results)
    generate_report(results)


if __name__ == "__main__":
    main()

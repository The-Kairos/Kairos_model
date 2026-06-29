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
BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent.parent   # test/benchmarks/
PROJECT_ROOT = BENCHMARKS_ROOT.parent.parent                      # repo root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCHMARKS_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ["KAIROS_LOW_MEM"] = "TRUE"
os.environ["KAIROS_BLIP_BATCH_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KAIROS_EMBEDDING_PROVIDER", "gemini")
os.environ.pop("MONGODB_URI", None)
os.environ.pop("MONGODB_DB_NAME", None)

os.chdir(PROJECT_ROOT)

from dataload.qvhighlights_loader import (
    prepare_qvhighlights,
    extract_val_videos_from_tarball,
)

CACHE_DIR = BENCHMARKS_ROOT / "cache" / "qvhighlights"
VIDEO_CACHE = CACHE_DIR / "qvh_videos"
RESULTS_DIR = Path(__file__).resolve().parent


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
        _embedding_values,
        _cosine_similarity,
        _to_vector,
    )
    return {
        "load_rag_embeddings": load_rag_embeddings,
        "embed_question": embed_question,
        "_embedding_values": _embedding_values,
        "_cosine_similarity": _cosine_similarity,
        "_to_vector": _to_vector,
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
        (clips, all_scores) where clips is a list of
        {"start": float, "end": float, "score": float, "scene_idx": int}
        and all_scores is a list of cosine similarity scores for every scene.
    """
    if rag_fns is None:
        rag_fns = _load_retrieval_components()

    scenes = checkpoint.get("scenes", [])
    num_scenes = len(scenes)
    if num_scenes == 0:
        return [], []

    all_contexts = rag_data.get("contexts", [])
    all_embeddings = rag_data.get("embeddings", [])

    scene_embeddings = all_embeddings[:num_scenes]
    if not scene_embeddings:
        return [], []

    rag_provider = rag_data.get("provider")
    rag_model = rag_data.get("model")
    query_embedding = rag_fns["embed_question"](
        query_text, provider=rag_provider, model=rag_model,
    )
    if isinstance(query_embedding, list) and len(query_embedding) > 0:
        query_embedding = query_embedding[0]
    query_vec = rag_fns["_to_vector"](
        rag_fns["_embedding_values"](query_embedding)
    )
    if query_vec is None:
        return [], []

    all_scores = []
    for i, emb in enumerate(scene_embeddings):
        emb_vec = rag_fns["_to_vector"](
            rag_fns["_embedding_values"](emb)
        )
        if emb_vec is None:
            all_scores.append(0.0)
            continue
        score = rag_fns["_cosine_similarity"](query_vec, emb_vec)
        all_scores.append(float(score))

    top_indices = np.argsort(all_scores)[-top_k:][::-1]

    clips = []
    for idx in top_indices:
        idx = int(idx)
        if idx >= num_scenes:
            continue
        scene = scenes[idx]
        clips.append({
            "start": scene.get("start_seconds", 0),
            "end": scene.get("end_seconds", 0),
            "score": all_scores[idx],
            "scene_idx": idx,
        })

    return clips, all_scores


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


def scenes_to_clip_saliency(scenes, all_scores, duration, clip_length=2.0):
    """Map scene-level cosine similarity scores to per-clip (2s) saliency scores.

    The official QVHighlights eval expects one saliency score per 2-second clip.
    We assign each 2s clip the max similarity score from any overlapping Kairos scene.
    """
    num_clips = int(duration / clip_length)
    if num_clips == 0:
        return []
    scores = [0.0] * num_clips
    for scene, score in zip(scenes, all_scores):
        start_sec = scene.get("start_seconds", 0)
        end_sec = scene.get("end_seconds", 0)
        start_clip = int(start_sec / clip_length)
        end_clip = min(int(end_sec / clip_length) + 1, num_clips)
        for i in range(start_clip, end_clip):
            scores[i] = max(scores[i], score)
    return scores


# ── Benchmark orchestration ──

def run_benchmark(video_entries, annotations_by_vid, split="val",
                  skip_pipeline=False, force_pipeline=False, top_k=5,
                  merge_adjacent=False, merge_gap_sec=2.0,
                  output_cache_name="qvhighlights_outputs",
                  mr_only=False, global_offset=0):
    """Run the full QVHighlights benchmark.

    Returns results dict with aggregate metrics and per-video/per-query details.
    """
    from metrics.qvhighlights.moment_retrieval_metric import compute_moment_retrieval
    from metrics.scenewalk.soda_metric import temporal_iou
    from metrics.qvhighlights.standalone_eval.eval import eval_submission

    output_root = CACHE_DIR / output_cache_name
    output_root.mkdir(parents=True, exist_ok=True)

    rag_fns = _load_retrieval_components()

    all_predictions = []
    all_ground_truths = []
    video_results = []
    query_details = []
    official_predictions = []
    official_ground_truth = []
    total_queries = 0
    total_queries_evaluated = 0

    for i, entry in enumerate(video_entries):
        vid = entry["vid"]
        video_path = entry["video_path"]
        global_idx = i + global_offset
        output_dir = str(output_root / f"video_{global_idx:04d}")
        queries = annotations_by_vid.get(vid, entry.get("queries", []))
        num_queries = len(queries)
        total_queries += num_queries

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(video_entries)}] {vid} | {num_queries} queries")
        print(f"{'='*60}")

        # ── Run or load pipeline ──
        checkpoint_path = Path(output_dir) / "checkpoint.json"
        rag_path = Path(output_dir) / "rag_embedding.json"

        if checkpoint_path.exists() and rag_path.exists() and not force_pipeline:
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
        scenes = checkpoint.get("scenes", [])
        num_scenes = len(scenes)
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
            qid = q.get("qid")
            duration = q.get("duration", 150)

            if not gt_windows:
                continue

            clips, all_scores = retrieve_clips_for_query(
                query_text, checkpoint, rag_data,
                top_k=top_k, rag_fns=rag_fns,
            )

            if merge_adjacent and len(clips) > 1:
                clips = merge_adjacent_clips(clips, gap_threshold=merge_gap_sec)

            video_predictions.append(clips)
            video_ground_truths.append(gt_windows)

            # Build official-format prediction
            pred_windows = [[c["start"], c["end"], c["score"]] for c in clips[:10]]
            pred_entry = {
                "qid": qid,
                "query": query_text,
                "vid": vid,
                "pred_relevant_windows": pred_windows,
            }
            gt_entry = {
                "qid": qid,
                "query": query_text,
                "vid": vid,
                "duration": duration,
                "relevant_windows": gt_windows,
            }

            if not mr_only:
                pred_entry["pred_saliency_scores"] = scenes_to_clip_saliency(
                    scenes, all_scores, duration,
                )
                gt_entry["relevant_clip_ids"] = q.get("relevant_clip_ids", [])
                gt_entry["saliency_scores"] = q.get("saliency_scores", [])

            official_predictions.append(pred_entry)
            official_ground_truth.append(gt_entry)

            if clips:
                top_clip = clips[0]
                best_iou = max(
                    temporal_iou(top_clip["start"], top_clip["end"], gt[0], gt[1])
                    for gt in gt_windows
                )
            else:
                best_iou = 0.0

            query_details.append({
                "vid": vid,
                "qid": qid,
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

    # ── Official Moment-DETR evaluation ──
    official_metrics = {}
    if official_predictions:
        print("Running official Moment-DETR evaluation...")
        try:
            official_results = eval_submission(
                official_predictions, official_ground_truth,
                verbose=True, match_number=True,
            )
            official_metrics = dict(official_results.get("brief", {}))
            print("\nOfficial Metrics:")
            for k, v in official_metrics.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  [WARN] Official eval failed: {e}")

    results = {
        "dataset": "QVHighlights",
        "split": split,
        "num_videos": len(video_results),
        "num_queries": total_queries_evaluated,
        "top_k": top_k,
        "merge_adjacent": merge_adjacent,
        "merge_gap_sec": merge_gap_sec if merge_adjacent else None,
        "aggregate": aggregate,
        "official_metrics": official_metrics,
        "per_video": video_results,
        "query_details_sample": query_details[:100],
    }

    return results, official_predictions


# ── Output ──

def print_summary(results):
    agg = results["aggregate"]
    official = results.get("official_metrics", {})
    print(f"\n{'='*60}")
    print(f"  QVHighlights Benchmark Results")
    print(f"  {results['num_videos']} videos | {results['num_queries']} queries | top_k={results['top_k']}")
    if results.get("merge_adjacent"):
        print(f"  Scene merging: ON (gap={results['merge_gap_sec']}s)")
    print(f"{'='*60}")

    if official:
        print(f"\n  === Official Metrics (Moment-DETR eval) ===")
        print(f"  {'Metric':<25} {'Kairos':>10} {'Moment-DETR':>12}")
        print(f"  {'-'*49}")
        baselines = {
            "MR-full-R1@0.5": 52.89, "MR-full-R1@0.7": 33.02,
            "MR-full-mAP": 30.73, "MR-full-mAP@0.5": 54.82,
            "MR-full-mAP@0.75": 29.40,
        }
        for key in ["MR-full-R1@0.5", "MR-full-R1@0.7",
                     "MR-full-mAP@0.5", "MR-full-mAP@0.75", "MR-full-mAP"]:
            if key in official:
                bl = baselines.get(key, "")
                bl_str = f"{bl}" if bl else "—"
                print(f"  {key:<25} {official[key]:>9.2f}% {bl_str:>12}")
        for key in sorted(official.keys()):
            if key.startswith("HL-"):
                print(f"  {key:<25} {official[key]:>9.2f}%")
        print()

    print(f"  === Internal Metrics ===")
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


def save_predictions_jsonl(predictions, results):
    """Save predictions in official Moment-DETR JSONL format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merge_tag = "_merged" if results.get("merge_adjacent") else ""
    out_path = RESULTS_DIR / f"qvhighlights_predictions_{timestamp}{merge_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"  Predictions JSONL saved to {out_path}")
    return out_path


def generate_report(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = results["aggregate"]
    official = results.get("official_metrics", {})

    lines = [
        "# QVHighlights Benchmark — Kairos (Official Metrics)\n",
        f"**Dataset:** QVHighlights (Lei et al., NeurIPS 2021)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Split:** {results.get('split', 'val')}",
        f"**Videos evaluated:** {results['num_videos']}",
        f"**Queries evaluated:** {results['num_queries']}",
        f"**Top-K:** {results['top_k']}",
        f"**Scene Merging:** {'Yes (gap=' + str(results.get('merge_gap_sec', 2.0)) + 's)' if results.get('merge_adjacent') else 'No'}",
        f"**Evaluation:** Official Moment-DETR standalone_eval + internal R@K/mIoU\n",
        "---\n",
    ]

    # ── Standard paper comparison table ──
    if official:
        lines += [
            "## Moment Retrieval (Official Metrics)\n",
            "| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |",
            "|--------|--------|--------|---------|----------|---------|",
            f"| Moment-DETR (supervised) | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |",
            f"| CLIP (zero-shot) | 16.88 | 5.19 | 18.11 | 7.00 | 7.67 |",
            f"| **Kairos (zero-shot)** | **{official.get('MR-full-R1@0.5', '—')}** "
            f"| **{official.get('MR-full-R1@0.7', '—')}** "
            f"| **{official.get('MR-full-mAP@0.5', '—')}** "
            f"| **{official.get('MR-full-mAP@0.75', '—')}** "
            f"| **{official.get('MR-full-mAP', '—')}** |",
            "",
        ]

        # MR by length bucket
        mr_short = official.get("MR-short-mAP")
        mr_middle = official.get("MR-middle-mAP")
        mr_long = official.get("MR-long-mAP")
        if mr_short is not None:
            lines += [
                "### Moment Retrieval by GT Window Length\n",
                "| Length Bucket | mAP Avg |",
                "|--------------|---------|",
                f"| Short (0-10s) | {mr_short} |",
                f"| Middle (10-30s) | {mr_middle} |",
                f"| Long (30-150s) | {mr_long} |",
                f"| Full (all) | {official.get('MR-full-mAP', '—')} |",
                "",
            ]

        # Highlight detection
        hl_keys = [k for k in official if k.startswith("HL-")]
        if hl_keys:
            lines += [
                "## Highlight Detection\n",
                "| Threshold | mAP | HIT@1 |",
                "|-----------|-----|-------|",
            ]
            for label in ["Fair", "Good", "VeryGood"]:
                mAP_key = f"HL-min-{label}-mAP"
                hit_key = f"HL-min-{label}-Hit1"
                mAP_val = official.get(mAP_key, "—")
                hit_val = official.get(hit_key, "—")
                lines.append(f"| {label} (>={2 if label == 'Fair' else 3 if label == 'Good' else 4}) | {mAP_val} | {hit_val} |")

            lines += [
                "",
                "| Method | HD mAP (VeryGood) | HD HIT@1 (VeryGood) |",
                "|--------|-------------------|---------------------|",
                f"| Moment-DETR (supervised) | 35.69 | 55.55 |",
                f"| QD-DETR (supervised) | 38.94 | 62.40 |",
                f"| UniVTG (supervised) | 38.20 | 60.96 |",
                f"| **Kairos (zero-shot)** | **{official.get('HL-min-VeryGood-mAP', '—')}** | **{official.get('HL-min-VeryGood-Hit1', '—')}** |",
                "",
            ]

    lines += [
        "\n*Kairos operates zero-shot (no training on QVHighlights). "
        "All baselines are trained on the QVHighlights training split with moment-level supervision.*\n",
        "---\n",
    ]

    # ── Internal metrics (backwards compat) ──
    lines += [
        "## Internal R@K IoU=T Metrics\n",
        "| Metric | Kairos |",
        "|--------|--------|",
    ]
    for key in ["R@1_IoU=0.3", "R@1_IoU=0.5", "R@1_IoU=0.7",
                "R@5_IoU=0.3", "R@5_IoU=0.5", "R@5_IoU=0.7"]:
        val = agg.get(key, 0)
        lines.append(f"| {key} | {val:.1f}% |")
    lines.append(f"| mIoU | {agg.get('mIoU', 0):.4f} |")

    # ── Per-video breakdown ──
    lines += [
        "\n---\n",
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

    # ── Sample query results ──
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


def merge_and_evaluate(prediction_files, split="val", mr_only=False):
    """Merge multiple prediction JSONL files and run official evaluation."""
    from metrics.qvhighlights.standalone_eval.eval import eval_submission

    all_preds = []
    seen_qids = set()
    for fpath in prediction_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pred = json.loads(line)
                qid = pred["qid"]
                if qid in seen_qids:
                    continue
                seen_qids.add(qid)
                all_preds.append(pred)

    print(f"[Merge] Loaded {len(all_preds)} unique predictions from "
          f"{len(prediction_files)} files")

    if split == "test":
        ann_path = CACHE_DIR / "highlight_test_with_gt.jsonl"
    else:
        ann_path = CACHE_DIR / f"highlight_{split}_release.jsonl"
    if not ann_path.exists():
        from dataload.qvhighlights_loader import download_qvhighlights_annotations
        ann_path = download_qvhighlights_annotations(CACHE_DIR, split=split)

    all_gt = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_gt.append(json.loads(line))

    gt_qids = set(g["qid"] for g in all_gt)
    pred_qids = set(p["qid"] for p in all_preds)
    missing = gt_qids - pred_qids
    extra = pred_qids - gt_qids

    if missing:
        print(f"  [WARN] {len(missing)} GT queries have no prediction")
    if extra:
        print(f"  [WARN] {len(extra)} predictions have no GT match")

    if mr_only:
        for pred in all_preds:
            pred.pop("pred_saliency_scores", None)

    match_number = len(missing) == 0 and len(extra) == 0

    print(f"[Merge] Running official evaluation (match_number={match_number})...")
    official_results = eval_submission(
        all_preds, all_gt,
        verbose=True, match_number=match_number,
    )

    official_metrics = dict(official_results.get("brief", {}))
    print(f"\n{'='*60}")
    print(f"  MERGED QVHighlights Results ({len(all_preds)} queries)")
    print(f"{'='*60}")

    baselines = {
        "MR-full-R1@0.5": 52.89, "MR-full-R1@0.7": 33.02,
        "MR-full-mAP": 30.73, "MR-full-mAP@0.5": 54.82,
        "MR-full-mAP@0.75": 29.40,
    }
    print(f"  {'Metric':<25} {'Kairos':>10} {'Moment-DETR':>12}")
    print(f"  {'-'*49}")
    for key in sorted(official_metrics.keys()):
        bl = baselines.get(key, "")
        bl_str = f"{bl}" if bl else ""
        print(f"  {key:<25} {official_metrics[key]:>9.2f}% {bl_str:>12}")
    print(f"{'='*60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    merged_pred_path = RESULTS_DIR / f"qvhighlights_predictions_MERGED_{timestamp}.jsonl"
    with open(merged_pred_path, "w", encoding="utf-8") as f:
        for pred in all_preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"  Merged predictions: {merged_pred_path}")

    merged_results = {
        "dataset": "QVHighlights",
        "split": split,
        "num_queries": len(all_preds),
        "num_gt_queries": len(all_gt),
        "source_files": [str(f) for f in prediction_files],
        "official_metrics": official_metrics,
        "match_number": match_number,
    }
    results_path = RESULTS_DIR / f"qvhighlights_results_MERGED_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    print(f"  Merged results: {results_path}")

    return official_metrics


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Run QVHighlights clip retrieval benchmark for Kairos"
    )
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit to N videos (default: all available)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="QVHighlights split (default: val)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of clips to retrieve per query (default: 10)")
    parser.add_argument("--merge-adjacent", action="store_true",
                        help="Merge adjacent retrieved scenes into longer clips")
    parser.add_argument("--merge-gap-sec", type=float, default=2.0,
                        help="Max gap between scenes to merge (default: 2.0)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline, use cached checkpoint/embeddings only")
    parser.add_argument("--force-pipeline", action="store_true",
                        help="Force re-run pipeline even if cached output exists")
    parser.add_argument("--output-cache-name", default="qvhighlights_outputs",
                        help="Cache directory name under test/benchmarks/cache")
    parser.add_argument("--mr-only", action="store_true",
                        help="Compute Moment Retrieval metrics only (skip Highlight Detection)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Process N videos per run (for batch processing)")
    parser.add_argument("--batch-offset", type=int, default=0,
                        help="Start from video index M (for batch processing)")
    parser.add_argument("--download-tarball", action="store_true",
                        help="Download and extract val videos from UNC tarball before processing")
    parser.add_argument("--merge-results", nargs="+", metavar="JSONL",
                        help="Merge prediction JSONL files and compute final metrics "
                             "(skips pipeline, just evaluates)")
    args = parser.parse_args()

    if args.merge_results:
        merge_and_evaluate(
            args.merge_results,
            split=args.split,
            mr_only=args.mr_only,
        )
        sys.exit(0)

    if args.batch_size is not None and args.max_videos is not None:
        needed = args.batch_offset + args.batch_size
        if args.max_videos < needed:
            print(f"[WARN] --max-videos {args.max_videos} too small for "
                  f"batch offset {args.batch_offset} + size {args.batch_size}. "
                  f"Overriding to None (all videos).")
            args.max_videos = None

    print(f"[QVHighlights Benchmark] max_videos={args.max_videos}, "
          f"split={args.split}, top_k={args.top_k}, "
          f"merge_adjacent={args.merge_adjacent}, mr_only={args.mr_only}")

    # ── Tarball download + extraction ──
    if args.download_tarball:
        tarball_path = CACHE_DIR / "qvhilights_videos.tar.gz"
        if not tarball_path.exists():
            print("[QVH] Tarball not found. Download it first with:")
            print(f"  wget -c '{VIDEOS_TARBALL_URL}' -O '{tarball_path}'")
            sys.exit(1)
        print(f"[QVH] Extracting val videos from tarball...")
        from dataload.qvhighlights_loader import VIDEOS_TARBALL_URL
        extract_val_videos_from_tarball(
            tarball_path, VIDEO_CACHE, CACHE_DIR, split=args.split,
        )

    # ── Prepare data ──
    video_entries = prepare_qvhighlights(
        cache_dir=CACHE_DIR,
        video_dir=VIDEO_CACHE,
        split=args.split,
        max_videos=args.max_videos,
    )

    if not video_entries:
        print("[ERROR] No videos available")
        sys.exit(1)

    # ── Batch slicing ──
    batch_global_offset = 0
    if args.batch_size is not None:
        total = len(video_entries)
        start = args.batch_offset
        end = min(start + args.batch_size, total)
        batch_global_offset = start
        video_entries = video_entries[start:end]
        print(f"[QVH] Batch: videos [{start}:{end}) of {total}")
        if not video_entries:
            print("[ERROR] Batch range is empty")
            sys.exit(1)

    annotations_by_vid = {entry["vid"]: entry["queries"] for entry in video_entries}

    # ── Run benchmark ──
    result = run_benchmark(
        video_entries,
        annotations_by_vid,
        split=args.split,
        skip_pipeline=args.skip_pipeline,
        force_pipeline=args.force_pipeline,
        top_k=args.top_k,
        merge_adjacent=args.merge_adjacent,
        merge_gap_sec=args.merge_gap_sec,
        output_cache_name=args.output_cache_name,
        mr_only=args.mr_only,
        global_offset=batch_global_offset,
    )

    if result is None:
        sys.exit(1)

    results, official_predictions = result

    print_summary(results)
    save_results(results)
    generate_report(results)
    if official_predictions:
        save_predictions_jsonl(official_predictions, results)


if __name__ == "__main__":
    main()

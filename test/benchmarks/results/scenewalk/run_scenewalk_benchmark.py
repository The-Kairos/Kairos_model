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
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Isolation: add project root and src/ to path ──
BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent.parent   # test/benchmarks/
PROJECT_ROOT = BENCHMARKS_ROOT.parent.parent                      # repo root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCHMARKS_ROOT))

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
    from metrics.scenewalk.soda_metric import compute_soda, rouge_l_f1
    from metrics.shared.bertscore_metric import compute_bertscore
    from metrics.shared.rouge_metric import compute_rouge_l
    return compute_soda, rouge_l_f1, compute_bertscore, compute_rouge_l


CACHE_DIR = BENCHMARKS_ROOT / "cache" / "scenewalk"
VIDEO_CACHE = CACHE_DIR / "videos"
RESULTS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CACHE_DIR / "scenewalk_manifest.json"
REWRITE_CACHE_DIR = CACHE_DIR / "aggregation_rewrites"


def prepare_scenewalk_videos(max_videos=5, min_duration_sec=120, min_segments=5,
                             manifest_path=MANIFEST_PATH,
                             exclude_video_ids=None,
                             max_download_candidates=None):
    """Find SceneWalk videos, download from YouTube. Returns list of video entries."""
    manifest_path = Path(manifest_path)
    exclude_video_ids = set(exclude_video_ids or [])

    if manifest_path.exists():
        videos = load_manifest(manifest_path)
        if exclude_video_ids:
            before = len(videos)
            videos = [v for v in videos if v.get("video_id") not in exclude_video_ids]
            if len(videos) != before:
                print(f"[SceneWalk] Filtered {before - len(videos)} excluded videos from cached manifest")
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
    candidate_count = max_download_candidates or max(
        max_videos + len(exclude_video_ids) + 5,
        max_videos * 50,
    )
    videos = group_segments_by_video(
        ds, max_videos=candidate_count * 3,
        min_segments=min_segments, min_duration_sec=min_duration_sec,
    )
    print(f"[SceneWalk] Attempting to download {len(videos)} candidate videos...")

    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i, video in enumerate(videos):
        if len(downloaded) >= max_videos:
            break
        vid_id = video["video_id"]
        if vid_id in exclude_video_ids:
            print(f"  [SKIP] Excluded development video: {vid_id}")
            continue
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

    save_manifest(downloaded, manifest_path)
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


def run_kairos_on_video(video_path, output_dir, redo_steps=None, redo_only=False):
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
            redo_steps=redo_steps,
            redo_only=redo_only,
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


def aggregate_segments_fixed_window(segments, window_sec=30.0, max_gap_sec=5.0):
    """Merge adjacent Kairos segments into fixed-duration evaluation windows.

    This is a reference-independent alignment step: it uses only Kairos
    timestamps, scene order, and fixed policy parameters.
    """
    if not segments:
        return []

    window_sec = max(1.0, float(window_sec))
    max_gap_sec = max(0.0, float(max_gap_sec))

    merged = []
    current = None

    for seg in sorted(segments, key=lambda s: (s.get("start", 0), s.get("end", 0))):
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))

        if current is None:
            current = {
                "start": start,
                "end": end,
                "texts": [text],
                "source_count": 1,
            }
            continue

        gap = max(0.0, start - current["end"])
        would_span = max(end, current["end"]) - current["start"]
        should_merge = gap <= max_gap_sec and would_span <= window_sec

        if should_merge:
            current["end"] = max(current["end"], end)
            current["texts"].append(text)
            current["source_count"] += 1
        else:
            merged.append({
                "start": current["start"],
                "end": current["end"],
                "text": " ".join(current["texts"]),
                "source_count": current["source_count"],
                "source_texts": current["texts"],
            })
            current = {
                "start": start,
                "end": end,
                "texts": [text],
                "source_count": 1,
            }

    if current is not None:
        merged.append({
            "start": current["start"],
            "end": current["end"],
            "text": " ".join(current["texts"]),
            "source_count": current["source_count"],
            "source_texts": current["texts"],
        })

    return merged


def apply_prediction_aggregation(segments, policy="none", window_sec=30.0, max_gap_sec=5.0):
    if policy == "none":
        return segments
    if policy == "fixed_window":
        return aggregate_segments_fixed_window(
            segments,
            window_sec=window_sec,
            max_gap_sec=max_gap_sec,
        )
    raise ValueError(f"Unknown aggregation policy: {policy}")


def _format_group_for_rewrite(segment):
    source_texts = segment.get("source_texts") or [segment.get("text", "")]
    lines = [
        f"Segment time: {segment.get('start', 0):.2f}s to {segment.get('end', 0):.2f}s",
        "",
    ]
    for idx, text in enumerate(source_texts, start=1):
        clean = " ".join((text or "").split())
        lines.append(f"Scene {idx}: {clean}")
    return "\n".join(lines)


def _rewrite_cache_path(video_id, prompt_path, output_cache_name, aggregation_policy,
                        window_sec, max_gap_sec):
    key = "|".join([
        str(video_id),
        str(prompt_path),
        str(output_cache_name),
        str(aggregation_policy),
        f"{float(window_sec):.3f}",
        f"{float(max_gap_sec):.3f}",
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return REWRITE_CACHE_DIR / f"{video_id}_{digest}.json"


def _call_rewrite_llm(client, deployment, prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You rewrite grouped video scene descriptions into concise visual segment captions.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=700,
        temperature=0.2,
        top_p=1.0,
        model=deployment,
        timeout=60.0,
    )
    return response.choices[0].message.content.strip()


def rewrite_aggregated_segments(segments, video_id, video_name, prompt_path,
                                output_cache_name, aggregation_policy,
                                window_sec, max_gap_sec, max_workers=6):
    """Rewrite aggregated segments as benchmark-only segment captions.

    Uses only Kairos prediction text/timestamps and caches outputs by policy.
    """
    if not segments:
        return []

    prompt_path = Path(prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Aggregation rewrite prompt not found: {prompt_path}")

    REWRITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _rewrite_cache_path(
        video_id,
        prompt_path,
        output_cache_name,
        aggregation_policy,
        window_sec,
        max_gap_sec,
    )

    cached = None
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_segments = cached.get("segments", [])
        if len(cached_segments) == len(segments):
            return cached_segments

    from main import deployment, get_llm_client

    client = get_llm_client()
    template = prompt_path.read_text(encoding="utf-8")
    rewritten = [None] * len(segments)

    def _rewrite_one(idx, segment):
        group_text = _format_group_for_rewrite(segment)
        prompt = template.replace("{{VIDEO_NAME}}", video_name or str(video_id))
        prompt = prompt.replace("{{GROUP_TEXT}}", group_text)
        text = _call_rewrite_llm(client, deployment, prompt)
        return idx, {
            "start": segment["start"],
            "end": segment["end"],
            "text": text,
            "source_count": segment.get("source_count", 1),
        }

    max_workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_rewrite_one, idx, segment)
            for idx, segment in enumerate(segments)
        ]
        for future in as_completed(futures):
            idx, rewritten_segment = future.result()
            rewritten[idx] = rewritten_segment

    payload = {
        "video_id": video_id,
        "video_name": video_name,
        "prompt_path": str(prompt_path),
        "output_cache_name": output_cache_name,
        "aggregation_policy": aggregation_policy,
        "window_sec": window_sec,
        "max_gap_sec": max_gap_sec,
        "segments": rewritten,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return rewritten


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


def run_benchmark(videos, skip_pipeline=False, redo_steps=None, redo_only=False,
                  output_cache_name="scenewalk_outputs",
                  aggregation_policy="none", aggregation_window_sec=30.0,
                  aggregation_max_gap_sec=5.0, rewrite_aggregates=False,
                  aggregation_rewrite_prompt=None, rewrite_max_workers=6,
                  manifest_name=None):
    """Run full benchmark: pipeline + metrics."""
    all_pred_segments = []
    all_ref_segments = []
    video_results = []

    sw_output_root = CACHE_DIR / output_cache_name
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
            checkpoint = run_kairos_on_video(
                video_path,
                output_dir,
                redo_steps=redo_steps,
                redo_only=redo_only,
            )
            elapsed = time.perf_counter() - t0
            print(f"  Pipeline completed in {elapsed:.1f}s")
            if checkpoint is None:
                print(f"  [ERROR] Pipeline produced no checkpoint")
                continue

        raw_pred_segs = extract_scene_segments(checkpoint)
        pred_segs = apply_prediction_aggregation(
            raw_pred_segs,
            policy=aggregation_policy,
            window_sec=aggregation_window_sec,
            max_gap_sec=aggregation_max_gap_sec,
        )
        if rewrite_aggregates:
            pred_segs = rewrite_aggregated_segments(
                pred_segs,
                video_id=vid_id,
                video_name=Path(video_path).name if video_path else vid_id,
                prompt_path=aggregation_rewrite_prompt,
                output_cache_name=output_cache_name,
                aggregation_policy=aggregation_policy,
                window_sec=aggregation_window_sec,
                max_gap_sec=aggregation_max_gap_sec,
                max_workers=rewrite_max_workers,
            )
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
            "num_raw_kairos_scenes": len(raw_pred_segs),
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
        "prediction_source": output_cache_name,
        "manifest_name": manifest_name,
        "aggregation": {
            "policy": aggregation_policy,
            "window_sec": aggregation_window_sec,
            "max_gap_sec": aggregation_max_gap_sec,
            "rewrite": rewrite_aggregates,
            "rewrite_prompt": aggregation_rewrite_prompt,
        },
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


def generate_reports(results, manifest_path=MANIFEST_PATH):
    """Generate MD benchmark report and comparison files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print("  [WARN] No manifest found, skipping report generation")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    agg = results["aggregate"]
    sw_output_root = CACHE_DIR / results.get("prediction_source", "scenewalk_outputs")
    aggregation = results.get("aggregation", {"policy": "none"})

    # ── Metric report ──
    lines = [
        "# SceneWalk Benchmark Report — Kairos\n",
        f"**Dataset:** SceneWalk (IVLLab/SceneWalk)",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Videos:** {results['num_videos']}\n",
        f"**Prediction source:** `{results.get('prediction_source', 'scenewalk_outputs')}`",
        f"**Manifest:** `{results.get('manifest_name') or 'scenewalk_manifest.json'}`",
        f"**Aggregation:** `{aggregation.get('policy', 'none')}`\n",
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
              "| # | Video ID | Duration | Kairos Scenes | Raw Scenes | GT Segments | Matched | SODA F1 |",
              "|---|----------|----------|---------------|------------|-------------|---------|---------|"]
    for v in results["per_video"]:
        lines.append(f"| {v['index']+1} | {v['video_id']} | {v['duration_min']:.0f} min | "
                     f"{v['num_kairos_scenes']} | {v.get('num_raw_kairos_scenes', v['num_kairos_scenes'])} | "
                     f"{v['num_gt_segments']} | "
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
            kairos_scenes = [
                {
                    "start_sec": seg["start"],
                    "end_sec": seg["end"],
                    "kairos_description": seg["text"],
                    "source_count": seg.get("source_count", 1),
                }
                for seg in apply_prediction_aggregation(
                    [
                        {
                            "start": ks["start_sec"],
                            "end": ks["end_sec"],
                            "text": ks["kairos_description"],
                        }
                        for ks in kairos_scenes
                    ],
                    policy=aggregation.get("policy", "none"),
                    window_sec=aggregation.get("window_sec", 30.0),
                    max_gap_sec=aggregation.get("max_gap_sec", 5.0),
                )
            ]
            if aggregation.get("rewrite"):
                kairos_scenes = [
                    {
                        "start_sec": seg["start"],
                        "end_sec": seg["end"],
                        "kairos_description": seg["text"],
                        "source_count": seg.get("source_count", 1),
                    }
                    for seg in rewrite_aggregated_segments(
                        [
                            {
                                "start": ks["start_sec"],
                                "end": ks["end_sec"],
                                "text": ks["kairos_description"],
                                "source_count": ks.get("source_count", 1),
                            }
                            for ks in kairos_scenes
                        ],
                        video_id=vid_id,
                        video_name=vid_id,
                        prompt_path=aggregation.get("rewrite_prompt"),
                        output_cache_name=results.get("prediction_source", "scenewalk_outputs"),
                        aggregation_policy=aggregation.get("policy", "none"),
                        window_sec=aggregation.get("window_sec", 30.0),
                        max_gap_sec=aggregation.get("max_gap_sec", 5.0),
                    )
                ]

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
            f"- **Kairos segments evaluated:** {c['num_kairos_scenes']}",
            f"- **Aggregation:** `{aggregation.get('policy', 'none')}`\n",
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
    parser.add_argument("--redo", action="append", default=[],
                        choices=["scenes", "frame_captions", "yolo", "audio_natural",
                                 "audio_speech", "llm", "narrative", "synopsis", "rag"],
                        help="Redo a pipeline step before benchmarking. Repeatable.")
    parser.add_argument("--redo-only", action="store_true",
                        help="Redo only the selected --redo steps, without dependents.")
    parser.add_argument("--output-cache-name", default="scenewalk_outputs",
                        help="Cache output directory name under test/benchmarks/cache.")
    parser.add_argument("--manifest-name", default="scenewalk_manifest.json",
                        help="Manifest file name under test/benchmarks/cache.")
    parser.add_argument("--exclude-video-id", action="append", default=[],
                        help="SceneWalk video ID to exclude from selection. Repeatable.")
    parser.add_argument("--max-download-candidates", type=int, default=None,
                        help="Maximum candidate videos to try downloading before failing.")
    parser.add_argument("--aggregate-predictions", default="none",
                        choices=["none", "fixed_window"],
                        help="Reference-independent prediction aggregation policy.")
    parser.add_argument("--aggregation-window-sec", type=float, default=30.0,
                        help="Target fixed window in seconds for fixed_window aggregation.")
    parser.add_argument("--aggregation-max-gap-sec", type=float, default=5.0,
                        help="Maximum gap between adjacent scenes to merge.")
    parser.add_argument("--rewrite-aggregates", action="store_true",
                        help="Rewrite aggregated predictions into segment-level captions.")
    parser.add_argument("--aggregation-rewrite-prompt",
                        default="prompts/benchmark_versions/aggregate_scene_segments_v1.txt",
                        help="Prompt file for --rewrite-aggregates.")
    parser.add_argument("--rewrite-max-workers", type=int, default=6,
                        help="Parallel workers for aggregate rewrite LLM calls.")
    args = parser.parse_args()

    print(f"[SceneWalk Benchmark] Starting with max_videos={args.max_videos}, "
          f"min_duration={args.min_duration}s, min_segments={args.min_segments}")

    videos = prepare_scenewalk_videos(
        max_videos=args.max_videos,
        min_duration_sec=args.min_duration,
        min_segments=args.min_segments,
        manifest_path=CACHE_DIR / args.manifest_name,
        exclude_video_ids=args.exclude_video_id,
        max_download_candidates=args.max_download_candidates,
    )
    if not videos:
        print("[ERROR] No SceneWalk videos available")
        sys.exit(1)

    results = run_benchmark(
        videos,
        skip_pipeline=args.skip_pipeline,
        redo_steps=args.redo,
        redo_only=args.redo_only,
        output_cache_name=args.output_cache_name,
        aggregation_policy=args.aggregate_predictions,
        aggregation_window_sec=args.aggregation_window_sec,
        aggregation_max_gap_sec=args.aggregation_max_gap_sec,
        rewrite_aggregates=args.rewrite_aggregates,
        aggregation_rewrite_prompt=args.aggregation_rewrite_prompt,
        rewrite_max_workers=args.rewrite_max_workers,
        manifest_name=args.manifest_name,
    )
    if results is None:
        sys.exit(1)

    print_summary(results)
    save_results(results)
    generate_reports(results, manifest_path=CACHE_DIR / args.manifest_name)


if __name__ == "__main__":
    main()

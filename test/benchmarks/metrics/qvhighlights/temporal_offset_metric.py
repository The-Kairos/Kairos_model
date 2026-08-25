"""
Temporal offset metric for moment retrieval evaluation.

Measures how far off (in seconds) predicted clip boundaries are from ground truth,
providing diagnostic insight beyond binary IoU thresholds.

Metrics computed per query (top-1 prediction vs best-matching GT window):
  - Start Offset:  pred_start - gt_start   (+X = starts too late)
  - End Offset:    pred_end - gt_end       (+X = ends too late)
  - Center Offset: center(pred) - center(gt)
  - ABE:           (|start_offset| + |end_offset|) / 2

When a query has multiple GT windows, the one with highest IoU against the top-1
prediction is used (same matching strategy as R@1).
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def _temporal_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0.0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return inter / union


def _best_gt_for_pred(pred_start, pred_end, gt_windows):
    """Return the GT window with highest IoU against the prediction."""
    best_iou = -1.0
    best_gt = gt_windows[0]
    for gt in gt_windows:
        iou = _temporal_iou(pred_start, pred_end, gt[0], gt[1])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    return best_gt, best_iou


def compute_temporal_offsets(
    predictions: List[Dict],
    ground_truths: Dict[int, List[List[float]]],
) -> Dict:
    """Compute temporal offset metrics across all queries.

    Args:
        predictions: List of dicts with keys: qid, pred_relevant_windows
                     (each window is [start, end, score], sorted by score desc)
        ground_truths: Dict mapping qid -> list of [start, end] windows

    Returns:
        Dict with per-query offsets and aggregate statistics.
    """
    per_query = []

    for pred in predictions:
        qid = pred["qid"]
        gt_windows = ground_truths.get(qid)
        if not gt_windows:
            continue

        windows = pred.get("pred_relevant_windows", [])
        if not windows:
            continue

        top1 = windows[0]
        pred_start, pred_end = top1[0], top1[1]
        pred_score = top1[2] if len(top1) > 2 else 0.0

        best_gt, iou = _best_gt_for_pred(pred_start, pred_end, gt_windows)
        gt_start, gt_end = best_gt[0], best_gt[1]

        start_offset = pred_start - gt_start
        end_offset = pred_end - gt_end
        pred_center = (pred_start + pred_end) / 2.0
        gt_center = (gt_start + gt_end) / 2.0
        center_offset = pred_center - gt_center
        abe = (abs(start_offset) + abs(end_offset)) / 2.0

        gt_duration = gt_end - gt_start
        pred_duration = pred_end - pred_start
        duration_ratio = pred_duration / gt_duration if gt_duration > 0 else 0.0

        length_bucket = "short"
        if gt_duration > 30:
            length_bucket = "long"
        elif gt_duration > 10:
            length_bucket = "middle"

        per_query.append({
            "qid": qid,
            "query": pred.get("query", ""),
            "pred_window": [pred_start, pred_end],
            "gt_window": [gt_start, gt_end],
            "iou": iou,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "center_offset": center_offset,
            "abe": abe,
            "gt_duration": gt_duration,
            "pred_duration": pred_duration,
            "duration_ratio": duration_ratio,
            "length_bucket": length_bucket,
        })

    if not per_query:
        return {"num_queries": 0, "per_query": [], "aggregate": {}, "by_bucket": {}}

    agg = _compute_stats(per_query)
    by_bucket = {}
    for bucket in ("short", "middle", "long"):
        subset = [q for q in per_query if q["length_bucket"] == bucket]
        if subset:
            by_bucket[bucket] = _compute_stats(subset)

    return {
        "num_queries": len(per_query),
        "aggregate": agg,
        "by_bucket": by_bucket,
        "per_query": per_query,
    }


def _compute_stats(queries: List[Dict]) -> Dict:
    """Compute aggregate statistics over a list of per-query offset dicts."""
    n = len(queries)
    fields = ["start_offset", "end_offset", "center_offset", "abe"]
    stats = {"num_queries": n}

    for f in fields:
        vals = [q[f] for q in queries]
        abs_vals = [abs(v) for v in vals]
        stats[f] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if n > 1 else 0.0,
            "abs_mean": statistics.mean(abs_vals),
            "abs_median": statistics.median(abs_vals),
            "min": min(vals),
            "max": max(vals),
        }

    ious = [q["iou"] for q in queries]
    stats["iou"] = {
        "mean": statistics.mean(ious),
        "median": statistics.median(ious),
    }

    ratios = [q["duration_ratio"] for q in queries]
    stats["duration_ratio"] = {
        "mean": statistics.mean(ratios),
        "median": statistics.median(ratios),
    }

    zero_iou = sum(1 for q in queries if q["iou"] == 0.0)
    stats["zero_iou_count"] = zero_iou
    stats["zero_iou_pct"] = zero_iou / n * 100.0

    return stats


def load_predictions(jsonl_path: str) -> List[Dict]:
    preds = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def load_ground_truths(jsonl_path: str) -> Dict[int, List[List[float]]]:
    gt = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry.get("qid")
            windows = entry.get("relevant_windows", [])
            if qid is not None and windows:
                gt[qid] = windows
    return gt


def format_report(results: Dict) -> str:
    """Format temporal offset results as a readable report."""
    lines = []
    lines.append("# Temporal Offset Analysis Report")
    lines.append("")
    lines.append(f"**Queries analyzed:** {results['num_queries']}")
    lines.append("")

    agg = results["aggregate"]
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Metric | Mean | |Mean| | Median | Std Dev | Min | Max |")
    lines.append("|--------|------:|-------:|-------:|--------:|-----:|-----:|")

    for field, label in [
        ("start_offset", "Start Offset (s)"),
        ("end_offset", "End Offset (s)"),
        ("center_offset", "Center Offset (s)"),
        ("abe", "ABE (s)"),
    ]:
        s = agg[field]
        lines.append(
            f"| {label} | {s['mean']:+.2f} | {s['abs_mean']:.2f} "
            f"| {s['median']:+.2f} | {s['stdev']:.2f} "
            f"| {s['min']:+.2f} | {s['max']:+.2f} |"
        )

    lines.append("")
    lines.append(f"- **Mean IoU:** {agg['iou']['mean']:.4f}")
    lines.append(f"- **Median IoU:** {agg['iou']['median']:.4f}")
    lines.append(f"- **Mean duration ratio (pred/gt):** {agg['duration_ratio']['mean']:.2f}")
    lines.append(f"- **Complete misses (IoU=0):** {agg['zero_iou_count']} ({agg['zero_iou_pct']:.1f}%)")
    lines.append("")

    lines.append("## Breakdown by GT Moment Length")
    lines.append("")
    for bucket, label in [("short", "Short (0-10s)"), ("middle", "Middle (10-30s)"), ("long", "Long (30-150s)")]:
        bs = results["by_bucket"].get(bucket)
        if not bs:
            continue
        lines.append(f"### {label} — {bs['num_queries']} queries")
        lines.append("")
        lines.append("| Metric | Mean | |Mean| | Median |")
        lines.append("|--------|------:|-------:|-------:|")
        for field, fl in [
            ("start_offset", "Start Offset"),
            ("end_offset", "End Offset"),
            ("center_offset", "Center Offset"),
            ("abe", "ABE"),
        ]:
            s = bs[field]
            lines.append(f"| {fl} | {s['mean']:+.2f} | {s['abs_mean']:.2f} | {s['median']:+.2f} |")
        lines.append(f"- IoU mean: {bs['iou']['mean']:.4f}, Duration ratio mean: {bs['duration_ratio']['mean']:.2f}")
        lines.append(f"- Complete misses: {bs['zero_iou_count']} ({bs['zero_iou_pct']:.1f}%)")
        lines.append("")

    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("| Pattern | Diagnosis |")
    lines.append("|---------|-----------|")
    lines.append("| Small |Center Offset|, large ABE | Right location, wrong boundaries (scene granularity) |")
    lines.append("| Large |Center Offset| | Wrong part of the video (retrieval error) |")
    lines.append("| Start Offset ~ 0, End Offset << 0 | Starts right but ends too early (scenes too short) |")
    lines.append("| Duration ratio << 1.0 | Predicted clips much narrower than GT (granularity mismatch) |")
    lines.append("| Duration ratio >> 1.0 | Predicted clips much wider than GT (over-merging) |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Compute temporal offset metrics")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file")
    parser.add_argument("--ground-truth", default=None, help="GT annotations JSONL (auto-downloads if not given)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--output", default=None, help="Output report path (default: stdout)")
    parser.add_argument("--output-json", default=None, help="Output raw JSON results")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}")
        sys.exit(1)

    if args.ground_truth:
        gt_path = Path(args.ground_truth)
    else:
        cache_dir = Path(__file__).resolve().parents[2] / "cache" / "qvhighlights"
        from dataload.qvhighlights_loader import download_qvhighlights_annotations
        gt_path = download_qvhighlights_annotations(cache_dir, split=args.split)
        if gt_path is None:
            print("ERROR: could not download GT annotations")
            sys.exit(1)

    print(f"Loading predictions from {pred_path}")
    predictions = load_predictions(str(pred_path))
    print(f"  {len(predictions)} predictions loaded")

    print(f"Loading ground truth from {gt_path}")
    ground_truths = load_ground_truths(str(gt_path))
    print(f"  {len(ground_truths)} GT entries loaded")

    results = compute_temporal_offsets(predictions, ground_truths)
    report = format_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Report saved to {out}")
    else:
        print()
        print(report)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        json_results = {k: v for k, v in results.items() if k != "per_query"}
        json_results["per_query_sample"] = results["per_query"][:5]
        out_json.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
        print(f"JSON results saved to {out_json}")

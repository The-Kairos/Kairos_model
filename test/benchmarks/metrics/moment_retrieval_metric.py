"""
Moment retrieval evaluation metrics for clip retrieval benchmarking.

Computes standard temporal grounding metrics:
  - R@K at IoU thresholds (0.3, 0.5, 0.7)
  - mIoU (mean IoU of top-1 predictions)

Reference: Metrics used across QVHighlights (NeurIPS 2021), Charades-STA (ICCV 2017),
ActivityNet Captions (ICCV 2017), and all standard temporal grounding benchmarks.
"""
from typing import List, Dict

from metrics.soda_metric import temporal_iou


def _best_iou_against_gt(clip_start, clip_end, gt_windows):
    """Compute the best IoU of a single clip against all ground-truth windows."""
    best = 0.0
    for gt in gt_windows:
        iou = temporal_iou(clip_start, clip_end, gt[0], gt[1])
        if iou > best:
            best = iou
    return best


def compute_moment_retrieval(
    predictions: List[List[Dict]],
    ground_truths: List[List[List[float]]],
    ks: List[int] = None,
    iou_thresholds: List[float] = None,
) -> Dict:
    """Compute moment retrieval metrics for a set of queries.

    Args:
        predictions: Per-query list of predicted clips, each sorted by score
                     descending. Each clip: {"start": float, "end": float, "score": float}
        ground_truths: Per-query list of ground-truth windows [[start, end], ...]
        ks: Values of K for R@K (default [1, 5])
        iou_thresholds: IoU thresholds (default [0.3, 0.5, 0.7])

    Returns:
        Dict with R@K_IoU=T for each (K, T) combo, plus mIoU.
    """
    if ks is None:
        ks = [1, 5]
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    n = len(predictions)
    if n == 0:
        result = {"mIoU": 0.0, "num_queries": 0}
        for k in ks:
            for t in iou_thresholds:
                result[f"R@{k}_IoU={t}"] = 0.0
        return result

    recall_hits = {}
    for k in ks:
        for t in iou_thresholds:
            recall_hits[(k, t)] = 0

    total_top1_iou = 0.0

    for pred_clips, gt_windows in zip(predictions, ground_truths):
        if not gt_windows:
            continue

        if not pred_clips:
            continue

        top1_iou = _best_iou_against_gt(
            pred_clips[0]["start"], pred_clips[0]["end"], gt_windows
        )
        total_top1_iou += top1_iou

        for k in ks:
            best_iou_in_topk = 0.0
            for clip in pred_clips[:k]:
                iou = _best_iou_against_gt(clip["start"], clip["end"], gt_windows)
                if iou > best_iou_in_topk:
                    best_iou_in_topk = iou
            for t in iou_thresholds:
                if best_iou_in_topk >= t:
                    recall_hits[(k, t)] += 1

    result = {
        "num_queries": n,
        "mIoU": total_top1_iou / n,
    }
    for k in ks:
        for t in iou_thresholds:
            result[f"R@{k}_IoU={t}"] = recall_hits[(k, t)] / n * 100.0

    return result

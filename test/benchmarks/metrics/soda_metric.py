"""
SODA metric for dense video captioning evaluation. Pure Python.

Implements Story Oriented Dense cAptioning (SODA) evaluation:
  1. Temporal IoU matching between predicted and reference segments
  2. Order-preserving optimal assignment via dynamic programming
  3. Caption quality scoring via a pluggable scorer (ROUGE-L or BERTScore)
  4. Precision / Recall / F1

Reference: Fujita et al., "SODA: Story Oriented Dense Video Captioning
Evaluation Framework", ECCV 2020.
"""
import re
from typing import Callable, List, Dict, Optional, Tuple


def temporal_iou(p_start, p_end, r_start, r_end):
    """Intersection-over-Union of two time intervals."""
    inter_start = max(p_start, r_start)
    inter_end = min(p_end, r_end)
    intersection = max(0.0, inter_end - inter_start)
    union = max(0.0, max(p_end, r_end) - min(p_start, r_start))
    if union <= 0:
        return 0.0
    return intersection / union


def _lcs_length(a, b):
    """Longest common subsequence length (for ROUGE-L)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_f1(pred_text, ref_text):
    """Single-pair ROUGE-L F1 (pure Python, used as default SODA scorer)."""
    pred_tokens = re.findall(r"\w+", pred_text.lower())
    ref_tokens = re.findall(r"\w+", ref_text.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def optimal_matching_dp(pred_segments, ref_segments, score_matrix, iou_threshold=0.3):
    """Order-preserving bipartite matching via dynamic programming.

    Finds the assignment of predictions to references that:
      - Preserves temporal order (if pred i matches ref j, then pred i+1
        can only match ref k > j)
      - Maximizes total caption quality score
      - Only considers pairs with temporal IoU >= threshold

    Returns list of (pred_idx, ref_idx, score) tuples.
    """
    n_pred = len(pred_segments)
    n_ref = len(ref_segments)

    iou_matrix = [[0.0] * n_ref for _ in range(n_pred)]
    for i, ps in enumerate(pred_segments):
        for j, rs in enumerate(ref_segments):
            iou_matrix[i][j] = temporal_iou(ps["start"], ps["end"],
                                            rs["start"], rs["end"])

    # dp[i][j] = best total score using pred[0..i-1] matched to ref[0..j-1]
    dp = [[0.0] * (n_ref + 1) for _ in range(n_pred + 1)]
    choice = [[None] * (n_ref + 1) for _ in range(n_pred + 1)]

    for i in range(1, n_pred + 1):
        for j in range(1, n_ref + 1):
            # Option 1: skip this ref
            if dp[i][j - 1] >= dp[i][j]:
                dp[i][j] = dp[i][j - 1]
                choice[i][j] = ("skip_ref", i, j - 1)

            # Option 2: skip this pred
            if dp[i - 1][j] >= dp[i][j]:
                dp[i][j] = dp[i - 1][j]
                choice[i][j] = ("skip_pred", i - 1, j)

            # Option 3: match pred i-1 with ref j-1 (if IoU sufficient)
            if iou_matrix[i - 1][j - 1] >= iou_threshold:
                s = score_matrix[i - 1][j - 1]
                candidate = dp[i - 1][j - 1] + s
                if candidate > dp[i][j]:
                    dp[i][j] = candidate
                    choice[i][j] = ("match", i - 1, j - 1)

    # Backtrack
    matches = []
    ci, cj = n_pred, n_ref
    while ci > 0 and cj > 0:
        ch = choice[ci][cj]
        if ch is None:
            break
        action = ch[0]
        if action == "match":
            pi, rj = ch[1], ch[2]
            matches.append((pi, rj, score_matrix[pi][rj]))
            ci, cj = pi, rj
        elif action == "skip_ref":
            ci, cj = ch[1], ch[2]
        elif action == "skip_pred":
            ci, cj = ch[1], ch[2]
        else:
            break

    matches.reverse()
    return matches


def compute_soda(pred_segments, ref_segments, scorer_fn=None, iou_threshold=0.3):
    """Compute SODA metric for a single video.

    Args:
        pred_segments: list of {"start": float, "end": float, "text": str}
        ref_segments: same format
        scorer_fn: (pred_text, ref_text) -> float. Defaults to rouge_l_f1.
        iou_threshold: minimum temporal IoU for a match candidate

    Returns:
        dict with precision, recall, f1, num_matched, matches, etc.
    """
    if scorer_fn is None:
        scorer_fn = rouge_l_f1

    if not pred_segments or not ref_segments:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "num_pred": len(pred_segments), "num_ref": len(ref_segments),
            "num_matched": 0, "matches": [],
        }

    n_pred = len(pred_segments)
    n_ref = len(ref_segments)

    score_matrix = [[0.0] * n_ref for _ in range(n_pred)]
    for i, ps in enumerate(pred_segments):
        for j, rs in enumerate(ref_segments):
            iou = temporal_iou(ps["start"], ps["end"], rs["start"], rs["end"])
            if iou >= iou_threshold:
                score_matrix[i][j] = scorer_fn(ps["text"], rs["text"])

    matches = optimal_matching_dp(pred_segments, ref_segments,
                                  score_matrix, iou_threshold)

    matched_scores = [s for _, _, s in matches]
    sum_scores = sum(matched_scores)

    precision = sum_scores / n_pred if n_pred > 0 else 0.0
    recall = sum_scores / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    match_details = []
    for pi, rj, sc in matches:
        match_details.append({
            "pred_idx": pi,
            "ref_idx": rj,
            "score": sc,
            "pred_text": pred_segments[pi]["text"][:100],
            "ref_text": ref_segments[rj]["text"][:100],
            "iou": temporal_iou(pred_segments[pi]["start"], pred_segments[pi]["end"],
                                ref_segments[rj]["start"], ref_segments[rj]["end"]),
        })

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred": n_pred,
        "num_ref": n_ref,
        "num_matched": len(matches),
        "matches": match_details,
    }


def compute_soda_batch(all_preds, all_refs, scorer_fn=None, iou_threshold=0.3):
    """Compute SODA across multiple videos.

    Args:
        all_preds: list of list-of-segments (one per video)
        all_refs: list of list-of-segments (one per video)

    Returns:
        dict with per-video scores and aggregates
    """
    per_video = []
    for preds, refs in zip(all_preds, all_refs):
        result = compute_soda(preds, refs, scorer_fn=scorer_fn,
                              iou_threshold=iou_threshold)
        per_video.append(result)

    n = len(per_video)
    if n == 0:
        return {"per_video": [], "mean_precision": 0, "mean_recall": 0, "mean_f1": 0}

    return {
        "per_video": per_video,
        "mean_precision": sum(v["precision"] for v in per_video) / n,
        "mean_recall": sum(v["recall"] for v in per_video) / n,
        "mean_f1": sum(v["f1"] for v in per_video) / n,
        "total_matched": sum(v["num_matched"] for v in per_video),
        "total_pred": sum(v["num_pred"] for v in per_video),
        "total_ref": sum(v["num_ref"] for v in per_video),
    }

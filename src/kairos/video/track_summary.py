"""Track summary building and formatting for YOLO detections."""

from __future__ import annotations

import math

import numpy as np

from kairos.video.spatial import (
    compute_relations,
    movement_label,
    path_metrics,
    position_label,
)
from kairos.video.tracking import build_tracks


def build_track_summaries(
    frames: list[np.ndarray],
    yolo_dict: dict[int, list[dict]],
    **kwargs: object,
) -> list[dict]:
    """Build per-scene track summaries with movement and relation labels.

    For each tracked object the function computes start/end positions,
    movement direction, path metrics, and inter-object relations.

    Args:
        frames: List of BGR image arrays for the scene (used to
            determine frame dimensions).
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries (each containing ``"track_id"``, ``"label"``,
            ``"confidence"``, and ``"bbox"`` keys).
        **kwargs: Additional keyword arguments forwarded to
            :func:`kairos.video.spatial.compute_relations`.

    Returns:
        A list of track summary dictionaries sorted by ``"track_id"``.
        Each dictionary contains:

        - ``"track_id"``: Integer track identifier.
        - ``"label"``: Object class label.
        - ``"confidence_avg"``: Mean detection confidence.
        - ``"start_frame"`` / ``"end_frame"``: First and last frame
          indices.
        - ``"start_pos"`` / ``"end_pos"``: Human-readable position
          labels.
        - ``"movement"``: Movement description string.
        - ``"path_length"``: Cumulative path distance.
        - ``"net_displacement"``: Straight-line displacement.
        - ``"direction_change_var"``: Heading change variance.
        - ``"relations"``: List of relation description strings.
    """
    tracks = build_tracks(yolo_dict)
    if not frames:
        return []
    frame_h, frame_w = frames[0].shape[:2]
    relations = compute_relations(tracks, yolo_dict, frame_w, frame_h, **kwargs)
    diag = math.hypot(frame_w, frame_h)
    summaries: list[dict] = []
    for track_id, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        if not dets:
            continue
        label: str = info.get("label", "unknown")
        start_bbox: list[float] = dets[0]["bbox"]
        end_bbox: list[float] = dets[-1]["bbox"]
        start_center: tuple[float, float] = (
            (start_bbox[0] + start_bbox[2]) / 2.0,
            (start_bbox[1] + start_bbox[3]) / 2.0,
        )
        end_center: tuple[float, float] = (
            (end_bbox[0] + end_bbox[2]) / 2.0,
            (end_bbox[1] + end_bbox[3]) / 2.0,
        )
        start_area: float = max(0.0, start_bbox[2] - start_bbox[0]) * max(
            0.0, start_bbox[3] - start_bbox[1]
        )
        end_area: float = max(0.0, end_bbox[2] - end_bbox[0]) * max(
            0.0, end_bbox[3] - end_bbox[1]
        )
        start_pos: str = position_label(
            start_center[0], start_center[1], frame_w, frame_h
        )
        end_pos: str = position_label(end_center[0], end_center[1], frame_w, frame_h)
        move: str = movement_label(
            start_center, end_center, start_area, end_area, frame_w, frame_h
        )
        pl, net_disp, angle_var = path_metrics(dets)
        if diag > 0:
            if net_disp < diag * 0.03 and pl > diag * 0.15 and angle_var > 0.2:
                move += ", looping/circling"
            elif pl > net_disp * 3 and angle_var > 0.3:
                move += ", moving in a loop"
        confs: list[float] = [d.get("confidence", 0.0) for d in dets]
        summaries.append(
            {
                "track_id": track_id,
                "label": label,
                "confidence_avg": round(sum(confs) / len(confs) if confs else 0.0, 3),
                "start_frame": dets[0]["frame_idx"],
                "end_frame": dets[-1]["frame_idx"],
                "start_pos": start_pos,
                "end_pos": end_pos,
                "movement": move,
                "path_length": round(pl, 3),
                "net_displacement": round(net_disp, 3),
                "direction_change_var": round(angle_var, 4),
                "relations": relations.get(track_id, []),
            }
        )
    summaries.sort(key=lambda d: d["track_id"])
    return summaries


def format_track_summary(summary: dict, style: str = "compact") -> str:
    """Format a single track summary dictionary as a human-readable string.

    Args:
        summary: A track summary dictionary as produced by
            :func:`build_track_summaries`.
        style: Formatting style.  ``"compact"`` produces a concise
            one-liner; ``"narrative"`` produces a natural-language
            sentence.

    Returns:
        A formatted string describing the track.
    """
    label: str = summary.get("label", "unknown")
    track_id: int = summary.get("track_id", "unknown")
    movement: str = summary.get("movement", "unknown")
    start_pos: str = summary.get("start_pos", "unknown")
    end_pos: str = summary.get("end_pos", "unknown")
    relations: list[str] = summary.get("relations", []) or []

    if style == "narrative":
        movement_phrase = movement.replace(",", "")
        base = (
            f"{label} #{track_id} is {movement_phrase} from {start_pos} to {end_pos}"
            if start_pos != "unknown"
            else f"{label} #{track_id} is {movement_phrase} to {end_pos}"
        )
        relation_phrases = [f"{label} #{track_id} is {rel}" for rel in relations]
        return "; ".join([base, *relation_phrases]) if relation_phrases else base

    base = f"{label} #{track_id}: {start_pos} -> {end_pos}, {movement}"
    if relations:
        base += f"; relations: {', '.join(relations)}"
    return base


def format_track_summaries(summaries: list[dict], style: str = "compact") -> list[str]:
    """Format a list of track summaries as human-readable strings.

    Args:
        summaries: A list of track summary dictionaries as produced by
            :func:`build_track_summaries`.
        style: Formatting style forwarded to
            :func:`format_track_summary`.

    Returns:
        A list of formatted strings, one per track summary.
    """
    return [format_track_summary(s, style=style) for s in summaries]

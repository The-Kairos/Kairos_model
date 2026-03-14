"""Track summary building and formatting for YOLO detections."""

import math

from kairos.video.tracking import build_tracks
from kairos.video.spatial import position_label, movement_label, path_metrics, compute_relations


def build_track_summaries(frames, yolo_dict, **kwargs) -> list:
    """Build per-scene track summaries with movement and relation labels."""
    tracks = build_tracks(yolo_dict)
    if not frames:
        return []
    frame_h, frame_w = frames[0].shape[:2]
    relations = compute_relations(tracks, yolo_dict, frame_w, frame_h, **kwargs)
    diag = math.hypot(frame_w, frame_h)
    summaries = []
    for track_id, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        if not dets:
            continue
        label = info.get("label", "unknown")
        start_bbox, end_bbox = dets[0]["bbox"], dets[-1]["bbox"]
        start_center = ((start_bbox[0] + start_bbox[2]) / 2.0, (start_bbox[1] + start_bbox[3]) / 2.0)
        end_center = ((end_bbox[0] + end_bbox[2]) / 2.0, (end_bbox[1] + end_bbox[3]) / 2.0)
        start_area = max(0.0, start_bbox[2] - start_bbox[0]) * max(0.0, start_bbox[3] - start_bbox[1])
        end_area = max(0.0, end_bbox[2] - end_bbox[0]) * max(0.0, end_bbox[3] - end_bbox[1])
        start_pos = position_label(start_center[0], start_center[1], frame_w, frame_h)
        end_pos = position_label(end_center[0], end_center[1], frame_w, frame_h)
        move = movement_label(start_center, end_center, start_area, end_area, frame_w, frame_h)
        pl, net_disp, angle_var = path_metrics(dets)
        if diag > 0:
            if net_disp < diag * 0.03 and pl > diag * 0.15 and angle_var > 0.2:
                move += ", looping/circling"
            elif pl > net_disp * 3 and angle_var > 0.3:
                move += ", moving in a loop"
        confs = [d.get("confidence", 0.0) for d in dets]
        summaries.append({
            "track_id": track_id, "label": label,
            "confidence_avg": round(sum(confs) / len(confs) if confs else 0.0, 3),
            "start_frame": dets[0]["frame_idx"], "end_frame": dets[-1]["frame_idx"],
            "start_pos": start_pos, "end_pos": end_pos, "movement": move,
            "path_length": round(pl, 3), "net_displacement": round(net_disp, 3),
            "direction_change_var": round(angle_var, 4),
            "relations": relations.get(track_id, []),
        })
    summaries.sort(key=lambda d: d["track_id"])
    return summaries


def format_track_summary(summary: dict, style: str = "compact") -> str:
    label = summary.get("label", "unknown")
    track_id = summary.get("track_id", "unknown")
    movement = summary.get("movement", "unknown")
    start_pos = summary.get("start_pos", "unknown")
    end_pos = summary.get("end_pos", "unknown")
    relations = summary.get("relations", []) or []

    if style == "narrative":
        movement_phrase = movement.replace(",", "")
        base = f"{label} #{track_id} is {movement_phrase} from {start_pos} to {end_pos}" if start_pos != "unknown" else f"{label} #{track_id} is {movement_phrase} to {end_pos}"
        relation_phrases = [f"{label} #{track_id} is {rel}" for rel in relations]
        return "; ".join([base] + relation_phrases) if relation_phrases else base

    base = f"{label} #{track_id}: {start_pos} -> {end_pos}, {movement}"
    if relations:
        base += f"; relations: {', '.join(relations)}"
    return base


def format_track_summaries(summaries: list, style: str = "compact") -> list:
    return [format_track_summary(s, style=style) for s in summaries]

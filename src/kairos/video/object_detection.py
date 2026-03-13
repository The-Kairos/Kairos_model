"""YOLOv8 object detection and tracking per scene."""

import math
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

from kairos.core.utils import print_prefixed


def run_yolo_on_frame(model, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list:
    """Run YOLOv8 on a single frame and return detections."""
    results = model.predict(frame, conf=conf, iou=iou, stream=True, verbose=False)
    detections = []
    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            detections.append({
                "label": model.names[cls],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })
    return detections


def run_yolo_track_on_frames(model, frames: list, conf: float = 0.25, iou: float = 0.45, tracker: str = "bytetrack.yaml"):
    """Run YOLOv8 tracking on a list of frames. Returns results or None."""
    try:
        return model.track(frames, conf=conf, iou=iou, tracker=tracker, persist=True, stream=True, verbose=False)
    except Exception:
        return None


def _parse_yolo_results(results, model) -> dict:
    yolo_dict = {}
    for idx, r in enumerate(results):
        dets = []
        if not hasattr(r, "boxes"):
            yolo_dict[idx] = dets
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            track_id = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id[0])
                except Exception:
                    track_id = None
            dets.append({
                "label": model.names[cls],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
                "track_id": track_id,
            })
        yolo_dict[idx] = dets
    return yolo_dict


def _bbox_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _has_track_ids(yolo_dict: dict) -> bool:
    for dets in yolo_dict.values():
        for det in dets:
            if det.get("track_id") is not None:
                return True
    return False


def _assign_track_ids_iou(yolo_dict: dict, iou_threshold: float = 0.3) -> dict:
    """Simple IoU-based tracker fallback."""
    next_id = 1
    active_tracks = []
    for frame_idx in sorted(yolo_dict.keys()):
        dets = yolo_dict.get(frame_idx, [])
        used_track_ids = set()
        for det in dets:
            best_track, best_iou = None, 0.0
            for track in active_tracks:
                if track["label"] != det.get("label"):
                    continue
                if track["id"] in used_track_ids:
                    continue
                iou = _bbox_iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track and best_iou >= iou_threshold:
                det["track_id"] = best_track["id"]
                best_track["bbox"] = det["bbox"]
                best_track["last_frame"] = frame_idx
                used_track_ids.add(best_track["id"])
            else:
                det["track_id"] = next_id
                active_tracks.append({
                    "id": next_id, "bbox": det["bbox"],
                    "label": det.get("label"), "last_frame": frame_idx,
                })
                used_track_ids.add(next_id)
                next_id += 1
        active_tracks = [t for t in active_tracks if frame_idx - t["last_frame"] <= 1]
    return yolo_dict


def _build_tracks(yolo_dict: dict) -> dict:
    tracks = {}
    for frame_idx, dets in yolo_dict.items():
        for det in dets:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            track = tracks.setdefault(track_id, {"label": det.get("label", "unknown"), "detections": []})
            track["detections"].append({
                "frame_idx": frame_idx,
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "confidence": det.get("confidence", 0.0),
            })
    return tracks


def _position_label(x_center, y_center, frame_w, frame_h) -> str:
    if frame_w <= 0 or frame_h <= 0:
        return "unknown"
    horiz = "left" if x_center < frame_w / 3 else "center" if x_center < 2 * frame_w / 3 else "right"
    vert = "top" if y_center < frame_h / 3 else "middle" if y_center < 2 * frame_h / 3 else "bottom"
    return f"{vert}-{horiz}"


def _movement_label(start_center, end_center, start_area, end_area, frame_w, frame_h) -> str:
    dx = end_center[0] - start_center[0]
    dy = end_center[1] - start_center[1]
    diag = math.hypot(frame_w, frame_h)
    dist = math.hypot(dx, dy)
    if diag <= 0:
        return "movement unknown"
    if dist < diag * 0.02:
        movement = "mostly stationary"
    else:
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        if abs(dx) > abs(dy) * 1.5:
            movement = f"moving {horiz}"
        elif abs(dy) > abs(dx) * 1.5:
            movement = f"moving {vert}"
        else:
            movement = f"moving {vert}-{horiz}"
    if start_area > 0 and end_area > 0:
        change = (end_area - start_area) / start_area
        if change > 0.2:
            movement += ", getting closer"
        elif change < -0.2:
            movement += ", getting farther"
    return movement


def _relative_relation_from_centers(cx1, cy1, cx2, cy2) -> str:
    dx, dy = cx2 - cx1, cy2 - cy1
    if abs(dx) >= abs(dy):
        return "left-of" if dx > 0 else "right-of"
    return "above" if dy > 0 else "below"


def _opposite_relation(rel: str) -> str:
    return {"left-of": "right-of", "right-of": "left-of", "above": "below", "below": "above"}.get(rel, rel)


def _angle_variance(angles: list) -> float:
    if len(angles) < 2:
        return 0.0
    diffs = [math.atan2(math.sin(angles[i] - angles[i - 1]), math.cos(angles[i] - angles[i - 1])) for i in range(1, len(angles))]
    mean = sum(diffs) / len(diffs)
    return sum((d - mean) ** 2 for d in diffs) / len(diffs)


def _path_metrics(dets: list) -> tuple:
    positions = []
    for d in dets:
        bbox = d.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        positions.append((d["frame_idx"], cx, cy))
    positions.sort(key=lambda p: p[0])
    if len(positions) < 2:
        return 0.0, 0.0, 0.0
    path_length = 0.0
    angles = []
    for i in range(1, len(positions)):
        _, x_prev, y_prev = positions[i - 1]
        _, x_curr, y_curr = positions[i]
        dx, dy = x_curr - x_prev, y_curr - y_prev
        step = math.hypot(dx, dy)
        path_length += step
        if step > 0:
            angles.append(math.atan2(dy, dx))
    start, end = positions[0], positions[-1]
    net_displacement = math.hypot(end[1] - start[1], end[2] - start[2])
    return path_length, net_displacement, _angle_variance(angles)


def _compute_relations(tracks, yolo_dict, frame_w, frame_h,
                       rel_min_frames=2, proximity_ratio=0.12,
                       moving_with_min_frames=2, moving_with_cos=0.8,
                       moving_with_speed_ratio=(0.5, 2.0), moving_with_min_speed=1.0):
    """Compute spatial relations and moving-with relations for tracked objects."""
    diag = math.hypot(frame_w, frame_h)
    if diag <= 0:
        return {}

    frame_centers = {}
    for frame_idx in sorted(yolo_dict.keys()):
        centers = []
        for d in yolo_dict.get(frame_idx, []):
            tid = d.get("track_id")
            if tid is None:
                continue
            bbox = d.get("bbox", [0, 0, 0, 0])
            centers.append({
                "track_id": tid, "label": d.get("label", "unknown"),
                "cx": (bbox[0] + bbox[2]) / 2.0, "cy": (bbox[1] + bbox[3]) / 2.0,
            })
        if centers:
            frame_centers[frame_idx] = centers

    rel_counts = {}
    for _, centers in frame_centers.items():
        for i in range(len(centers)):
            a = centers[i]
            for j in range(i + 1, len(centers)):
                b = centers[j]
                rel_ab = _relative_relation_from_centers(a["cx"], a["cy"], b["cx"], b["cy"])
                rel_ba = _opposite_relation(rel_ab)
                rel_counts.setdefault((a["track_id"], b["track_id"]), {}).setdefault(rel_ab, 0)
                rel_counts[(a["track_id"], b["track_id"])][rel_ab] += 1
                rel_counts.setdefault((b["track_id"], a["track_id"]), {}).setdefault(rel_ba, 0)
                rel_counts[(b["track_id"], a["track_id"])][rel_ba] += 1

    rel_results = {}
    for pair, counts in rel_counts.items():
        rel, count = max(counts.items(), key=lambda kv: kv[1])
        if count >= rel_min_frames:
            rel_results[pair] = rel

    track_positions = {}
    for tid, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        for d in dets:
            bbox = d.get("bbox", [0, 0, 0, 0])
            track_positions.setdefault(tid, []).append(
                (d["frame_idx"], (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            )

    track_vel = {}
    for tid, positions in track_positions.items():
        positions = sorted(positions, key=lambda p: p[0])
        for i in range(1, len(positions)):
            f_prev, x_prev, y_prev = positions[i - 1]
            f_curr, x_curr, y_curr = positions[i]
            track_vel.setdefault(tid, {})[f_curr] = (x_curr - x_prev, y_curr - y_prev)

    moving_counts = {}
    max_dist = diag * proximity_ratio
    for frame_idx, centers in frame_centers.items():
        center_map = {c["track_id"]: c for c in centers}
        ids = list(center_map.keys())
        for i in range(len(ids)):
            v1 = track_vel.get(ids[i], {}).get(frame_idx)
            if v1 is None:
                continue
            c1 = center_map[ids[i]]
            for j in range(i + 1, len(ids)):
                v2 = track_vel.get(ids[j], {}).get(frame_idx)
                if v2 is None:
                    continue
                c2 = center_map[ids[j]]
                dist = math.hypot(c2["cx"] - c1["cx"], c2["cy"] - c1["cy"])
                if dist > max_dist:
                    continue
                speed1, speed2 = math.hypot(*v1), math.hypot(*v2)
                if speed1 < moving_with_min_speed or speed2 < moving_with_min_speed:
                    continue
                cos_sim = (v1[0] * v2[0] + v1[1] * v2[1]) / (speed1 * speed2)
                ratio = speed1 / speed2 if speed2 else 0.0
                if cos_sim >= moving_with_cos and moving_with_speed_ratio[0] <= ratio <= moving_with_speed_ratio[1]:
                    moving_counts[(ids[i], ids[j])] = moving_counts.get((ids[i], ids[j]), 0) + 1
                    moving_counts[(ids[j], ids[i])] = moving_counts.get((ids[j], ids[i]), 0) + 1

    relations_map = {}
    track_labels = {tid: info.get("label", "unknown") for tid, info in tracks.items()}
    for (tid, other_id), rel in rel_results.items():
        relations_map.setdefault(tid, []).append(f"{rel} {track_labels.get(other_id, 'unknown')} #{other_id}")
    for (tid, other_id), count in moving_counts.items():
        if count >= moving_with_min_frames:
            relations_map.setdefault(tid, []).append(f"moving-with {track_labels.get(other_id, 'unknown')} #{other_id}")
    for tid in list(relations_map.keys()):
        relations_map[tid] = sorted(set(relations_map[tid]))
    return relations_map


def build_track_summaries(frames, yolo_dict, **kwargs) -> list:
    """Build per-scene track summaries with movement and relation labels."""
    tracks = _build_tracks(yolo_dict)
    if not frames:
        return []
    frame_h, frame_w = frames[0].shape[:2]
    relations = _compute_relations(tracks, yolo_dict, frame_w, frame_h, **kwargs)
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
        start_pos = _position_label(start_center[0], start_center[1], frame_w, frame_h)
        end_pos = _position_label(end_center[0], end_center[1], frame_w, frame_h)
        movement = _movement_label(start_center, end_center, start_area, end_area, frame_w, frame_h)
        path_length, net_disp, angle_var = _path_metrics(dets)
        if diag > 0:
            if net_disp < diag * 0.03 and path_length > diag * 0.15 and angle_var > 0.2:
                movement += ", looping/circling"
            elif path_length > net_disp * 3 and angle_var > 0.3:
                movement += ", moving in a loop"
        confs = [d.get("confidence", 0.0) for d in dets]
        summaries.append({
            "track_id": track_id, "label": label,
            "confidence_avg": round(sum(confs) / len(confs) if confs else 0.0, 3),
            "start_frame": dets[0]["frame_idx"], "end_frame": dets[-1]["frame_idx"],
            "start_pos": start_pos, "end_pos": end_pos, "movement": movement,
            "path_length": round(path_length, 3), "net_displacement": round(net_disp, 3),
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


def detect_object_yolo(
    scenes: list,
    model_size: str = "data/models/yolov8s.pt",
    conf: float = 0.5,
    iou: float = 0.45,
    output_dir: str = None,
    use_bytetrack: bool = True,
    tracker: str = "bytetrack.yaml",
    fallback_iou: float = 0.3,
    frame_key: str = "frames",
    summary_key: str = "yolo_detections",
    debug: bool = False,
    **track_kwargs,
) -> list:
    """Run YOLO on all scenes. Adds track summaries under *summary_key*."""
    model = YOLO(model_size)
    results_scenes = []

    for s, scene in enumerate(scenes):
        new_scene = dict(scene)
        frames = scene.get(frame_key, [])
        yolo_dict = {}

        if use_bytetrack and frames:
            results = run_yolo_track_on_frames(model, frames, conf=conf, iou=iou, tracker=tracker)
            if results is not None:
                yolo_dict = _parse_yolo_results(results, model)

        if not yolo_dict:
            for idx, frame in enumerate(frames):
                yolo_dict[idx] = run_yolo_on_frame(model, frame, conf=conf, iou=iou)

        if yolo_dict and not _has_track_ids(yolo_dict):
            yolo_dict = _assign_track_ids_iou(yolo_dict, iou_threshold=fallback_iou)

        if output_dir is not None:
            for idx, frame in enumerate(frames):
                debug_draw_yolo(
                    frame=frame,
                    detections=yolo_dict.get(idx, []),
                    save_path=f"./{output_dir}/scene_{s:03d}/detection_{idx:03d}.jpg",
                )

        new_scene[summary_key] = build_track_summaries(frames, yolo_dict, **track_kwargs)
        results_scenes.append(new_scene)

        if debug:
            lines = format_track_summaries(new_scene[summary_key], style="compact")
            print_prefixed("(YOLOv8)", f"Scene {s}:")
            for line in (lines or ["none detected"]):
                print_prefixed("(YOLOv8)", line, indent=4)

    return results_scenes


# Debug drawing

YOLO_COLOR_MAP = {}


def get_color_for_label(label: str):
    if label not in YOLO_COLOR_MAP:
        YOLO_COLOR_MAP[label] = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
    return YOLO_COLOR_MAP[label]


def debug_draw_yolo(frame: np.ndarray, detections: list, save_path: str = None) -> np.ndarray:
    """Draw YOLO detections on a frame for debugging."""
    pad = 20
    drawn = cv2.copyMakeBorder(frame, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    for det in detections:
        label, conf = det["label"], det["confidence"]
        x1, y1, x2, y2 = (int(v) + pad for v in det["bbox"])
        color = get_color_for_label(label)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

        track_id = det.get("track_id")
        text = f"{label}#{track_id} {conf:.2f}" if track_id is not None else f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(drawn, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(drawn, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(save_path, drawn)

    return drawn

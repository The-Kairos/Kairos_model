# src/yolo_inference.py
from ultralytics import YOLO
from src.debug_utils import print_prefixed
import math
import os
import random
import cv2
import numpy as np


def run_yolo_on_frame(
    model,
    frame: np.ndarray,  # process a single frame (np.ndarray)
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Run YOLOv8 on a single frame (np.ndarray).

    Args:
        model: Loaded YOLO model object
        frame: np.ndarray frame (BGR/RGB image)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        detections: list of dictionaries with:
            {
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
    """
    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        verbose=False
    )

    detections = []

    for r in results:
        if not hasattr(r, "boxes"):
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy,
            })

    return detections


def run_yolo_track_on_frames(
    model,
    frames: list,
    conf: float = 0.25,
    iou: float = 0.45,
    tracker: str = "bytetrack.yaml",
):
    """
    Run YOLOv8 tracking on a list of frames.
    Returns a list of results or None if tracking fails.
    """
    try:
        results = model.track(
            frames,
            conf=conf,
            iou=iou,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        return list(results)
    except Exception:
        return None


def _parse_yolo_results(results, model):
    yolo_dict = {}

    for idx, r in enumerate(results):
        dets = []

        if not hasattr(r, "boxes"):
            yolo_dict[idx] = dets
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            track_id = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id[0])
                except Exception:
                    track_id = None

            dets.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy,
                "track_id": track_id,
            })

        yolo_dict[idx] = dets

    return yolo_dict


def _bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def _has_track_ids(yolo_dict):
    for dets in yolo_dict.values():
        for det in dets:
            if det.get("track_id") is not None:
                return True
    return False


def _assign_track_ids_iou(yolo_dict, iou_threshold: float = 0.3):
    """
    Simple IOU-based tracker fallback (used when ByteTrack is unavailable).
    """
    next_id = 1
    active_tracks = []

    for frame_idx in sorted(yolo_dict.keys()):
        dets = yolo_dict.get(frame_idx, [])
        used_track_ids = set()

        for det in dets:
            best_track = None
            best_iou = 0.0

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
                    "id": next_id,
                    "bbox": det["bbox"],
                    "label": det.get("label"),
                    "last_frame": frame_idx,
                })
                used_track_ids.add(next_id)
                next_id += 1

        active_tracks = [
            t for t in active_tracks
            if frame_idx - t["last_frame"] <= 1
        ]

    return yolo_dict


def _build_tracks(yolo_dict):
    tracks = {}
    for frame_idx, dets in yolo_dict.items():
        for det in dets:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            track = tracks.setdefault(track_id, {
                "label": det.get("label", "unknown"),
                "detections": [],
            })
            track["detections"].append({
                "frame_idx": frame_idx,
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "confidence": det.get("confidence", 0.0),
            })
    return tracks


def _position_label(x_center, y_center, frame_w, frame_h):
    if frame_w <= 0 or frame_h <= 0:
        return "unknown"

    horiz = (
        "left" if x_center < frame_w / 3
        else "center" if x_center < 2 * frame_w / 3
        else "right"
    )
    vert = (
        "top" if y_center < frame_h / 3
        else "middle" if y_center < 2 * frame_h / 3
        else "bottom"
    )
    return f"{vert}-{horiz}"


def _movement_label(start_center, end_center, start_area, end_area, frame_w, frame_h):
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


def _relative_relation_from_centers(cx1, cy1, cx2, cy2):
    dx = cx2 - cx1
    dy = cy2 - cy1

    if abs(dx) >= abs(dy):
        return "left-of" if dx > 0 else "right-of"
    return "above" if dy > 0 else "below"


def _opposite_relation(rel):
    return {
        "left-of": "right-of",
        "right-of": "left-of",
        "above": "below",
        "below": "above",
    }.get(rel, rel)


def _angle_variance(angles):
    if len(angles) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        diff = math.atan2(math.sin(diff), math.cos(diff))
        diffs.append(diff)

    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return var


def _path_metrics(dets):
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
        dx = x_curr - x_prev
        dy = y_curr - y_prev
        step = math.hypot(dx, dy)
        path_length += step
        if step > 0:
            angles.append(math.atan2(dy, dx))

    start = positions[0]
    end = positions[-1]
    net_displacement = math.hypot(end[1] - start[1], end[2] - start[2])
    angle_var = _angle_variance(angles)

    return path_length, net_displacement, angle_var


def _compute_relations(
    tracks,
    yolo_dict,
    frame_w: int,
    frame_h: int,
    rel_min_frames: int = 2,
    proximity_ratio: float = 0.12,
    moving_with_min_frames: int = 2,
    moving_with_cos: float = 0.8,
    moving_with_speed_ratio=(0.5, 2.0),
    moving_with_min_speed: float = 1.0,
):
    """
    Compute center-based relative position relations and moving-with relations
    (proximity + velocity) for tracked objects.

    Returns: {track_id: [relation strings]}
    """
    diag = math.hypot(frame_w, frame_h)
    if diag <= 0:
        return {}

    # Per-frame centers
    frame_centers = {}
    for frame_idx in sorted(yolo_dict.keys()):
        centers = []
        for d in yolo_dict.get(frame_idx, []):
            tid = d.get("track_id")
            if tid is None:
                continue
            bbox = d.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            centers.append({
                "track_id": tid,
                "label": d.get("label", "unknown"),
                "cx": cx,
                "cy": cy,
            })
        if centers:
            frame_centers[frame_idx] = centers

    # Relative position counts
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

    # Velocity per track per frame
    track_positions = {}
    for tid, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        for d in dets:
            bbox = d.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            track_positions.setdefault(tid, []).append((d["frame_idx"], cx, cy))

    track_vel = {}
    for tid, positions in track_positions.items():
        positions = sorted(positions, key=lambda p: p[0])
        for i in range(1, len(positions)):
            f_prev, x_prev, y_prev = positions[i - 1]
            f_curr, x_curr, y_curr = positions[i]
            vx = x_curr - x_prev
            vy = y_curr - y_prev
            track_vel.setdefault(tid, {})[f_curr] = (vx, vy)

    # Moving-with counts (proximity + velocity)
    moving_counts = {}
    max_dist = diag * proximity_ratio

    for frame_idx, centers in frame_centers.items():
        center_map = {c["track_id"]: c for c in centers}

        ids = list(center_map.keys())
        for i in range(len(ids)):
            tid1 = ids[i]
            c1 = center_map[tid1]
            v1 = track_vel.get(tid1, {}).get(frame_idx)
            if v1 is None:
                continue
            for j in range(i + 1, len(ids)):
                tid2 = ids[j]
                c2 = center_map[tid2]
                v2 = track_vel.get(tid2, {}).get(frame_idx)
                if v2 is None:
                    continue

                dist = math.hypot(c2["cx"] - c1["cx"], c2["cy"] - c1["cy"])
                if dist > max_dist:
                    continue

                speed1 = math.hypot(v1[0], v1[1])
                speed2 = math.hypot(v2[0], v2[1])
                if speed1 < moving_with_min_speed or speed2 < moving_with_min_speed:
                    continue

                dot = v1[0] * v2[0] + v1[1] * v2[1]
                cos_sim = dot / (speed1 * speed2)
                ratio = speed1 / speed2 if speed2 else 0.0

                if cos_sim >= moving_with_cos and moving_with_speed_ratio[0] <= ratio <= moving_with_speed_ratio[1]:
                    moving_counts[(tid1, tid2)] = moving_counts.get((tid1, tid2), 0) + 1
                    moving_counts[(tid2, tid1)] = moving_counts.get((tid2, tid1), 0) + 1

    relations_map = {}
    track_labels = {tid: info.get("label", "unknown") for tid, info in tracks.items()}

    for (tid, other_id), rel in rel_results.items():
        other_label = track_labels.get(other_id, "unknown")
        relations_map.setdefault(tid, []).append(f"{rel} {other_label} #{other_id}")

    for (tid, other_id), count in moving_counts.items():
        if count >= moving_with_min_frames:
            other_label = track_labels.get(other_id, "unknown")
            relations_map.setdefault(tid, []).append(f"moving-with {other_label} #{other_id}")

    for tid in list(relations_map.keys()):
        unique = sorted(set(relations_map[tid]))
        relations_map[tid] = unique

    return relations_map


def build_track_summaries(
    frames,
    yolo_dict,
    rel_min_frames: int = 2,
    proximity_ratio: float = 0.12,
    moving_with_min_frames: int = 2,
    moving_with_cos: float = 0.8,
    moving_with_speed_ratio=(0.5, 2.0),
    moving_with_min_speed: float = 1.0,
):
    """
    Build per-scene track summaries with movement and relation labels.
    Returns list of summary dicts.
    """
    tracks = _build_tracks(yolo_dict)

    if not frames:
        return []

    frame_h, frame_w = frames[0].shape[:2]
    relations = _compute_relations(
        tracks,
        yolo_dict,
        frame_w,
        frame_h,
        rel_min_frames=rel_min_frames,
        proximity_ratio=proximity_ratio,
        moving_with_min_frames=moving_with_min_frames,
        moving_with_cos=moving_with_cos,
        moving_with_speed_ratio=moving_with_speed_ratio,
        moving_with_min_speed=moving_with_min_speed,
    )

    diag = math.hypot(frame_w, frame_h)
    summaries = []

    for track_id, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        if not dets:
            continue

        label = info.get("label", "unknown")
        start_bbox = dets[0]["bbox"]
        end_bbox = dets[-1]["bbox"]

        start_center = ((start_bbox[0] + start_bbox[2]) / 2.0,
                        (start_bbox[1] + start_bbox[3]) / 2.0)
        end_center = ((end_bbox[0] + end_bbox[2]) / 2.0,
                      (end_bbox[1] + end_bbox[3]) / 2.0)

        start_area = max(0.0, (start_bbox[2] - start_bbox[0])) * max(0.0, (start_bbox[3] - start_bbox[1]))
        end_area = max(0.0, (end_bbox[2] - end_bbox[0])) * max(0.0, (end_bbox[3] - end_bbox[1]))

        start_pos = _position_label(start_center[0], start_center[1], frame_w, frame_h)
        end_pos = _position_label(end_center[0], end_center[1], frame_w, frame_h)
        movement = _movement_label(start_center, end_center, start_area, end_area, frame_w, frame_h)

        path_length, net_disp, angle_var = _path_metrics(dets)

        if diag > 0:
            if net_disp < diag * 0.03 and path_length > diag * 0.15 and angle_var > 0.2:
                movement += ", looping/circling"
            elif path_length > net_disp * 3 and angle_var > 0.3:
                movement += ", moving in a loop"

        rel_list = relations.get(track_id, [])

        confs = [d.get("confidence", 0.0) for d in dets]
        conf_avg = sum(confs) / len(confs) if confs else 0.0

        summaries.append({
            "track_id": track_id,
            "label": label,
            "confidence_avg": round(conf_avg, 3),
            "start_frame": dets[0]["frame_idx"],
            "end_frame": dets[-1]["frame_idx"],
            "start_pos": start_pos,
            "end_pos": end_pos,
            "movement": movement,
            "path_length": round(path_length, 3),
            "net_displacement": round(net_disp, 3),
            "direction_change_var": round(angle_var, 4),
            "relations": rel_list,
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
        base = f"{label} #{track_id} is {movement_phrase} to {end_pos}"
        if start_pos and start_pos != "unknown":
            base = f"{label} #{track_id} is {movement_phrase} from {start_pos} to {end_pos}"

        relation_phrases = [
            f"{label} #{track_id} is {rel}"
            for rel in relations
        ]
        if relation_phrases:
            return "; ".join([base] + relation_phrases)
        return base

    base = f"{label} #{track_id}: {start_pos} -> {end_pos}, {movement}"
    if relations:
        base += f"; relations: {', '.join(relations)}"
    return base


def format_track_summaries(summaries: list, style: str = "compact") -> list:
    return [format_track_summary(s, style=style) for s in summaries]


def detect_object_yolo(
    scenes: list,  # process full scenes list
    model_size: str = "model/yolov8s.pt",
    conf: float = 0.5,
    iou: float = 0.45,
    output_dir: str = None,
    use_bytetrack: bool = True,
    tracker: str = "bytetrack.yaml",
    fallback_iou: float = 0.3,
    frame_key: str = "frames",
    summary_key: str = "yolo_detections",
    debug: bool = False,
    rel_min_frames: int = 2,
    proximity_ratio: float = 0.12,
    moving_with_min_frames: int = 2,
    moving_with_cos: float = 0.8,
    moving_with_speed_ratio=(0.5, 2.0),
    moving_with_min_speed: float = 1.0,
):
    """
    Run YOLO on a list of scenes.
    Adds:
      - scene[summary_key] = [track summary dicts]

    Args:
        scenes: list of scene dictionaries
        model_size: YOLO model name (e.g., yolov8s)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        updated scenes with YOLO outputs added
    """

    model = YOLO(model_size)

    results_scenes = []

    for s, scene in enumerate(scenes):
        new_scene = dict(scene)

        frames = scene.get(frame_key, [])
        yolo_dict = {}

        if use_bytetrack and frames:
            results = run_yolo_track_on_frames(
                model,
                frames,
                conf=conf,
                iou=iou,
                tracker=tracker,
            )
            if results is not None:
                yolo_dict = _parse_yolo_results(results, model)

        if not yolo_dict:
            # fallback to per-frame detection
            for idx, frame in enumerate(frames):
                detections = run_yolo_on_frame(
                    model,
                    frame,
                    conf=conf,
                    iou=iou
                )
                yolo_dict[idx] = detections

        if yolo_dict and not _has_track_ids(yolo_dict):
            yolo_dict = _assign_track_ids_iou(yolo_dict, iou_threshold=fallback_iou)

        # debug draw (if requested)
        if output_dir is not None:
            for idx, frame in enumerate(frames):
                detections = yolo_dict.get(idx, [])
                debug_draw_yolo(
                    frame=frame,
                    detections=detections,
                    save_path=f"./{output_dir}/scene_{s:03d}/detection_{idx:03d}.jpg",
                )

        track_summaries = build_track_summaries(
            frames,
            yolo_dict,
            rel_min_frames=rel_min_frames,
            proximity_ratio=proximity_ratio,
            moving_with_min_frames=moving_with_min_frames,
            moving_with_cos=moving_with_cos,
            moving_with_speed_ratio=moving_with_speed_ratio,
            moving_with_min_speed=moving_with_min_speed,
        )

        new_scene[summary_key] = track_summaries
        results_scenes.append(new_scene)

        if debug:
            lines = format_track_summaries(track_summaries, style="compact")
            print_prefixed("(YOLOv8)", f"Scene {s}:")
            if lines:
                for line in lines:
                    print_prefixed("(YOLOv8)", line, indent=4)
            else:
                print_prefixed("(YOLOv8)", "none detected", indent=4)

    return results_scenes

# ================================================================
# saving images for debugging

# cache class -> color mapping so colors stay consistent
YOLO_COLOR_MAP = {}


def get_color_for_label(label: str):
    """Return a bright, unique color for each label."""
    if label not in YOLO_COLOR_MAP:
        YOLO_COLOR_MAP[label] = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255)
        )
    return YOLO_COLOR_MAP[label]


def debug_draw_yolo(
    frame: np.ndarray,
    detections: list,
    save_path: str = None
):
    """
    Draw YOLO detections on a frame for debugging.
    - Smaller text
    - Per-class consistent colors
    """

    pad = 20
    drawn = cv2.copyMakeBorder(
        frame,
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1 += pad
        y1 += pad
        x2 += pad
        y2 += pad

        # Unique color for class
        color = get_color_for_label(label)

        # Thinner lines
        thickness = 2

        # --- Draw bounding box
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)

        # --- Smaller text
        font_scale = 0.4   # half size from before
        font_thickness = 1

        track_id = det.get("track_id")
        if track_id is not None:
            text = f"{label}#{track_id} {conf:.2f}"
        else:
            text = f"{label} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        # Text background
        cv2.rectangle(
            drawn,
            (x1, y1 - th - 4),
            (x1 + tw + 2, y1),
            color,
            -1
        )

        # Text on top
        cv2.putText(
            drawn,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

    # --- save if needed
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        cv2.imwrite(save_path, drawn)

    return drawn

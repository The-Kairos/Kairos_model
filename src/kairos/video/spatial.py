"""Spatial analysis: position labels, movement detection, and inter-object relations."""

import math


def position_label(x_center, y_center, frame_w, frame_h) -> str:
    if frame_w <= 0 or frame_h <= 0:
        return "unknown"
    horiz = (
        "left"
        if x_center < frame_w / 3
        else "center"
        if x_center < 2 * frame_w / 3
        else "right"
    )
    vert = (
        "top"
        if y_center < frame_h / 3
        else "middle"
        if y_center < 2 * frame_h / 3
        else "bottom"
    )
    return f"{vert}-{horiz}"


def movement_label(
    start_center, end_center, start_area, end_area, frame_w, frame_h
) -> str:
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
    return {
        "left-of": "right-of",
        "right-of": "left-of",
        "above": "below",
        "below": "above",
    }.get(rel, rel)


def _angle_variance(angles: list) -> float:
    if len(angles) < 2:
        return 0.0
    diffs = [
        math.atan2(
            math.sin(angles[i] - angles[i - 1]), math.cos(angles[i] - angles[i - 1])
        )
        for i in range(1, len(angles))
    ]
    mean = sum(diffs) / len(diffs)
    return sum((d - mean) ** 2 for d in diffs) / len(diffs)


def path_metrics(dets: list) -> tuple:
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


# ---------------------------------------------------------------------------
# compute_relations helpers
# ---------------------------------------------------------------------------


def _extract_frame_centers(yolo_dict):
    """Extract per-frame center positions from YOLO detections."""
    frame_centers = {}
    for frame_idx in sorted(yolo_dict.keys()):
        centers = []
        for d in yolo_dict.get(frame_idx, []):
            tid = d.get("track_id")
            if tid is None:
                continue
            bbox = d.get("bbox", [0, 0, 0, 0])
            centers.append(
                {
                    "track_id": tid,
                    "label": d.get("label", "unknown"),
                    "cx": (bbox[0] + bbox[2]) / 2.0,
                    "cy": (bbox[1] + bbox[3]) / 2.0,
                }
            )
        if centers:
            frame_centers[frame_idx] = centers
    return frame_centers


def _compute_spatial_relations(frame_centers, rel_min_frames):
    """Count pairwise spatial relations across frames and keep dominant ones."""
    rel_counts = {}
    for _, centers in frame_centers.items():
        for i in range(len(centers)):
            a = centers[i]
            for j in range(i + 1, len(centers)):
                b = centers[j]
                rel_ab = _relative_relation_from_centers(
                    a["cx"], a["cy"], b["cx"], b["cy"]
                )
                rel_ba = _opposite_relation(rel_ab)
                rel_counts.setdefault((a["track_id"], b["track_id"]), {}).setdefault(
                    rel_ab, 0
                )
                rel_counts[(a["track_id"], b["track_id"])][rel_ab] += 1
                rel_counts.setdefault((b["track_id"], a["track_id"]), {}).setdefault(
                    rel_ba, 0
                )
                rel_counts[(b["track_id"], a["track_id"])][rel_ba] += 1

    rel_results = {}
    for pair, counts in rel_counts.items():
        rel, count = max(counts.items(), key=lambda kv: kv[1])
        if count >= rel_min_frames:
            rel_results[pair] = rel
    return rel_results


def _compute_track_velocities(tracks):
    """Compute per-frame velocity vectors for each track."""
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
    return track_vel


def _compute_moving_with(
    frame_centers,
    track_vel,
    diag,
    proximity_ratio,
    moving_with_cos,
    moving_with_speed_ratio,
    moving_with_min_speed,
):
    """Detect pairs of tracks moving together based on velocity similarity and proximity."""
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
                if (
                    cos_sim >= moving_with_cos
                    and moving_with_speed_ratio[0]
                    <= ratio
                    <= moving_with_speed_ratio[1]
                ):
                    moving_counts[(ids[i], ids[j])] = (
                        moving_counts.get((ids[i], ids[j]), 0) + 1
                    )
                    moving_counts[(ids[j], ids[i])] = (
                        moving_counts.get((ids[j], ids[i]), 0) + 1
                    )
    return moving_counts


def compute_relations(
    tracks,
    yolo_dict,
    frame_w,
    frame_h,
    rel_min_frames=2,
    proximity_ratio=0.12,
    moving_with_min_frames=2,
    moving_with_cos=0.8,
    moving_with_speed_ratio=(0.5, 2.0),
    moving_with_min_speed=1.0,
):
    """Compute spatial relations and moving-with relations for tracked objects."""
    diag = math.hypot(frame_w, frame_h)
    if diag <= 0:
        return {}

    frame_centers = _extract_frame_centers(yolo_dict)
    rel_results = _compute_spatial_relations(frame_centers, rel_min_frames)
    track_vel = _compute_track_velocities(tracks)
    moving_counts = _compute_moving_with(
        frame_centers,
        track_vel,
        diag,
        proximity_ratio,
        moving_with_cos,
        moving_with_speed_ratio,
        moving_with_min_speed,
    )

    relations_map = {}
    track_labels = {tid: info.get("label", "unknown") for tid, info in tracks.items()}
    for (tid, other_id), rel in rel_results.items():
        relations_map.setdefault(tid, []).append(
            f"{rel} {track_labels.get(other_id, 'unknown')} #{other_id}"
        )
    for (tid, other_id), count in moving_counts.items():
        if count >= moving_with_min_frames:
            relations_map.setdefault(tid, []).append(
                f"moving-with {track_labels.get(other_id, 'unknown')} #{other_id}"
            )
    for tid in list(relations_map.keys()):
        relations_map[tid] = sorted(set(relations_map[tid]))
    return relations_map

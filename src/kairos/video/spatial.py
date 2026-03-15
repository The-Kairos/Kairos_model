"""Spatial analysis: position labels, movement detection, and inter-object relations."""

from __future__ import annotations

import math


def position_label(
    x_center: float, y_center: float, frame_w: int, frame_h: int
) -> str:
    """Compute a human-readable position label for a point in a frame.

    The frame is divided into a 3×3 grid and the label combines a
    vertical component (``"top"``, ``"middle"``, ``"bottom"``) with a
    horizontal component (``"left"``, ``"center"``, ``"right"``).

    Args:
        x_center: Horizontal coordinate of the point.
        y_center: Vertical coordinate of the point.
        frame_w: Width of the frame in pixels.
        frame_h: Height of the frame in pixels.

    Returns:
        A position string such as ``"top-left"`` or ``"middle-center"``.
        Returns ``"unknown"`` if *frame_w* or *frame_h* is non-positive.
    """
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
    start_center: tuple[float, float],
    end_center: tuple[float, float],
    start_area: float,
    end_area: float,
    frame_w: int,
    frame_h: int,
) -> str:
    """Describe the movement of a tracked object between its first and last detection.

    The movement direction is determined from the displacement vector and
    an optional depth cue is appended based on bounding-box area change
    (``"getting closer"`` or ``"getting farther"``).

    Args:
        start_center: ``(x, y)`` center of the object in the first
            detection.
        end_center: ``(x, y)`` center of the object in the last
            detection.
        start_area: Bounding-box area of the first detection.
        end_area: Bounding-box area of the last detection.
        frame_w: Width of the frame in pixels.
        frame_h: Height of the frame in pixels.

    Returns:
        A human-readable movement description string, e.g.
        ``"moving right, getting closer"`` or ``"mostly stationary"``.
    """
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


def _relative_relation_from_centers(
    cx1: float, cy1: float, cx2: float, cy2: float
) -> str:
    """Determine the dominant spatial relation of object 1 relative to object 2.

    The relation is based on the displacement vector from center 1 to
    center 2.  The dominant axis (horizontal vs. vertical) decides the
    label.

    Args:
        cx1: X-coordinate of the first object's center.
        cy1: Y-coordinate of the first object's center.
        cx2: X-coordinate of the second object's center.
        cy2: Y-coordinate of the second object's center.

    Returns:
        One of ``"left-of"``, ``"right-of"``, ``"above"``, or
        ``"below"``, describing how object 1 is positioned relative to
        object 2.
    """
    dx, dy = cx2 - cx1, cy2 - cy1
    if abs(dx) >= abs(dy):
        return "left-of" if dx > 0 else "right-of"
    return "above" if dy > 0 else "below"


def _opposite_relation(rel: str) -> str:
    """Return the spatial opposite of a relation string.

    Args:
        rel: A spatial relation string (e.g. ``"left-of"``).

    Returns:
        The opposite relation (e.g. ``"right-of"``).  If the input is
        not recognised, it is returned unchanged.
    """
    return {
        "left-of": "right-of",
        "right-of": "left-of",
        "above": "below",
        "below": "above",
    }.get(rel, rel)


def _angle_variance(angles: list[float]) -> float:
    """Compute the variance of successive angular differences.

    This provides a measure of how much the direction of movement
    changes over time.

    Args:
        angles: A list of angles in radians (e.g. from
            ``math.atan2``).

    Returns:
        The variance of the angular differences.  Returns ``0.0`` when
        fewer than two angles are provided.
    """
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


def path_metrics(dets: list[dict]) -> tuple[float, float, float]:
    """Compute path-related metrics from a sequence of detections.

    Args:
        dets: A list of detection dictionaries, each containing at least
            ``"frame_idx"`` (int) and ``"bbox"`` (list of four floats).

    Returns:
        A tuple of ``(path_length, net_displacement, angle_variance)``
        where:

        - *path_length* is the cumulative distance travelled along the
          path.
        - *net_displacement* is the straight-line distance between the
          first and last positions.
        - *angle_variance* is the variance of successive heading
          changes (see :func:`_angle_variance`).

        All three are ``0.0`` when fewer than two detections are
        provided.
    """
    positions: list[tuple[int, float, float]] = []
    for d in dets:
        bbox = d.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        positions.append((d["frame_idx"], cx, cy))
    positions.sort(key=lambda p: p[0])
    if len(positions) < 2:
        return 0.0, 0.0, 0.0
    path_length: float = 0.0
    angles: list[float] = []
    for i in range(1, len(positions)):
        _, x_prev, y_prev = positions[i - 1]
        _, x_curr, y_curr = positions[i]
        dx, dy = x_curr - x_prev, y_curr - y_prev
        step = math.hypot(dx, dy)
        path_length += step
        if step > 0:
            angles.append(math.atan2(dy, dx))
    start, end = positions[0], positions[-1]
    net_displacement: float = math.hypot(end[1] - start[1], end[2] - start[2])
    return path_length, net_displacement, _angle_variance(angles)


# ---------------------------------------------------------------------------
# compute_relations helpers
# ---------------------------------------------------------------------------


def _extract_frame_centers(
    yolo_dict: dict[int, list[dict]],
) -> dict[int, list[dict]]:
    """Extract per-frame center positions from YOLO detections.

    Args:
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries.  Each detection must include ``"track_id"``,
            ``"label"``, and ``"bbox"`` keys.

    Returns:
        A mapping from frame index to a list of center dictionaries,
        each containing ``"track_id"``, ``"label"``, ``"cx"``, and
        ``"cy"`` keys.  Frames with no tracked detections are omitted.
    """
    frame_centers: dict[int, list[dict]] = {}
    for frame_idx in sorted(yolo_dict.keys()):
        centers: list[dict] = []
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


def _compute_spatial_relations(
    frame_centers: dict[int, list[dict]], rel_min_frames: int
) -> dict[tuple[int, int], str]:
    """Count pairwise spatial relations across frames and keep dominant ones.

    For every pair of tracked objects, the function tallies which spatial
    relation (left-of, right-of, above, below) occurs most often.  Only
    relations that appear in at least *rel_min_frames* frames are kept.

    Args:
        frame_centers: Per-frame center dictionaries as returned by
            :func:`_extract_frame_centers`.
        rel_min_frames: Minimum number of frames a relation must appear
            in to be included in the output.

    Returns:
        A mapping from ``(track_id_a, track_id_b)`` to the dominant
        spatial relation string.
    """
    rel_counts: dict[tuple[int, int], dict[str, int]] = {}
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

    rel_results: dict[tuple[int, int], str] = {}
    for pair, counts in rel_counts.items():
        rel, count = max(counts.items(), key=lambda kv: kv[1])
        if count >= rel_min_frames:
            rel_results[pair] = rel
    return rel_results


def _compute_track_velocities(
    tracks: dict[int, dict],
) -> dict[int, dict[int, tuple[float, float]]]:
    """Compute per-frame velocity vectors for each track.

    Velocity is calculated as the displacement between consecutive
    detections for the same track.

    Args:
        tracks: A mapping from track ID to a dictionary containing a
            ``"detections"`` list.  Each detection must include
            ``"frame_idx"`` and ``"bbox"`` keys.

    Returns:
        A nested mapping ``{track_id: {frame_idx: (vx, vy)}}`` where
        ``(vx, vy)`` is the velocity vector at that frame.
    """
    track_positions: dict[int, list[tuple[int, float, float]]] = {}
    for tid, info in tracks.items():
        dets = sorted(info["detections"], key=lambda d: d["frame_idx"])
        for d in dets:
            bbox = d.get("bbox", [0, 0, 0, 0])
            track_positions.setdefault(tid, []).append(
                (d["frame_idx"], (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            )

    track_vel: dict[int, dict[int, tuple[float, float]]] = {}
    for tid, positions in track_positions.items():
        positions = sorted(positions, key=lambda p: p[0])
        for i in range(1, len(positions)):
            _f_prev, x_prev, y_prev = positions[i - 1]
            f_curr, x_curr, y_curr = positions[i]
            track_vel.setdefault(tid, {})[f_curr] = (x_curr - x_prev, y_curr - y_prev)
    return track_vel


def _compute_moving_with(
    frame_centers: dict[int, list[dict]],
    track_vel: dict[int, dict[int, tuple[float, float]]],
    diag: float,
    proximity_ratio: float,
    moving_with_cos: float,
    moving_with_speed_ratio: tuple[float, float],
    moving_with_min_speed: float,
) -> dict[tuple[int, int], int]:
    """Detect pairs of tracks moving together.

    Two tracks are considered to be "moving with" each other in a given
    frame when they are within a distance threshold, their velocity
    vectors are similar (cosine similarity ≥ *moving_with_cos*), and
    their speed ratio falls within *moving_with_speed_ratio*.

    Args:
        frame_centers: Per-frame center dictionaries as returned by
            :func:`_extract_frame_centers`.
        track_vel: Per-frame velocity vectors as returned by
            :func:`_compute_track_velocities`.
        diag: Diagonal length of the frame (used to compute the
            proximity threshold).
        proximity_ratio: Maximum distance between two tracks (as a
            fraction of *diag*) for them to be considered proximate.
        moving_with_cos: Minimum cosine similarity between velocity
            vectors.
        moving_with_speed_ratio: ``(min_ratio, max_ratio)`` acceptable
            speed ratio range between two tracks.
        moving_with_min_speed: Minimum speed (in pixels per frame) for
            a track to be considered as moving.

    Returns:
        A mapping from ``(track_id_a, track_id_b)`` to the number of
        frames in which the pair was observed moving together.
    """
    moving_counts: dict[tuple[int, int], int] = {}
    max_dist: float = diag * proximity_ratio
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
    tracks: dict[int, dict],
    yolo_dict: dict[int, list[dict]],
    frame_w: int,
    frame_h: int,
    rel_min_frames: int = 2,
    proximity_ratio: float = 0.12,
    moving_with_min_frames: int = 2,
    moving_with_cos: float = 0.8,
    moving_with_speed_ratio: tuple[float, float] = (0.5, 2.0),
    moving_with_min_speed: float = 1.0,
) -> dict[int, list[str]]:
    """Compute spatial relations and moving-with relations for tracked objects.

    Combines dominant pairwise spatial relations (left-of, right-of,
    above, below) with velocity-based "moving-with" detection.

    Args:
        tracks: A mapping from track ID to track info dictionaries
            (as returned by :func:`kairos.video.tracking.build_tracks`).
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries.
        frame_w: Width of the frame in pixels.
        frame_h: Height of the frame in pixels.
        rel_min_frames: Minimum number of frames a spatial relation must
            appear in to be included.
        proximity_ratio: Maximum proximity distance as a fraction of the
            frame diagonal.
        moving_with_min_frames: Minimum number of frames two tracks must
            be observed moving together.
        moving_with_cos: Minimum cosine similarity for velocity vectors.
        moving_with_speed_ratio: Acceptable speed ratio range
            ``(min, max)``.
        moving_with_min_speed: Minimum speed threshold (in pixels per
            frame).

    Returns:
        A mapping from track ID to a sorted list of relation description
        strings.  Returns an empty dictionary if the frame diagonal is
        non-positive.
    """
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

    relations_map: dict[int, list[str]] = {}
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

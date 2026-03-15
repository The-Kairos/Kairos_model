"""Object tracking: IoU-based fallback tracker and track building."""


def _bbox_iou(b1: list[float], b2: list[float]) -> float:
    """Compute the Intersection-over-Union (IoU) of two axis-aligned bounding boxes.

    Both boxes are expected in ``[x1, y1, x2, y2]`` format.

    Args:
        b1: First bounding box as ``[x1, y1, x2, y2]``.
        b2: Second bounding box as ``[x1, y1, x2, y2]``.

    Returns:
        The IoU value in the range ``[0.0, 1.0]``.
    """
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def has_track_ids(yolo_dict: dict[int, list[dict]]) -> bool:
    """Check whether any detection in *yolo_dict* already has a track ID.

    Args:
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries.

    Returns:
        ``True`` if at least one detection contains a non-``None``
        ``"track_id"`` value, ``False`` otherwise.
    """
    for dets in yolo_dict.values():
        for det in dets:
            if det.get("track_id") is not None:
                return True
    return False


def assign_track_ids_iou(
    yolo_dict: dict[int, list[dict]], iou_threshold: float = 0.3
) -> dict[int, list[dict]]:
    """Assign track IDs to detections using a simple IoU-based fallback tracker.

    Detections are processed frame-by-frame in order.  Each detection is
    matched to the active track with the highest IoU (same label
    required).  Unmatched detections start a new track.  Tracks that
    have not been seen for more than one frame are pruned.

    Args:
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries.  Each detection must contain ``"bbox"`` and
            ``"label"`` keys.  The ``"track_id"`` key is set in-place.
        iou_threshold: Minimum IoU required to match a detection to an
            existing track.

    Returns:
        The same *yolo_dict* mapping, with ``"track_id"`` assigned to
        every detection in-place.
    """
    next_id: int = 1
    active_tracks: list[dict] = []
    for frame_idx in sorted(yolo_dict.keys()):
        dets = yolo_dict.get(frame_idx, [])
        used_track_ids: set[int] = set()
        for det in dets:
            best_track = None
            best_iou: float = 0.0
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
                active_tracks.append(
                    {
                        "id": next_id,
                        "bbox": det["bbox"],
                        "label": det.get("label"),
                        "last_frame": frame_idx,
                    }
                )
                used_track_ids.add(next_id)
                next_id += 1
        active_tracks = [t for t in active_tracks if frame_idx - t["last_frame"] <= 1]
    return yolo_dict


def build_tracks(yolo_dict: dict[int, list[dict]]) -> dict[int, dict]:
    """Group detections by track ID into per-track dictionaries.

    Args:
        yolo_dict: A mapping from frame index to a list of detection
            dictionaries.  Each detection should contain ``"track_id"``,
            ``"label"``, ``"bbox"``, and ``"confidence"`` keys.

    Returns:
        A mapping from track ID to a dictionary with:

        - ``"label"``: The object class label.
        - ``"detections"``: A list of detection records, each containing
          ``"frame_idx"``, ``"bbox"``, and ``"confidence"``.

        Detections without a ``"track_id"`` are skipped.
    """
    tracks: dict[int, dict] = {}
    for frame_idx, dets in yolo_dict.items():
        for det in dets:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            track = tracks.setdefault(
                track_id, {"label": det.get("label", "unknown"), "detections": []}
            )
            track["detections"].append(
                {
                    "frame_idx": frame_idx,
                    "bbox": det.get("bbox", [0, 0, 0, 0]),
                    "confidence": det.get("confidence", 0.0),
                }
            )
    return tracks

"""Object tracking: IoU-based fallback tracker and track building."""


def _bbox_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def has_track_ids(yolo_dict: dict) -> bool:
    for dets in yolo_dict.values():
        for det in dets:
            if det.get("track_id") is not None:
                return True
    return False


def assign_track_ids_iou(yolo_dict: dict, iou_threshold: float = 0.3) -> dict:
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


def build_tracks(yolo_dict: dict) -> dict:
    tracks = {}
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

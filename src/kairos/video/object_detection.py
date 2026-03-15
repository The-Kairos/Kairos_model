"""YOLOv8 object detection and tracking per scene (orchestrator)."""

from typing import Any

from ultralytics import YOLO

from kairos.core.utils import print_prefixed
from kairos.video.debug_draw import debug_draw_yolo
from kairos.video.track_summary import build_track_summaries, format_track_summaries
from kairos.video.tracking import assign_track_ids_iou, has_track_ids
from kairos.video.yolo_inference import (
    parse_yolo_results,
    run_yolo_on_frame,
    run_yolo_track_on_frames,
)


def detect_object_yolo(
    scenes: list[dict],
    model_size: str = "models/yolov8s.pt",
    model: Any = None,
    conf: float = 0.5,
    iou: float = 0.45,
    output_dir: str | None = None,
    use_bytetrack: bool = True,
    tracker: str = "bytetrack.yaml",
    fallback_iou: float = 0.3,
    frame_key: str = "frames",
    summary_key: str = "yolo_detections",
    debug: bool = False,
    **track_kwargs: Any,
) -> list[dict]:
    """Run YOLO detection and tracking on all scenes.

    For each scene the function runs YOLOv8 inference on every frame,
    assigns track IDs (via ByteTrack or an IoU-based fallback), builds
    per-track summaries, and stores them under *summary_key* in each
    scene dictionary.

    Args:
        scenes: A list of scene dictionaries, each expected to contain
            frames under the key specified by *frame_key*.
        model_size: Path to the YOLO model weights file.  Only used when
            *model* is ``None``.
        model: A pre-loaded ``ultralytics.YOLO`` model instance.  If
            ``None``, a new model is loaded from *model_size*.
        conf: Minimum confidence threshold for detections.
        iou: IoU threshold used during non-maximum suppression.
        output_dir: If provided, annotated debug images are saved to
            this directory.
        use_bytetrack: If ``True``, use ByteTrack for multi-frame
            tracking.  Falls back to per-frame detection when tracking
            fails.
        tracker: Tracker configuration file name (e.g.
            ``"bytetrack.yaml"``).
        fallback_iou: IoU threshold for the simple IoU-based fallback
            tracker when ByteTrack does not produce track IDs.
        frame_key: Key used to look up frames in each scene dictionary.
        summary_key: Key under which track summaries are stored in each
            output scene dictionary.
        debug: If ``True``, print compact track summaries to stdout.
        **track_kwargs: Additional keyword arguments forwarded to
            :func:`kairos.video.track_summary.build_track_summaries`.

    Returns:
        A new list of scene dictionaries, each augmented with track
        summaries under *summary_key*.
    """
    if model is None:
        model = YOLO(model_size)
    results_scenes: list[dict] = []

    for s, scene in enumerate(scenes):
        new_scene = dict(scene)
        frames = scene.get(frame_key, [])
        yolo_dict: dict[int, list[dict]] = {}

        if use_bytetrack and frames:
            results = run_yolo_track_on_frames(
                model, frames, conf=conf, iou=iou, tracker=tracker
            )
            if results is not None:
                yolo_dict = parse_yolo_results(results, model)

        if not yolo_dict:
            for idx, frame in enumerate(frames):
                yolo_dict[idx] = run_yolo_on_frame(model, frame, conf=conf, iou=iou)

        if yolo_dict and not has_track_ids(yolo_dict):
            yolo_dict = assign_track_ids_iou(yolo_dict, iou_threshold=fallback_iou)

        if output_dir is not None:
            for idx, frame in enumerate(frames):
                debug_draw_yolo(
                    frame=frame,
                    detections=yolo_dict.get(idx, []),
                    save_path=f"./{output_dir}/scene_{s:03d}/detection_{idx:03d}.jpg",
                )

        new_scene[summary_key] = build_track_summaries(
            frames, yolo_dict, **track_kwargs
        )
        results_scenes.append(new_scene)

        if debug:
            lines = format_track_summaries(new_scene[summary_key], style="compact")
            print_prefixed("(YOLOv8)", f"Scene {s}:")
            for line in lines or ["none detected"]:
                print_prefixed("(YOLOv8)", line, indent=4)

    return results_scenes

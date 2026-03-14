"""YOLOv8 object detection and tracking per scene (orchestrator)."""

from ultralytics import YOLO

from kairos.core.utils import print_prefixed
from kairos.video.yolo_inference import run_yolo_on_frame, run_yolo_track_on_frames, parse_yolo_results
from kairos.video.tracking import has_track_ids, assign_track_ids_iou
from kairos.video.track_summary import build_track_summaries, format_track_summaries
from kairos.video.debug_draw import debug_draw_yolo


def detect_object_yolo(
    scenes: list,
    model_size: str = "models/yolov8s.pt",
    model=None,
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
    if model is None:
        model = YOLO(model_size)
    results_scenes = []

    for s, scene in enumerate(scenes):
        new_scene = dict(scene)
        frames = scene.get(frame_key, [])
        yolo_dict = {}

        if use_bytetrack and frames:
            results = run_yolo_track_on_frames(model, frames, conf=conf, iou=iou, tracker=tracker)
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

        new_scene[summary_key] = build_track_summaries(frames, yolo_dict, **track_kwargs)
        results_scenes.append(new_scene)

        if debug:
            lines = format_track_summaries(new_scene[summary_key], style="compact")
            print_prefixed("(YOLOv8)", f"Scene {s}:")
            for line in (lines or ["none detected"]):
                print_prefixed("(YOLOv8)", line, indent=4)

    return results_scenes

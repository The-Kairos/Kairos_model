"""Frame sampling from video scenes at fixed counts or FPS."""

import os

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, new_size: int = 320) -> np.ndarray:
    """Resize so the longest side equals *new_size*, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = new_size / max(w, h)
    return cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _read_frames_at_positions(
    input_video_path: str,
    frame_positions: list[int],
    new_size: int = 320,
) -> list[np.ndarray]:
    """Open a video, seek to each position, read and resize the frame."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = sorted(
        set(max(0, min(pos, total_frames - 1)) for pos in frame_positions)
    )

    frames: list[np.ndarray] = []
    for frame_num in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames.append(resize_frame(frame, new_size))

    cap.release()
    return frames


def _save_scene_frames(
    frames: list[np.ndarray],
    scene_index: int,
    output_dir: str,
) -> list[str]:
    """Save frames to disk and return their paths."""
    scene_folder = os.path.join(output_dir, f"scene_{scene_index:03d}")
    os.makedirs(scene_folder, exist_ok=True)
    paths = []
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(scene_folder, f"frame_{idx:02d}.jpg")
        cv2.imwrite(frame_path, frame)
        paths.append(frame_path)
    return paths


def sample_from_clip(
    input_video_path: str,
    scene_index: int,
    start_seconds: float,
    end_seconds: float,
    num_frames: int = 5,
    new_size: int = 320,
) -> list[np.ndarray]:
    """Sample *num_frames* evenly-spaced frames from a scene interval."""
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = max(0, min(round(start_seconds * fps), total_frames - 1))
    end_frame = max(0, min(round(end_seconds * fps), total_frames - 1))

    if end_frame <= start_frame or num_frames <= 1:
        frame_positions = [start_frame]
    else:
        total_range = end_frame - start_frame
        gap = total_range / (num_frames + 1)
        frame_positions = [round(start_frame + i * gap) for i in range(num_frames)]

    return _read_frames_at_positions(input_video_path, frame_positions, new_size)


def sample_from_clip_fps(
    input_video_path: str,
    scene_index: int,
    start_seconds: float,
    end_seconds: float,
    fps: float = 4.0,
    new_size: int = 320,
    return_meta: bool = False,
):
    """Sample frames from a scene interval at a fixed FPS."""
    cap = cv2.VideoCapture(input_video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = max(0, min(round(start_seconds * video_fps), total_frames - 1))
    end_frame = max(0, min(round(end_seconds * video_fps), total_frames - 1))

    if end_frame <= start_frame or fps <= 0:
        frame_positions = [start_frame]
    else:
        step = 1.0 / fps
        times = list(np.arange(start_seconds, end_seconds, step))
        if not times:
            times = [start_seconds]
        frame_positions = [round(t * video_fps) for t in times]

    frames = _read_frames_at_positions(input_video_path, frame_positions, new_size)

    if return_meta:
        # Reconstruct indices/timestamps from the positions that were actually read
        positions = sorted(
            set(max(0, min(pos, total_frames - 1)) for pos in frame_positions)
        )
        # Only keep as many as frames we got back
        positions = positions[: len(frames)]
        frame_indices = list(positions)
        frame_timestamps = [pos / video_fps if video_fps else 0.0 for pos in positions]
        return frames, frame_indices, frame_timestamps
    return frames


def sample_frames(
    input_video_path: str,
    scenes: list[dict],
    num_frames: int = 4,
    new_size: int = 320,
    output_dir: str | None = None,
) -> list[dict]:
    """Loop over scenes and attach sampled frames to each."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    enriched_scenes: list[dict] = []

    for scene in scenes:
        frames = sample_from_clip(
            input_video_path=input_video_path,
            scene_index=scene["scene_index"],
            start_seconds=scene["start_seconds"],
            end_seconds=scene["end_seconds"],
            num_frames=num_frames,
            new_size=new_size,
        )

        frame_paths = (
            _save_scene_frames(frames, scene["scene_index"], output_dir)
            if output_dir
            else None
        )

        new_scene = dict(scene)
        new_scene["frames"] = frames
        new_scene["frame_paths"] = frame_paths
        enriched_scenes.append(new_scene)

    return enriched_scenes


def sample_fps(
    input_video_path: str,
    scenes: list[dict],
    fps: float = 4.0,
    new_size: int = 320,
    output_dir: str | None = None,
    frames_key: str = "frames",
    frame_paths_key: str = "frame_paths",
    store_paths: bool = False,
    store_meta: bool = False,
) -> list[dict]:
    """Loop over scenes and attach frames sampled at a fixed FPS."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    enriched_scenes: list[dict] = []

    for scene in scenes:
        if store_meta:
            frames, frame_indices, frame_timestamps = sample_from_clip_fps(
                input_video_path=input_video_path,
                scene_index=scene["scene_index"],
                start_seconds=scene["start_seconds"],
                end_seconds=scene["end_seconds"],
                fps=fps,
                new_size=new_size,
                return_meta=True,
            )
        else:
            frames = sample_from_clip_fps(
                input_video_path=input_video_path,
                scene_index=scene["scene_index"],
                start_seconds=scene["start_seconds"],
                end_seconds=scene["end_seconds"],
                fps=fps,
                new_size=new_size,
                return_meta=False,
            )
            frame_indices = None
            frame_timestamps = None

        frame_paths: list[str] | None = None
        if output_dir is not None:
            saved_paths = _save_scene_frames(frames, scene["scene_index"], output_dir)
            if store_paths:
                frame_paths = saved_paths

        new_scene = dict(scene)
        new_scene[frames_key] = frames
        if store_paths:
            new_scene[frame_paths_key] = frame_paths
        if store_meta:
            new_scene["frame_indices"] = frame_indices
            new_scene["frame_timestamps"] = frame_timestamps
            new_scene["sample_fps"] = fps
        enriched_scenes.append(new_scene)

    return enriched_scenes

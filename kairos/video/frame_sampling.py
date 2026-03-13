"""Frame sampling from video scenes at fixed counts or FPS."""

import os
from typing import Optional

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, new_size: int = 320) -> np.ndarray:
    """Resize so the longest side equals *new_size*, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = new_size / max(w, h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


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
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, min(int(round(start_seconds * fps)), total_frames - 1))
    end_frame = max(0, min(int(round(end_seconds * fps)), total_frames - 1))

    if end_frame <= start_frame:
        frame_positions = [start_frame]
    elif num_frames <= 1:
        frame_positions = [start_frame]
    else:
        total_range = end_frame - start_frame
        gap = total_range / (num_frames + 1)
        frame_positions = [int(round(start_frame + i * gap)) for i in range(num_frames)]

    frame_positions = sorted(
        set(max(0, min(pos, total_frames - 1)) for pos in frame_positions)
    )

    frames: list[np.ndarray] = []
    for frame_num in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames.append(resize_frame(frame, new_size))

    cap.release()
    return frames


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
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, min(int(round(start_seconds * video_fps)), total_frames - 1))
    end_frame = max(0, min(int(round(end_seconds * video_fps)), total_frames - 1))

    if end_frame <= start_frame or fps <= 0:
        frame_positions = [start_frame]
    else:
        step = 1.0 / fps
        times = list(np.arange(start_seconds, end_seconds, step))
        if not times:
            times = [start_seconds]
        frame_positions = [int(round(t * video_fps)) for t in times]

    frame_positions = sorted(
        set(max(0, min(pos, total_frames - 1)) for pos in frame_positions)
    )

    frames: list[np.ndarray] = []
    frame_indices: Optional[list[int]] = [] if return_meta else None
    frame_timestamps: Optional[list[float]] = [] if return_meta else None

    for frame_num in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frames.append(resize_frame(frame, new_size))
        if return_meta:
            frame_indices.append(frame_num)
            frame_timestamps.append(frame_num / video_fps if video_fps else 0.0)

    cap.release()
    if return_meta:
        return frames, frame_indices, frame_timestamps
    return frames


def sample_frames(
    input_video_path: str,
    scenes: list[dict],
    num_frames: int = 4,
    new_size: int = 320,
    output_dir: Optional[str] = None,
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

        frame_paths: Optional[list[str]] = None
        if output_dir is not None:
            scene_folder = os.path.join(output_dir, f"scene_{scene['scene_index']:03d}")
            os.makedirs(scene_folder, exist_ok=True)
            frame_paths = []
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(scene_folder, f"frame_{idx:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

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
    output_dir: Optional[str] = None,
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

        frame_paths: Optional[list[str]] = None
        if output_dir is not None:
            scene_folder = os.path.join(output_dir, f"scene_{scene['scene_index']:03d}")
            os.makedirs(scene_folder, exist_ok=True)
            if store_paths:
                frame_paths = []
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(scene_folder, f"frame_{idx:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                if store_paths:
                    frame_paths.append(frame_path)

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

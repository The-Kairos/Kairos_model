"""Frame sampling from video scenes at fixed counts or FPS."""

import os

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, new_size: int = 320) -> np.ndarray:
    """Resize a frame so the longest side equals *new_size*, preserving aspect ratio.

    Args:
        frame: The input BGR image as a NumPy array.
        new_size: The desired length (in pixels) of the longest side after
            resizing.

    Returns:
        The resized image as a NumPy array.
    """
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
    """Open a video, seek to each position, read and resize the frame.

    Duplicate and out-of-range positions are clamped and deduplicated
    before reading.

    Args:
        input_video_path: Path to the input video file.
        frame_positions: A list of zero-based frame indices to read.
        new_size: Passed to :func:`resize_frame` as the target longest
            side.

    Returns:
        A list of resized frames (NumPy arrays) for positions that were
        successfully read.

    Raises:
        ValueError: If the video file cannot be opened.
    """
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
    """Save frames to disk and return their file paths.

    Frames are saved as JPEG files inside a per-scene sub-directory named
    ``scene_<NNN>`` under *output_dir*.

    Args:
        frames: List of BGR image arrays to save.
        scene_index: Numeric index of the scene, used to name the
            sub-directory.
        output_dir: Root directory under which scene folders are created.

    Returns:
        A list of absolute file paths for the saved images.
    """
    scene_folder = os.path.join(output_dir, f"scene_{scene_index:03d}")
    os.makedirs(scene_folder, exist_ok=True)
    paths: list[str] = []
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
    """Sample *num_frames* evenly-spaced frames from a scene interval.

    Frame positions are distributed across the interval
    ``[start_seconds, end_seconds]`` with equal spacing.

    Args:
        input_video_path: Path to the input video file.
        scene_index: Index of the scene (currently informational only).
        start_seconds: Start time of the scene in seconds.
        end_seconds: End time of the scene in seconds.
        num_frames: Number of frames to sample.
        new_size: Passed to :func:`resize_frame` as the target longest
            side.

    Returns:
        A list of resized sampled frames (NumPy arrays).
    """
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
) -> list[np.ndarray] | tuple[list[np.ndarray], list[int], list[float]]:
    """Sample frames from a scene interval at a fixed FPS.

    Args:
        input_video_path: Path to the input video file.
        scene_index: Index of the scene (currently informational only).
        start_seconds: Start time of the scene in seconds.
        end_seconds: End time of the scene in seconds.
        fps: Target sampling rate in frames per second.
        new_size: Passed to :func:`resize_frame` as the target longest
            side.
        return_meta: If ``True``, also return frame indices and
            timestamps alongside the frames.

    Returns:
        If *return_meta* is ``False``, a list of resized frames (NumPy
        arrays).  If ``True``, a tuple of
        ``(frames, frame_indices, frame_timestamps)`` where
        *frame_indices* is a list of zero-based frame numbers and
        *frame_timestamps* is a list of corresponding seconds.
    """
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
    """Loop over scenes and attach sampled frames to each.

    For every scene dictionary the function adds a ``"frames"`` key
    (list of NumPy arrays) and, when *output_dir* is provided, a
    ``"frame_paths"`` key (list of saved file paths).

    Args:
        input_video_path: Path to the input video file.
        scenes: A list of scene dictionaries, each containing at least
            ``"scene_index"``, ``"start_seconds"``, and
            ``"end_seconds"``.
        num_frames: Number of frames to sample per scene.
        new_size: Passed to :func:`resize_frame` as the target longest
            side.
        output_dir: If provided, sampled frames are saved to disk inside
            this directory.

    Returns:
        A new list of scene dictionaries augmented with ``"frames"`` and
        optionally ``"frame_paths"``.
    """
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
    """Loop over scenes and attach frames sampled at a fixed FPS.

    Args:
        input_video_path: Path to the input video file.
        scenes: A list of scene dictionaries, each containing at least
            ``"scene_index"``, ``"start_seconds"``, and
            ``"end_seconds"``.
        fps: Target sampling rate in frames per second.
        new_size: Passed to :func:`resize_frame` as the target longest
            side.
        output_dir: If provided, sampled frames are saved to disk inside
            this directory.
        frames_key: Dictionary key under which sampled frames are stored
            in each scene.
        frame_paths_key: Dictionary key under which saved file paths are
            stored (only when *store_paths* is ``True``).
        store_paths: If ``True`` and *output_dir* is set, include saved
            file paths in each scene dictionary.
        store_meta: If ``True``, include ``"frame_indices"``,
            ``"frame_timestamps"``, and ``"sample_fps"`` in each scene
            dictionary.

    Returns:
        A new list of scene dictionaries augmented with sampled frames
        and optional metadata.
    """
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

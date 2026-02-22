from __future__ import annotations

import os
import socket
import sys
from urllib.parse import urlparse
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRAME_BOUNDARY_DIR = Path(__file__).resolve().parent / "frame_boundaries"
DEFAULT_VIDEO = PROJECT_ROOT / "Videos" / "Young Sheldon - First Day of High School.mp4"
HF_CACHE_DIR = PROJECT_ROOT / "test" / "pyscene_alt" / ".hf_cache"
HF_HUB_DIR = HF_CACHE_DIR / "hub"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_DIR))
if HF_HUB_DIR.exists():
    os.environ.setdefault("KAIROS_HF_LOCAL_ONLY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from src.debug_utils import format_timecode as _format_timecode
except Exception:  # pragma: no cover - fallback for standalone runs
    def _format_timecode(seconds: float | None) -> str:
        if seconds is None:
            return "??:??:??.???"
        try:
            ms_total = int(round(float(seconds) * 1000))
        except (TypeError, ValueError):
            return "??:??:??.???"
        sec_total, ms = divmod(ms_total, 1000)
        mins_total, sec = divmod(sec_total, 60)
        hrs, mins = divmod(mins_total, 60)
        return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"


def format_timecode(seconds: float | None) -> str:
    return _format_timecode(seconds)


def sanitize_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned.strip().strip(".")


def to_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _env_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def hf_local_only() -> bool:
    return (
        _env_truthy(os.getenv("HF_HUB_OFFLINE"))
        or _env_truthy(os.getenv("TRANSFORMERS_OFFLINE"))
        or _env_truthy(os.getenv("KAIROS_HF_LOCAL_ONLY"))
    )

_HF_OFFLINE_STATE: bool | None = None


def _proxy_host_port(value: str) -> tuple[str | None, int | None]:
    proxy_url = value.strip()
    if "://" not in proxy_url:
        proxy_url = f"http://{proxy_url}"
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def _proxy_unreachable(timeout: float) -> bool:
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if not proxy:
        return False
    host, port = _proxy_host_port(proxy)
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return False
    except OSError:
        return True


def _direct_reachable(host: str, timeout: float) -> bool:
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_hf_offline_if_unreachable(
    host: str = "huggingface.co",
    timeout: float = 0.5,
) -> bool:
    global _HF_OFFLINE_STATE
    if _HF_OFFLINE_STATE is not None:
        return _HF_OFFLINE_STATE
    if hf_local_only():
        _HF_OFFLINE_STATE = True
        return True
    if _proxy_unreachable(timeout):
        if _direct_reachable(host, timeout):
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("HTTP_PROXY", None)
            _HF_OFFLINE_STATE = False
            return False
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("KAIROS_HF_LOCAL_ONLY", "1")
        _HF_OFFLINE_STATE = True
        return True
    if _direct_reachable(host, timeout):
        _HF_OFFLINE_STATE = False
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("KAIROS_HF_LOCAL_ONLY", "1")
    _HF_OFFLINE_STATE = True
    return True


def get_video_info(video_path: str | Path) -> Tuple[float, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration = (frame_count / fps) if fps > 0 else 0.0
    return fps, frame_count, duration


def _resize_frame(frame, new_size: int):
    from src.frame_sampling import resize_frame

    return resize_frame(frame, new_size=new_size)


def sample_video_frames(
    video_path: str | Path,
    sample_fps: float,
    new_size: int,
) -> Tuple[List["Image.Image"], List[float], dict]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    if sample_fps <= 0:
        sample_fps = max(fps, 1.0)

    step = 1.0 / sample_fps
    next_time = 0.0

    frames: List[Image.Image] = []
    timestamps: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        time_sec = (frame_idx / fps) if fps > 0 else 0.0
        if time_sec + 1e-6 >= next_time:
            resized = _resize_frame(frame, new_size=new_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(time_sec)
            next_time += step
        frame_idx += 1

    cap.release()

    info = {"fps": fps, "frame_count": frame_count, "duration": duration}
    return frames, timestamps, info


def load_frames_at_times(
    video_path: str | Path,
    times: Iterable[float],
    new_size: int,
) -> List["Image.Image"]:
    import cv2
    from PIL import Image

    times_list = list(times)
    if not times_list:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[Image.Image] = []
    missing: List[int] = []
    for idx, t in enumerate(times_list):
        frame_idx = int(round(t * fps)) if fps > 0 else 0
        frame_idx = max(0, min(frame_idx, frame_count - 1)) if frame_count > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            missing.append(idx)
            continue
        resized = _resize_frame(frame, new_size=new_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    if missing:
        raise RuntimeError(f"Failed to read {len(missing)} frame(s) from video.")
    return frames


def _read_frame_at_time(
    cap,
    time_sec: float,
    fps: float,
    frame_count: int,
    new_size: int,
):
    import cv2

    if fps <= 0:
        frame_idx = 0
    else:
        frame_idx = int(round(time_sec * fps))
    if frame_count > 0:
        frame_idx = max(0, min(frame_idx, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return _resize_frame(frame, new_size=new_size)


def save_scene_boundary_frames(
    video_path: str | Path,
    scenes: List[dict],
    test_name: str,
    new_size: int,
    output_dir: Path | None = None,
) -> List[dict]:
    import cv2

    if output_dir is None:
        output_dir = FRAME_BOUNDARY_DIR

    video_path = Path(video_path)
    video_stem = sanitize_filename(video_path.stem)
    test_name = sanitize_filename(test_name)

    base_dir = Path(output_dir) / video_stem / test_name
    base_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    updated: List[dict] = []
    for idx, scene in enumerate(scenes):
        start_time = float(scene.get("start_seconds", 0.0))
        end_time = float(scene.get("end_seconds", start_time))
        mid_time = (start_time + end_time) / 2.0 if end_time >= start_time else start_time

        middle_frame = _read_frame_at_time(cap, mid_time, fps, frame_count, new_size)
        middle_path = None
        if middle_frame is not None:
            middle_filename = f"scene{idx}.jpg"
            middle_path = base_dir / middle_filename
            cv2.imwrite(str(middle_path), middle_frame)

        scene_out = dict(scene)
        middle_rel = to_relative(middle_path) if middle_path else None
        scene_out["middle_frame"] = middle_rel
        scene_out["start_frame"] = middle_rel
        scene_out["end_frame"] = middle_rel
        updated.append(scene_out)

    cap.release()
    return updated


def finalize_scene_times(scenes: List[dict]) -> List[dict]:
    finalized: List[dict] = []
    for idx, scene in enumerate(scenes):
        start_seconds = float(scene.get("start_seconds", 0.0))
        end_seconds = float(scene.get("end_seconds", start_seconds))
        duration = max(0.0, end_seconds - start_seconds)
        scene_out = dict(scene)
        scene_out["scene_index"] = idx + 1
        scene_out["start_timecode"] = format_timecode(start_seconds)
        scene_out["end_timecode"] = format_timecode(end_seconds)
        scene_out["duration_seconds"] = duration
        finalized.append(scene_out)
    return finalized

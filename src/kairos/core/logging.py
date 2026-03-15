"""Pipeline step logging: hardware metrics, GPU stats, and timing decorator.

This module provides infrastructure for recording pipeline run metadata
including hardware context (CPU, RAM, disk, GPU), per-step resource
consumption, and wall-clock timing.  The :func:`log_step` decorator
wraps any pipeline function so that it returns both its normal output
**and** a dictionary of resource metrics collected before and after
execution.

Typical usage::

    from kairos.core.logging import initiate_log, complete_log, save_log, log_step

    log = initiate_log("/data/video.mp4", "nightly run", params={...})

    @log_step()
    def detect_scenes(video_path: str) -> list:
        ...

    scenes, step_log = detect_scenes("/data/video.mp4")
    full_log = complete_log(log, {"detect_scenes": step_log}, "00:12:34", len(scenes))
    save_log(full_log, "logs/run")
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import psutil
import torch

from kairos.core.utils import print_prefixed

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

P = ParamSpec("P")
T = TypeVar("T")


def get_system_context() -> dict[str, Any]:
    """Return a summary of the current hardware, OS, and GPU environment.

    The returned dictionary contains the following top-level keys:

    * ``os_info`` — operating system name, version, and architecture.
    * ``cpu_info`` — processor model, core counts, and clock frequency.
    * ``ram_info`` — total, available, and used RAM in gigabytes.
    * ``disk_info`` — total, used, and free disk space in gigabytes.
    * ``gpu_info`` — GPU model, VRAM usage, and driver version (via
      ``nvidia-smi``).  If no NVIDIA GPU is available the ``gpu_model``
      value is ``None``.

    Returns:
        A nested dictionary of system information suitable for JSON
        serialisation.
    """
    uname = platform.uname()
    system: dict[str, Any] = {
        "os": f"{uname.system} {uname.release}",
        "os_version": uname.version,
        "machine_type": uname.machine,
        "hostname": uname.node,
        "python_version": sys.version.split()[0],
    }

    cpu_info: dict[str, Any] = {
        "cpu_model": uname.processor or platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_frequency_MHz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
    }

    svmem = psutil.virtual_memory()
    ram_info: dict[str, Any] = {
        "total_RAM_GB": round(svmem.total / (1024**3), 2),
        "available_RAM_GB": round(svmem.available / (1024**3), 2),
        "used_RAM_GB": round(svmem.used / (1024**3), 2),
        "RAM_usage_percent": svmem.percent,
    }

    disk = psutil.disk_usage("/")
    disk_info: dict[str, Any] = {
        "disk_total_GB": round(disk.total / (1024**3), 2),
        "disk_used_GB": round(disk.used / (1024**3), 2),
        "disk_free_GB": round(disk.free / (1024**3), 2),
        "disk_usage_percent": disk.percent,
    }

    try:
        gpu_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        gpu_name, mem_total, mem_used, driver = gpu_output.strip().split(", ")
        gpu_info: dict[str, Any] = {
            "gpu_model": gpu_name,
            "gpu_memory_total_MB": int(mem_total),
            "gpu_memory_used_MB": int(mem_used),
            "gpu_driver_version": driver,
        }
    except Exception:
        gpu_info = {"gpu_model": None}

    return {
        "os_info": system,
        "cpu_info": cpu_info,
        "ram_info": ram_info,
        "disk_info": disk_info,
        "gpu_info": gpu_info,
    }


def get_gpu_stats() -> list[dict[str, Any]]:
    """Return per-GPU utilisation statistics.

    The function tries three strategies in order:

    1. **NVML** (``pynvml``) — provides the most detailed stats
       including GPU and memory utilisation percentages.
    2. **PyTorch CUDA** — falls back to ``torch.cuda`` for memory
       figures when NVML is unavailable.  Utilisation percentages are
       reported as ``None`` in this case.
    3. **Empty list** — returned when no GPU information can be
       obtained at all.

    Returns:
        A list of dictionaries, one per GPU, each containing:

        * ``id`` — GPU index.
        * ``name`` — device name string.
        * ``memory_used_MB`` / ``memory_total_MB`` — VRAM figures.
        * ``gpu_util_percent`` / ``mem_util_percent`` — utilisation
          percentages (may be ``None``).
    """
    if _NVML_AVAILABLE:
        gpus: list[dict[str, Any]] = []
        try:
            count: int = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                gpus.append(
                    {
                        "id": i,
                        "name": name,
                        "memory_used_MB": mem.used // (1024**2),
                        "memory_total_MB": mem.total // (1024**2),
                        "gpu_util_percent": util.gpu,
                        "mem_util_percent": util.memory,
                    }
                )
            return gpus
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            return [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_used_MB": torch.cuda.memory_allocated(i) // (1024**2),
                    "memory_total_MB": torch.cuda.get_device_properties(i).total_mem
                    // (1024**2),
                    "gpu_util_percent": None,
                    "mem_util_percent": None,
                }
                for i in range(torch.cuda.device_count())
            ]
        except Exception:
            pass

    return []


def initiate_log(
    video_path: str, run_description: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create the initial log dictionary at the start of a pipeline run.

    Records the start timestamp, video path, run description, caller-
    supplied parameters, and a snapshot of the current system context
    (see :func:`get_system_context`).

    Args:
        video_path: Filesystem path to the source video being processed.
        run_description: Free-form human-readable label for the run
            (e.g. ``"nightly regression"``).
        params: Optional dictionary of pipeline parameters to store
            alongside the log.  Defaults to an empty ``dict``.

    Returns:
        A dictionary suitable for later completion via
        :func:`complete_log`.  Contains keys ``"run_description"``,
        ``"video_path"``, ``"start_process"`` (Unix timestamp),
        ``"computer"``, and ``"params"``.
    """
    return {
        "run_description": run_description,
        "video_path": video_path,
        "start_process": time.time(),
        "computer": get_system_context(),
        "params": params or {},
    }


def complete_log(
    log: dict[str, Any],
    steps: dict[str, dict[str, Any]],
    vid_len: str,
    scene_num: int,
    vid_df: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalise a pipeline log with timing totals and per-step details.

    Merges the initial log created by :func:`initiate_log` with
    aggregated step timings, refreshed system context, and optional
    video-level metadata.

    Args:
        log: The initial log dictionary returned by
            :func:`initiate_log`.
        steps: Mapping of step names to their individual log
            dictionaries (each must contain a ``"wall_time_sec"`` key).
        vid_len: Human-readable video duration string
            (e.g. ``"00:12:34"``).
        scene_num: Total number of scenes detected in the video.
        vid_df: Optional dictionary of additional video-level metadata
            (e.g. codec info, resolution) to merge into the log.

    Returns:
        A new dictionary containing the complete run log ready for
        serialisation.  The original *log* dictionary is **not**
        mutated.
    """
    new_log: dict[str, Any] = {
        "run_description": log["run_description"],
        "video_path": log["video_path"],
        "video_length": vid_len,
        "total_process_sec": sum(steps[s]["wall_time_sec"] for s in steps),
        "scene_number": scene_num,
        "start_process": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(log["start_process"])
        ),
        "end_process": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "computer": get_system_context(),
        "params": log["params"],
        "steps": steps,
    }
    if vid_df is not None:
        new_log.update(vid_df)
    return new_log


def save_log(data: dict[str, Any], path: str) -> str:
    """Save a JSON-serialisable dictionary to a timestamped file.

    A timestamp of the form ``YYYYMMDD_HHMMSS`` is appended to the
    base filename so that successive saves never overwrite each other.
    Parent directories are created automatically.

    Args:
        data: The dictionary to serialise.  All values must be
            JSON-serialisable.
        path: Destination path **without** the timestamp suffix.
            If no extension is provided, ``.json`` is used.

    Returns:
        The actual file path that was written (including the appended
        timestamp).
    """
    folder: str = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".json"
    path = f"{base}_{timestamp}{ext}"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print_prefixed("(Log)", f"Saved: {path}")
    return path


def log_step() -> Callable[[Callable[P, T]], Callable[P, tuple[T, dict[str, Any]]]]:
    """Decorator factory that wraps a function with resource logging.

    The decorated function's return value is augmented: instead of
    returning just its normal output it returns a ``(output, log_dict)``
    tuple where *log_dict* captures resource usage metrics collected
    **before** and **after** the call:

    * ``wall_time_sec`` / ``cpu_time_sec`` — elapsed real and CPU time.
    * ``ram_before_MB`` / ``ram_after_MB`` / ``ram_used_MB`` — RSS
      memory snapshots.
    * ``io_read_MB`` / ``io_write_MB`` — cumulative I/O delta.
    * ``gpu_before`` / ``gpu_after`` — per-GPU stats from
      :func:`get_gpu_stats`.
    * ``cuda_before_MB`` / ``cuda_after_MB`` / ``cuda_peak_MB`` —
      PyTorch CUDA memory figures (``None`` when no GPU is available).

    Returns:
        A parameterless decorator that can be applied to any callable.

    Example::

        @log_step()
        def detect_objects(frames: list) -> list:
            ...

        results, metrics = detect_objects(my_frames)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, tuple[T, dict[str, Any]]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, dict[str, Any]]:
            process = psutil.Process(os.getpid())

            cpu_before: float = time.process_time()
            ram_before: int = process.memory_info().rss // (1024**2)
            io_before = process.io_counters()
            gpu_before: list[dict[str, Any]] = get_gpu_stats()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                cuda_before: list[int] | None = [
                    torch.cuda.memory_allocated(i) // (1024**2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_before = None

            t0: float = time.time()
            output: T = func(*args, **kwargs)
            t1: float = time.time()

            cpu_after: float = time.process_time()
            ram_after: int = process.memory_info().rss // (1024**2)
            io_after = process.io_counters()
            gpu_after: list[dict[str, Any]] = get_gpu_stats()

            if torch.cuda.is_available():
                cuda_after: list[int] | None = [
                    torch.cuda.memory_allocated(i) // (1024**2)
                    for i in range(torch.cuda.device_count())
                ]
                cuda_peak: list[int] | None = [
                    torch.cuda.max_memory_allocated(i) // (1024**2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_after = cuda_peak = None

            log_entry: dict[str, Any] = {
                "wall_time_sec": round(t1 - t0, 5),
                "cpu_time_sec": round(cpu_after - cpu_before, 5),
                "ram_before_MB": ram_before,
                "ram_after_MB": ram_after,
                "ram_used_MB": ram_after - ram_before,
                "io_read_MB": (io_after.read_bytes - io_before.read_bytes) / (1024**2),
                "io_write_MB": (io_after.write_bytes - io_before.write_bytes)
                / (1024**2),
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "cuda_before_MB": cuda_before,
                "cuda_after_MB": cuda_after,
                "cuda_peak_MB": cuda_peak,
            }

            return output, log_entry

        return wrapper

    return decorator

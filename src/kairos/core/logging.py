"""Pipeline step logging: hardware metrics, GPU stats, and timing decorator."""

import functools
import json
import os
import platform
import subprocess
import sys
import time

import psutil
import torch

from kairos.core.utils import print_prefixed

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def get_system_context() -> dict:
    """Return a summary of the current hardware, OS, and GPU."""
    uname = platform.uname()
    system = {
        "os": f"{uname.system} {uname.release}",
        "os_version": uname.version,
        "machine_type": uname.machine,
        "hostname": uname.node,
        "python_version": sys.version.split()[0],
    }

    cpu_info = {
        "cpu_model": uname.processor or platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_frequency_MHz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
    }

    svmem = psutil.virtual_memory()
    ram_info = {
        "total_RAM_GB": round(svmem.total / (1024**3), 2),
        "available_RAM_GB": round(svmem.available / (1024**3), 2),
        "used_RAM_GB": round(svmem.used / (1024**3), 2),
        "RAM_usage_percent": svmem.percent,
    }

    disk = psutil.disk_usage("/")
    disk_info = {
        "disk_total_GB": round(disk.total / (1024**3), 2),
        "disk_used_GB": round(disk.used / (1024**3), 2),
        "disk_free_GB": round(disk.free / (1024**3), 2),
        "disk_usage_percent": disk.percent,
    }

    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        gpu_name, mem_total, mem_used, driver = gpu_output.strip().split(", ")
        gpu_info = {
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


def get_gpu_stats() -> list:
    """Return GPU stats via NVML, PyTorch CUDA fallback, or empty list."""
    if _NVML_AVAILABLE:
        gpus = []
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                gpus.append({
                    "id": i,
                    "name": name,
                    "memory_used_MB": mem.used // (1024**2),
                    "memory_total_MB": mem.total // (1024**2),
                    "gpu_util_percent": util.gpu,
                    "mem_util_percent": util.memory,
                })
            return gpus
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            return [{
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_used_MB": torch.cuda.memory_allocated(i) // (1024**2),
                "memory_total_MB": torch.cuda.get_device_properties(i).total_mem // (1024**2),
                "gpu_util_percent": None,
                "mem_util_percent": None,
            } for i in range(torch.cuda.device_count())]
        except Exception:
            pass

    return []


def initiate_log(video_path: str, run_description: str, params: dict | None = None) -> dict:
    return {
        "run_description": run_description,
        "video_path": video_path,
        "start_process": time.time(),
        "computer": get_system_context(),
        "params": params or {},
    }


def complete_log(log: dict, steps: dict, vid_len: str, scene_num: int, vid_df: dict | None = None) -> dict:
    new_log = {
        "run_description": log["run_description"],
        "video_path": log["video_path"],
        "video_length": vid_len,
        "total_process_sec": sum(steps[s]["wall_time_sec"] for s in steps),
        "scene_number": scene_num,
        "start_process": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log["start_process"])),
        "end_process": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "computer": get_system_context(),
        "params": log["params"],
        "steps": steps,
    }
    if vid_df is not None:
        new_log.update(vid_df)
    return new_log


def save_log(data: dict, path: str) -> str:
    """Save serializable data to a timestamped JSON file. Returns the saved path."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".json"
    path = f"{base}_{timestamp}{ext}"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print_prefixed("(Log)", f"Saved: {path}")
    return path


def log_step():
    """Decorator that logs CPU, RAM, GPU, IO, runtime and returns (output, log_dict)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())

            cpu_before = time.process_time()
            ram_before = process.memory_info().rss // (1024 ** 2)
            io_before = process.io_counters()
            gpu_before = get_gpu_stats()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                cuda_before = [
                    torch.cuda.memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_before = None

            t0 = time.time()
            output = func(*args, **kwargs)
            t1 = time.time()

            cpu_after = time.process_time()
            ram_after = process.memory_info().rss // (1024 ** 2)
            io_after = process.io_counters()
            gpu_after = get_gpu_stats()

            if torch.cuda.is_available():
                cuda_after = [
                    torch.cuda.memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
                cuda_peak = [
                    torch.cuda.max_memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_after = cuda_peak = None

            log_entry = {
                "wall_time_sec": round(t1 - t0, 5),
                "cpu_time_sec": round(cpu_after - cpu_before, 5),
                "ram_before_MB": ram_before,
                "ram_after_MB": ram_after,
                "ram_used_MB": ram_after - ram_before,
                "io_read_MB": (io_after.read_bytes - io_before.read_bytes) / (1024**2),
                "io_write_MB": (io_after.write_bytes - io_before.write_bytes) / (1024**2),
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "cuda_before_MB": cuda_before,
                "cuda_after_MB": cuda_after,
                "cuda_peak_MB": cuda_peak,
            }

            return output, log_entry
        return wrapper
    return decorator

"""
system_metrics.py - CPU/GPU/memory usage for benchmarking.
"""

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def get_system_usage():
    """Return dict with gpu_memory_mb, cpu_percent, memory_mb."""
    out = {}
    if psutil:
        try:
            out["cpu_percent"] = psutil.cpu_percent()
            out["memory_mb"] = psutil.virtual_memory().used / (1024 ** 2)
        except Exception:
            pass
    if torch and torch.cuda.is_available():
        out["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
    return out

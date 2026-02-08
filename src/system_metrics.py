import psutil
import time

def get_system_usage():
    """
    Return current system usage (non-destructive snapshot).

    Returns:
        dict containing:
            cpu_percent
            ram_percent
            ram_used_mb
            ram_available_mb
            timestamp
    """
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": mem.percent,
        "ram_used_mb": mem.used / (1024 * 1024),
        "ram_available_mb": mem.available / (1024 * 1024),
        "timestamp": time.time()
    }

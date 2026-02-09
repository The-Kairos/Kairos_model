import psutil
import time
import torch

def get_system_usage():
    """
    Return current system usage (CPU, RAM, GPU).
    """
    mem = psutil.virtual_memory()
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": mem.percent,
        "ram_used_gb": mem.used / (1024**3),
        "ram_total_gb": mem.total / (1024**3),
        "timestamp": time.time()
    }
    
    # GPU Metrics
    if torch.cuda.is_available():
        try:
            # global free, total
            free, total = torch.cuda.mem_get_info()
            metrics["gpu_used_gb"] = (total - free) / (1024**3)
            metrics["gpu_total_gb"] = total / (1024**3)
            metrics["gpu_percent"] = (metrics["gpu_used_gb"] / metrics["gpu_total_gb"]) * 100
        except:
            metrics["gpu_used_gb"] = 0
            metrics["gpu_total_gb"] = 0
            metrics["gpu_percent"] = 0
            
    return metrics

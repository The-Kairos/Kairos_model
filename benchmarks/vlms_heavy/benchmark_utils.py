import time
import torch
import psutil
import os

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def benchmark_inference(func, *args, **kwargs):
    """
    Measures duration and GPU memory usage for a given function call.
    """
    torch.cuda.empty_cache()
    start_mem = get_gpu_memory()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_mem = get_gpu_memory()
    
    metrics = {
        "duration_sec": end_time - start_time,
        "gpu_mem_used_mb": end_mem - start_mem,
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    }
    
    return result, metrics

def load_test_image(image_path):
    from PIL import Image
    return Image.open(image_path).convert("RGB")

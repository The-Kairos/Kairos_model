import time
import psutil
import os
import torch

class StageProfiler:
    def __init__(self, name: str):
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.metrics = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.mem_before = self.process.memory_info().rss

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.vram_before = torch.cuda.memory_allocated()
        else:
            self.vram_before = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated()
        else:
            peak_vram = 0

        self.t1 = time.perf_counter()
        self.mem_after = self.process.memory_info().rss

        self.metrics = {
            "stage": self.name,
            "time_s": self.t1 - self.t0,
            "ram_delta_mb": (self.mem_after - self.mem_before) / 1024 / 1024,
            "peak_vram_mb": peak_vram / 1024 / 1024,
        }

import time
from contextlib import contextmanager

@contextmanager
def timed(section_name: str, timings: dict):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    timings[section_name] = (end - start) * 1000  # milliseconds

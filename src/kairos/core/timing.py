"""Pipeline stage timing decorator and JSON timing report.

Usage::

    from kairos.core.timing import timed_stage, save_timing_report

    @timed_stage("scene_detection")
    def detect_scenes(video_path, **kwargs):
        ...

    # At the end of the pipeline:
    save_timing_report("output/timings.json")
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kairos.core.utils import print_prefixed

_timing_lock = threading.Lock()
_timing_records: list[dict[str, Any]] = []


def timed_stage(stage_name: str) -> Callable:
    """Decorator that logs wall-clock time for a pipeline stage.

    The timing is printed immediately and also accumulated in a
    module-level list that can be saved to disk via :func:`save_timing_report`.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as exc:
                success = False
                error_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter() - t0
                record = {
                    "stage": stage_name,
                    "function": func.__qualname__,
                    "wall_time_sec": round(elapsed, 4),
                    "success": success,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                if error_msg:
                    record["error"] = error_msg
                with _timing_lock:
                    _timing_records.append(record)

                status = "✓" if success else "✗"
                print_prefixed(
                    "(Timer)",
                    f"{status} {stage_name}: {elapsed:.2f}s",
                )

            return result

        return wrapper

    return decorator


def get_timing_records() -> list[dict[str, Any]]:
    """Return a copy of all accumulated timing records."""
    with _timing_lock:
        return list(_timing_records)


def clear_timing_records() -> None:
    """Clear all accumulated timing records."""
    with _timing_lock:
        _timing_records.clear()


def save_timing_report(path: str | Path) -> str:
    """Write all timing records to a JSON file and return the path."""
    records = get_timing_records()
    total = sum(r["wall_time_sec"] for r in records)
    report = {
        "total_wall_time_sec": round(total, 4),
        "stage_count": len(records),
        "stages": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print_prefixed("(Timer)", f"Timing report saved: {path}")
    return str(path)

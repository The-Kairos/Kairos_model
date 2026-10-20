"""Opt-in, thread-safe API usage records without prompt or response text."""
import json
import logging
import os
from pathlib import Path
import threading
import time
from functools import wraps
from datetime import datetime, timezone

_LOCK = threading.Lock()


def measured_stage(stage):
    """Measure wall time around a complete stage, including parallel work."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            error = None
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                record_metric({"kind": "stage", "stage": stage,
                               "elapsed_sec": time.perf_counter() - started,
                               "error_type": type(error).__name__ if error else None})
        return wrapped
    return decorate


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def record_metric(record):
    destination = os.getenv("KAIROS_LLM_METRICS_PATH")
    if not destination:
        return
    record = dict(record, run_id=os.getenv("KAIROS_LLM_RUN_ID"),
                  timestamp=datetime.now(timezone.utc).isoformat())
    try:
        with _LOCK:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
    except (OSError, ValueError, TypeError):
        logging.getLogger(__name__).warning("Could not write LLM metrics", exc_info=False)


def measured_chat_completion(client, *, stage, attempt=1, **kwargs):
    """One SDK call; SDK-internal retries are included in its elapsed time."""
    started = time.perf_counter()
    response = None
    error = None
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as exc:
        error = exc
        raise
    finally:
        usage = _field(response, "usage")
        choices = _field(response, "choices") or []
        record_metric({
            "kind": "request", "stage": stage, "attempt": attempt,
            "deployment": kwargs.get("model"), "model": _field(response, "model"),
            "elapsed_sec": time.perf_counter() - started,
            "prompt_tokens": _field(usage, "prompt_tokens"),
            "completion_tokens": _field(usage, "completion_tokens"),
            "cached_tokens": _field(_field(usage, "prompt_tokens_details"), "cached_tokens"),
            "finish_reason": _field(choices[0], "finish_reason") if choices else None,
            "error_type": type(error).__name__ if error else None,
            "status_code": getattr(error, "status_code", None),
        })

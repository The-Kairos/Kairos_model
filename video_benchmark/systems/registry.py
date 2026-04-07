from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from systems.base import BaseSystem


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_systems_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _bench_root() / "config" / "systems.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system(system_key: str, cfg: dict[str, Any] | None = None) -> BaseSystem:
    cfg = cfg or load_systems_config()
    systems = cfg.get("systems") or {}
    if system_key not in systems:
        raise KeyError(f"Unknown system {system_key!r}; defined: {list(systems)}")
    spec = systems[system_key]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    params = dict(spec.get("params") or {})
    return cls(name=system_key, **params)

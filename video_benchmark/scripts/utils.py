from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_yaml(cli_value: str | None) -> Path:
    root = bench_root()
    if not cli_value:
        return root / "config" / "dataset.yaml"
    name = cli_value.strip()
    p = Path(name)
    if p.is_absolute():
        return p
    if name.lower().endswith((".yaml", ".yml")):
        return root / p
    return root / "config" / f"{name}.yaml"


def data_dir() -> Path:
    return bench_root() / "data"


def outputs_dir() -> Path:
    return bench_root() / "outputs"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_video_path(entry: dict[str, Any]) -> Path:
    rel = entry.get("file") or entry.get("path")
    if not rel:
        raise ValueError(f"video_info entry missing file/path: {entry}")
    p = (data_dir() / rel).resolve()
    return p


def load_dataset_files(dataset_cfg: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    root = data_dir()
    q_path = root / dataset_cfg["queries_file"]
    a_path = root / dataset_cfg["ground_truth_file"]
    v_path = root / dataset_cfg["video_catalog_file"]
    queries = load_json(q_path)
    answers = load_json(a_path)
    videos = load_json(v_path)
    if not isinstance(queries, list):
        raise TypeError("queries.json must be a list")
    if not isinstance(answers, list):
        raise TypeError("answers.json must be a list")
    if not isinstance(videos, list):
        raise TypeError("video_info.json must be a list")
    return queries, answers, videos


def index_answers(answers: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for a in answers:
        key = (a.get("video_id", ""), a.get("question_id", ""))
        out[key] = a
    return out


def merge_query_ground_truth(queries: list[dict], answers: list[dict]) -> list[dict]:
    by_q = index_answers(answers)
    merged = []
    for q in queries:
        vid = q.get("video_id")
        qid = q.get("question_id")
        gold = by_q.get((vid, qid))
        row = {**q, "gold": gold}
        merged.append(row)
    return merged

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neo4j import GraphDatabase

from src.kg_neo4j import (
    DEFAULT_AUTH_CANDIDATES,
    NEO4J_URI,
    sanitize_video_database_name,
)


VIDEO_PATH = "Titanic.1997.mkv"
VARIANTS = ("full", "no_blip", "no_yolo", "no_asr", "no_ast")
NODE_CATEGORIES = ("Scene", "Character", "Object", "Location", "Action", "Emotion", "Topic")
RELATIONSHIP_TYPES = (
    "NEXT",
    "HAS_TOPIC",
    "IS_SHOWN_IN",
    "OCCURS_IN",
    "DOES",
    "INTERACTS_WITH",
    "INVOLVES",
    "TARGETS",
    "FEELS",
    "CAUSES",
    "SPEAKS_TO",
    "MENTIONS",
    "IN",
    "NEAR",
    "LEFT_OF",
    "RIGHT_OF",
    "IN_FRONT_OF",
    "BEHIND",
    "ON",
    "INSIDE",
    "CONTINUES_INTO",
    "AFTER",
    "BEFORE",
    "OVERLAPS",
)
ABLATION_ROOT = PROJECT_ROOT / "_processed_ablations" / "Titanic.1997.mkv"
SUMMARY_PATH = ABLATION_ROOT / "ablation_summary.json"
OUTPUT_CSV = ABLATION_ROOT / "graph_part_counts.csv"
OUTPUT_PERC_CSV = ABLATION_ROOT / "graph_part_counts_perc.csv"


def emit(message: str) -> None:
    print(message, flush=True)


def connect_driver():
    last_error = None
    for username, password in DEFAULT_AUTH_CANDIDATES:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(username, password))
            driver.verify_connectivity()
            return driver, username
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Neo4j connectivity failed for all auth candidates: {last_error}")


def load_variant_database_names() -> dict[str, str]:
    names = {}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            for entry in summary.get("variants", []):
                if not isinstance(entry, dict):
                    continue
                variant = entry.get("variant")
                database_name = (entry.get("neo4j", {}) or {}).get("database_name")
                if isinstance(variant, str) and isinstance(database_name, str) and database_name.strip():
                    names[variant] = database_name.strip()
        except Exception:
            pass

    for variant in VARIANTS:
        names.setdefault(
            variant,
            sanitize_video_database_name(f"{Path(VIDEO_PATH).name}__{variant}"),
        )
    return names


def count_label(driver, database_name: str, label: str) -> int:
    query = f"MATCH (n:{label}) RETURN count(n) AS count"
    records, _, _ = driver.execute_query(query, database_=database_name)
    return int(records[0]["count"]) if records else 0


def count_relationship_type(driver, database_name: str, rel_type: str) -> int:
    query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
    records, _, _ = driver.execute_query(query, database_=database_name)
    return int(records[0]["count"]) if records else 0


def format_delta_vs_full(value, full_value) -> str:
    try:
        current = int(value)
        baseline = int(full_value)
    except Exception:
        return str(value)
    if baseline == 0:
        return str(current)
    pct = ((current - baseline) / baseline) * 100.0
    return f"{current} ({pct:+.1f}%)"


def build_variant_counts(driver, database_name: str) -> dict[str, int]:
    counts = {}
    for label in NODE_CATEGORIES:
        counts[f"node_{label.lower()}"] = count_label(driver, database_name, label)
    for rel_type in RELATIONSHIP_TYPES:
        counts[f"rel_{rel_type.lower()}"] = count_relationship_type(driver, database_name, rel_type)
    return counts


def main():
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    variant_databases = load_variant_database_names()

    per_variant_counts: dict[str, dict[str, int | str]] = {}
    driver, username = connect_driver()
    emit(f"Connected to Neo4j at {NEO4J_URI} as {username}")
    try:
        for variant in VARIANTS:
            database_name = variant_databases[variant]
            emit(f"[count] {variant} -> {database_name}")
            counts: dict[str, int | str] = {"database_name": database_name}
            try:
                counts.update(build_variant_counts(driver, database_name))
            except Exception as exc:
                counts["error"] = f"{type(exc).__name__}: {exc}"
            per_variant_counts[variant] = counts
    finally:
        driver.close()

    narrative_relationships = (
        "NEXT",
        "HAS_TOPIC",
        "DOES",
        "INTERACTS_WITH",
        "INVOLVES",
        "TARGETS",
        "FEELS",
        "SPEAKS_TO",
        "MENTIONS",
    )
    spatial_relationships = (
        "IS_SHOWN_IN",
        "OCCURS_IN",
        "IN",
        "NEAR",
        "LEFT_OF",
        "RIGHT_OF",
        "IN_FRONT_OF",
        "BEHIND",
        "ON",
        "INSIDE",
    )
    temporal_relationships = (
        "CAUSES",
        "CONTINUES_INTO",
        "AFTER",
        "BEFORE",
        "OVERLAPS",
    )

    non_scene_node_keys = [
        f"node_{label.lower()}"
        for label in NODE_CATEGORIES
        if label != "Scene"
    ]
    narrative_row_keys = [f"rel_{rel_type.lower()}" for rel_type in narrative_relationships]
    spatial_row_keys = [f"rel_{rel_type.lower()}" for rel_type in spatial_relationships]
    temporal_row_keys = [f"rel_{rel_type.lower()}" for rel_type in temporal_relationships]

    ordered_rows = [
        {"label": "[meta]", "keys": ["database_name"]},
        {"label": "[nodes]", "keys": non_scene_node_keys},
    ]
    ordered_rows.extend(
        {"label": f"  -> node_{label.lower()}", "keys": [f"node_{label.lower()}"]}
        for label in NODE_CATEGORIES
        if label != "Scene"
    )
    ordered_rows.append({"label": "[relationships:narrative]", "keys": narrative_row_keys})
    ordered_rows.extend(
        {"label": f"  -> rel_{rel_type.lower()}", "keys": [f"rel_{rel_type.lower()}"]}
        for rel_type in narrative_relationships
    )
    ordered_rows.append({"label": "[relationships:spatial]", "keys": spatial_row_keys})
    ordered_rows.extend(
        {"label": f"  -> rel_{rel_type.lower()}", "keys": [f"rel_{rel_type.lower()}"]}
        for rel_type in spatial_relationships
    )
    ordered_rows.append({"label": "[relationships:temporal]", "keys": temporal_row_keys})
    ordered_rows.extend(
        {"label": f"  -> rel_{rel_type.lower()}", "keys": [f"rel_{rel_type.lower()}"]}
        for rel_type in temporal_relationships
    )
    ordered_rows.append({"label": "[meta:error]", "keys": ["error"]})

    fieldnames = ["part", *VARIANTS]
    raw_rows = []
    for row_spec in ordered_rows:
        row = {"part": row_spec["label"]}
        row_keys = row_spec["keys"]
        for variant in VARIANTS:
            if row_keys == ["database_name"] or row_keys == ["error"]:
                row[variant] = per_variant_counts.get(variant, {}).get(row_keys[0], "")
                continue
            values = [
                per_variant_counts.get(variant, {}).get(key, 0)
                for key in row_keys
            ]
            row[variant] = sum(int(value or 0) for value in values)
        raw_rows.append(row)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    perc_rows = []
    for row in raw_rows:
        perc_row = {"part": row["part"], "full": row.get("full", "")}
        full_value = row.get("full", "")
        for variant in VARIANTS[1:]:
            if row["part"] in {"[meta]", "[meta:error]"}:
                perc_row[variant] = row.get(variant, "")
            else:
                perc_row[variant] = format_delta_vs_full(row.get(variant, ""), full_value)
        perc_rows.append(perc_row)

    with OUTPUT_PERC_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(perc_rows)

    emit(f"Wrote counts CSV: {OUTPUT_CSV}")
    emit(f"Wrote percentage counts CSV: {OUTPUT_PERC_CSV}")


if __name__ == "__main__":
    main()

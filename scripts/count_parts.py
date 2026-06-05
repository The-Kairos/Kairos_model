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
VARIANTS = ("full", "no_yolo", "no_asr", "no_ast", "no_blip")
NODE_CATEGORIES = ("Character", "Object", "Location", "Action", "Emotion", "Topic")
RELATIONSHIP_TYPES = (
    "HAS_TOPIC",
    "IS_SHOWN_IN",
    "IS_IN",
    "OCCURS_IN",
    "DOES",
    "INTERACTS_WITH",
    "INVOLVES",
    "TARGETS",
    "FEELS",
    "CAUSES",
    "SPEAKS_TO",
    "MENTIONS",
)
ABLATION_ROOT = PROJECT_ROOT / "_processed_ablations" / "Titanic.1997.mkv"
SUMMARY_PATH = ABLATION_ROOT / "ablation_summary.json"
OUTPUT_CSV = ABLATION_ROOT / "graph_part_counts.csv"


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


def main():
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    variant_databases = load_variant_database_names()

    fieldnames = ["variant", "database_name"]
    fieldnames.extend(f"node_{label.lower()}" for label in NODE_CATEGORIES)
    fieldnames.extend(f"rel_{rel_type.lower()}" for rel_type in RELATIONSHIP_TYPES)
    fieldnames.extend(["status", "error"])

    rows = []
    driver, username = connect_driver()
    emit(f"Connected to Neo4j at {NEO4J_URI} as {username}")
    try:
        for variant in VARIANTS:
            database_name = variant_databases[variant]
            emit(f"[count] {variant} -> {database_name}")
            row = {
                "variant": variant,
                "database_name": database_name,
                "status": "success",
                "error": "",
            }
            try:
                for label in NODE_CATEGORIES:
                    row[f"node_{label.lower()}"] = count_label(driver, database_name, label)
                for rel_type in RELATIONSHIP_TYPES:
                    row[f"rel_{rel_type.lower()}"] = count_relationship_type(driver, database_name, rel_type)
            except Exception as exc:
                row["status"] = "failure"
                row["error"] = f"{type(exc).__name__}: {exc}"
                for field in fieldnames:
                    row.setdefault(field, "")
            rows.append(row)
    finally:
        driver.close()

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    emit(f"Wrote counts CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

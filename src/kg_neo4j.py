import re
import time
import json
from datetime import datetime, timezone
from pathlib import Path


NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_PASSWORD = "kairos_kg"
DEFAULT_AUTH_CANDIDATES = (
    ("neo4j", NEO4J_PASSWORD),
    ("kairos_kg", NEO4J_PASSWORD),
)
NODE_CATEGORIES = ("Character", "Object", "Location", "Action", "Emotion", "Topic")


def sanitize_video_database_name(video_path: str) -> str:
    raw_name = Path(video_path).name.casefold()
    raw_name = re.sub(r"\.(mp4|mkv|avi|mov|webm|m4v)$", "", raw_name)
    cleaned = re.sub(r"[^a-z0-9-]+", "-", raw_name)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = "video"
    if not cleaned[0].isalnum():
        cleaned = f"vid-{cleaned}"
    cleaned = cleaned[:63].rstrip("-.")
    if len(cleaned) < 3:
        cleaned = f"vid-{cleaned}".rstrip("-.")
    return cleaned[:63]


def _scene_id(scene: dict, fallback_idx: int) -> str:
    return f"scene:{scene.get('scene_index', fallback_idx)}"


def _build_scene_rows(scenes: list[dict]) -> list[dict]:
    rows = []
    for idx, scene in enumerate(scenes or []):
        scene_index = scene.get("scene_index", idx)
        scene_id = _scene_id(scene, idx)
        rows.append(
            {
                "id": scene_id,
                "name": f"Scene {scene_index}",
                "scene_index": scene_index,
                "start_timecode": scene.get("start_timecode"),
                "end_timecode": scene.get("end_timecode"),
                "frame_captions": scene.get("frame_captions", []),
                "yolo_detections": json.dumps(scene.get("yolo_detections", []), ensure_ascii=False),
                "audio_speech": scene.get("audio_speech", ""),
                "audio_natural": scene.get("audio_natural", ""),
                "llm_scene_description": scene.get("llm_scene_description", ""),
            }
        )
    return rows


def _build_scene_next_rows(scenes: list[dict]) -> list[dict]:
    rows = []
    for idx in range(len(scenes or []) - 1):
        source_scene = scenes[idx]
        target_scene = scenes[idx + 1]
        rows.append(
            {
                "source_id": _scene_id(source_scene, idx),
                "target_id": _scene_id(target_scene, idx + 1),
                "scene_id": _scene_id(source_scene, idx),
            }
        )
    return rows


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_") or "unknown"


def _node_id_for(category: str, label: str) -> str:
    prefix_map = {
        "Character": "char",
        "Object": "obj",
        "Location": "loc",
        "Action": "action",
        "Emotion": "emotion",
        "Topic": "topic",
    }
    return f"{prefix_map[category]}:{_slugify(label)}"


def _build_category_rows(known_nodes: dict | None) -> dict[str, list[dict]]:
    rows_by_category = {category: [] for category in NODE_CATEGORIES}
    if not isinstance(known_nodes, dict):
        return rows_by_category

    for category in NODE_CATEGORIES:
        values = known_nodes.get(category, [])
        if not isinstance(values, list):
            continue
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            label = value.strip()
            if not label:
                continue
            node_id = _node_id_for(category, label)
            if node_id in seen:
                continue
            seen.add(node_id)
            rows_by_category[category].append(
                {
                    "id": node_id,
                    "name": label,
                    "category": category,
                }
            )
    return rows_by_category


def _build_relationship_rows(scenes: list[dict]) -> dict[str, list[dict]]:
    rows_by_type: dict[str, list[dict]] = {}
    for idx, scene in enumerate(scenes or []):
        scene_id = _scene_id(scene, idx)
        for rel in scene.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            rel_type = rel.get("type")
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            if not all(isinstance(value, str) and value.strip() for value in (rel_type, source_id, target_id)):
                continue
            rows_by_type.setdefault(rel_type.strip().upper(), []).append(
                {
                    "source_id": source_id.strip(),
                    "target_id": target_id.strip(),
                    "scene_id": scene_id,
                }
            )
    return rows_by_type


def _connect_driver(uri: str, auth_candidates: tuple[tuple[str, str], ...]):
    from neo4j import GraphDatabase

    last_error = None
    for username, password in auth_candidates:
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
            return driver, username
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Neo4j connectivity failed for all auth candidates: {last_error}")


def _wait_for_database_online(driver, database_name: str, timeout_sec: float = 30.0) -> None:
    deadline = time.time() + timeout_sec
    query = (
        f"SHOW DATABASE `{database_name}` "
        "YIELD currentStatus "
        "RETURN currentStatus LIMIT 1"
    )
    while time.time() < deadline:
        records, _, _ = driver.execute_query(query, database_="system")
        if records:
            status = records[0].get("currentStatus")
            if isinstance(status, str) and status.casefold() == "online":
                return
        time.sleep(0.5)
    raise TimeoutError(f"Neo4j database did not come online in time: {database_name}")


def _merge_category_nodes(driver, database_name: str, category: str, rows: list[dict]) -> None:
    if not rows:
        return
    query = (
        f"UNWIND $rows AS row "
        f"MERGE (n:{category} {{id: row.id}}) "
        "SET n.name = row.name, "
        "    n.category = row.category"
    )
    driver.execute_query(query, rows=rows, database_=database_name)


def _merge_scene_nodes(driver, database_name: str, rows: list[dict]) -> None:
    if not rows:
        return
    query = (
        "UNWIND $rows AS row "
        "MERGE (s:Scene {id: row.id}) "
        "SET s.name = row.name, "
        "    s.scene_index = row.scene_index, "
        "    s.start_timecode = row.start_timecode, "
        "    s.end_timecode = row.end_timecode, "
        "    s.frame_captions = row.frame_captions, "
        "    s.yolo_detections = row.yolo_detections, "
        "    s.audio_speech = row.audio_speech, "
        "    s.audio_natural = row.audio_natural, "
        "    s.llm_scene_description = row.llm_scene_description"
    )
    driver.execute_query(query, rows=rows, database_=database_name)


def _merge_relationships(driver, database_name: str, rel_type: str, rows: list[dict]) -> None:
    if not rows:
        return
    query = (
        f"UNWIND $rows AS row "
        "MATCH (a {id: row.source_id}) "
        "MATCH (b {id: row.target_id}) "
        f"MERGE (a)-[r:{rel_type} {{scene_id: row.scene_id}}]->(b)"
    )
    driver.execute_query(query, rows=rows, database_=database_name)


def sync_video_graph_to_neo4j(
    video_path: str,
    scenes: list[dict],
    known_nodes: dict | None,
    uri: str = NEO4J_URI,
    database_name_seed: str | None = None,
) -> dict:
    database_name = sanitize_video_database_name(database_name_seed or video_path)
    driver, connected_user = _connect_driver(uri, DEFAULT_AUTH_CANDIDATES)

    try:
        driver.execute_query(
            f"CREATE OR REPLACE DATABASE `{database_name}`",
            database_="system",
        )
        _wait_for_database_online(driver, database_name)

        category_rows = _build_category_rows(known_nodes)
        scene_rows = _build_scene_rows(scenes)
        next_rows = _build_scene_next_rows(scenes)
        relationship_rows = _build_relationship_rows(scenes)

        _merge_scene_nodes(driver, database_name, scene_rows)
        for category, rows in category_rows.items():
            _merge_category_nodes(driver, database_name, category, rows)
        _merge_relationships(driver, database_name, "NEXT", next_rows)
        for rel_type, rows in relationship_rows.items():
            _merge_relationships(driver, database_name, rel_type, rows)

        return {
            "database_name": database_name,
            "uri": uri,
            "username": connected_user,
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scene_nodes": len(scene_rows),
            "scene_next_relationships": len(next_rows),
            "graph_nodes": sum(len(rows) for rows in category_rows.values()),
            "graph_relationships": sum(len(rows) for rows in relationship_rows.values()),
        }
    finally:
        driver.close()

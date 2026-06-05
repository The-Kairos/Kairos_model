import csv
import json
import sys
from collections import defaultdict, deque
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
ABLATION_ROOT = PROJECT_ROOT / "_processed_ablations" / "Titanic.1997.mkv"
SUMMARY_PATH = ABLATION_ROOT / "ablation_summary.json"
OUTPUT_CSV = ABLATION_ROOT / "graph_measures.csv"


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
                database_name = (
                    entry.get("neo4j", {}) or {}
                ).get("database_name")
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


def fetch_graph(driver, database_name: str):
    node_query = (
        "MATCH (n) "
        "RETURN coalesce(n.id, elementId(n)) AS node_id"
    )
    rel_query = (
        "MATCH (a)-[r]->(b) "
        "RETURN "
        "coalesce(a.id, elementId(a)) AS source_id, "
        "coalesce(b.id, elementId(b)) AS target_id"
    )

    node_records, _, _ = driver.execute_query(node_query, database_=database_name)
    rel_records, _, _ = driver.execute_query(rel_query, database_=database_name)

    node_ids = [record["node_id"] for record in node_records]
    raw_edges = [(record["source_id"], record["target_id"]) for record in rel_records]
    return node_ids, raw_edges


def build_graph_structures(node_ids: list[str], raw_edges: list[tuple[str, str]]):
    nodes = list(dict.fromkeys(node_ids))
    node_set = set(nodes)

    simple_edges = set()
    out_adj = {node: set() for node in nodes}
    in_adj = {node: set() for node in nodes}
    undirected_adj = {node: set() for node in nodes}

    for source, target in raw_edges:
        if source not in node_set or target not in node_set:
            continue
        if (source, target) not in simple_edges:
            simple_edges.add((source, target))
            out_adj[source].add(target)
            in_adj[target].add(source)
            undirected_adj[source].add(target)
            undirected_adj[target].add(source)

    return nodes, simple_edges, out_adj, in_adj, undirected_adj


def weakly_connected_components(nodes, undirected_adj):
    seen = set()
    components = []
    for node in nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in undirected_adj[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def strongly_connected_components(nodes, out_adj, in_adj):
    seen = set()
    finish_order = []

    def dfs_forward(start):
        stack = [(start, 0)]
        seen.add(start)
        while stack:
            node, state = stack.pop()
            if state == 0:
                stack.append((node, 1))
                for neighbor in out_adj[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append((neighbor, 0))
            else:
                finish_order.append(node)

    for node in nodes:
        if node not in seen:
            dfs_forward(node)

    seen.clear()
    components = []

    for node in reversed(finish_order):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in in_adj[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    return components


def bfs_distances(source, adjacency):
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        base_distance = distances[current]
        for neighbor in adjacency[current]:
            if neighbor in distances:
                continue
            distances[neighbor] = base_distance + 1
            queue.append(neighbor)
    return distances


def average_shortest_path_and_diameter(gscc_nodes, out_adj):
    if len(gscc_nodes) < 2:
        return 0.0, 0

    total_distance = 0
    pair_count = 0
    diameter = 0

    for source in gscc_nodes:
        distances = bfs_distances(source, out_adj)
        for target in gscc_nodes:
            if source == target:
                continue
            distance = distances.get(target)
            if distance is None:
                continue
            total_distance += distance
            pair_count += 1
            if distance > diameter:
                diameter = distance

    if pair_count == 0:
        return 0.0, 0
    return total_distance / pair_count, diameter


def global_efficiency(nodes, out_adj):
    n = len(nodes)
    if n < 2:
        return 0.0

    total = 0.0
    for source in nodes:
        distances = bfs_distances(source, out_adj)
        for target, distance in distances.items():
            if source == target or distance <= 0:
                continue
            total += 1.0 / distance
    return total / (n * (n - 1))


def reciprocity(simple_edges):
    if not simple_edges:
        return 0.0
    reciprocated = 0
    for source, target in simple_edges:
        if (target, source) in simple_edges:
            reciprocated += 1
    return reciprocated / len(simple_edges)


def compute_metrics(node_ids: list[str], raw_edges: list[tuple[str, str]]) -> dict:
    nodes, simple_edges, out_adj, in_adj, undirected_adj = build_graph_structures(node_ids, raw_edges)

    node_count = len(nodes)
    relationship_count = len(raw_edges)
    simple_edge_count = len(simple_edges)
    non_self_simple_edge_count = sum(1 for source, target in simple_edges if source != target)

    edge_density = 0.0
    if node_count >= 2:
        edge_density = non_self_simple_edge_count / (node_count * (node_count - 1))

    average_degree = 0.0
    if node_count:
        average_degree = (2.0 * simple_edge_count) / node_count

    wccs = weakly_connected_components(nodes, undirected_adj)
    largest_wcc_size = max((len(component) for component in wccs), default=0)
    largest_wcc_ratio = (largest_wcc_size / node_count) if node_count else 0.0

    sccs = strongly_connected_components(nodes, out_adj, in_adj)
    largest_scc = max(sccs, key=len, default=[])
    largest_scc_size = len(largest_scc)
    largest_scc_ratio = (largest_scc_size / node_count) if node_count else 0.0

    average_shortest_path_length, graph_diameter = average_shortest_path_and_diameter(largest_scc, out_adj)
    efficiency = global_efficiency(nodes, out_adj)
    reciprocity_value = reciprocity(simple_edges)

    return {
        "node_count": node_count,
        "relationship_count": relationship_count,
        "edge_density_directed": edge_density,
        "average_degree": average_degree,
        "weakly_connected_components": len(wccs),
        "largest_wcc_size": largest_wcc_size,
        "largest_wcc_ratio": largest_wcc_ratio,
        "strongly_connected_components": len(sccs),
        "largest_scc_size": largest_scc_size,
        "largest_scc_ratio": largest_scc_ratio,
        "average_shortest_path_length": average_shortest_path_length,
        "graph_diameter": graph_diameter,
        "global_efficiency": efficiency,
        "reciprocity": reciprocity_value,
    }


def main():
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    variant_databases = load_variant_database_names()
    fieldnames = [
        "variant",
        "database_name",
        "node_count",
        "relationship_count",
        "edge_density_directed",
        "average_degree",
        "weakly_connected_components",
        "largest_wcc_size",
        "largest_wcc_ratio",
        "strongly_connected_components",
        "largest_scc_size",
        "largest_scc_ratio",
        "average_shortest_path_length",
        "graph_diameter",
        "global_efficiency",
        "reciprocity",
        "status",
        "error",
    ]

    rows = []
    driver, username = connect_driver()
    emit(f"Connected to Neo4j at {NEO4J_URI} as {username}")
    try:
        for variant in VARIANTS:
            database_name = variant_databases[variant]
            emit(f"[measure] {variant} -> {database_name}")
            row = {
                "variant": variant,
                "database_name": database_name,
                "status": "success",
                "error": "",
            }
            try:
                node_ids, raw_edges = fetch_graph(driver, database_name)
                row.update(compute_metrics(node_ids, raw_edges))
            except Exception as exc:
                row.update({
                    "status": "failure",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                for field in fieldnames:
                    row.setdefault(field, "")
            rows.append(row)
    finally:
        driver.close()

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    emit(f"Wrote measures CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

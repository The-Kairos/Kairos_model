import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


def load_rag_embeddings(path: str) -> Dict[str, Any]:
    """Lightweight loader for RAG embedding JSON files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG embedding file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_question_via_gemini(question: str):
    """Embed a question using Gemini API. Requires GEMINI_API_KEY env var.

    Returns a numpy vector (1D float32 array).
    """
    # Add parent directory to path for this import only
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    
    try:
        from src.rag_convo import embed_question
        res = embed_question(question)
    except ImportError as e:
        raise RuntimeError(f"Could not import embed_question from src.rag_convo: {e}")

    # res is typically a list of embedding objects or a single embedding
    emb = res[0] if isinstance(res, list) and res else res
    if hasattr(emb, "values"):
        return np.array(emb.values, dtype=np.float32)
    if isinstance(emb, dict) and "values" in emb:
        return np.array(emb["values"], dtype=np.float32)
    # fallback to numeric conversion
    return np.array(emb, dtype=np.float32)



def normalize_name(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def resolve_rag_path(checkpoint_path: str, doc: Dict[str, Any]) -> Optional[str]:
    rag_rel = None
    if isinstance(doc.get("rag_embedding"), dict):
        rag_rel = doc.get("rag_embedding", {}).get("rag_path")
    if not rag_rel:
        # fallback guessed name
        rag_rel = os.path.join(os.path.dirname(checkpoint_path), "rag_embedding.json")

    rag_path = os.path.join(os.path.dirname(checkpoint_path), rag_rel) if not os.path.isabs(rag_rel) else rag_rel
    if os.path.exists(rag_path):
        return rag_path
    return None


def compute_kmeans_clusters(embeddings: List[List[float]], num_clusters: int = 8, random_state: int = 42) -> Dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

    X = np.array(embeddings, dtype=np.float32)
    if X.shape[0] == 0:
        return {}
    k = min(num_clusters, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_.tolist()

    return {
        "algorithm": "kmeans",
        "num_clusters": int(k),
        "cluster_assignments": labels.tolist(),
        "centroids": centroids,
    }


def compute_hdbscan_clusters(embeddings: List[List[float]], min_cluster_size: int = 3) -> Dict[str, Any]:
    try:
        import hdbscan
    except Exception:
        return {}

    X = np.array(embeddings, dtype=np.float32)
    if X.shape[0] == 0:
        return {}

    # Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)

    # compute centroids for each non-noise cluster
    centroids = []
    unique = sorted(set([int(l) for l in labels if l >= 0]))
    for l in unique:
        members = X[labels == l]
        centroids.append(np.mean(members, axis=0).tolist())

    return {
        "algorithm": "hdbscan",
        "num_clusters": len(unique),
        "cluster_assignments": [int(x) for x in labels],
        # centroids list aligns with cluster ids in `unique`
        "centroids": centroids,
        "cluster_id_map": {i: cid for i, cid in enumerate(unique)},
    }


def _to_vector(e) -> np.ndarray:
    # support objects that have a `values` attribute (e.g. Gemini EmbeddedResponse)
    if hasattr(e, "values"):
        try:
            return np.array(e.values, dtype=np.float32)
        except Exception:
            pass
    if isinstance(e, dict) and "values" in e:
        return np.array(e["values"], dtype=np.float32)
    return np.array(e, dtype=np.float32)


def merge_retrieval(
    query_vec: np.ndarray,
    scene_embeddings: List[Any],
    contexts: List[str],
    cluster_metadata: Optional[Dict[str, Any]] = None,
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3,
) -> List[Tuple[str, float]]:
    s_vecs = np.array([_to_vector(e) for e in scene_embeddings], dtype=np.float32)
    # normalize if needed
    # base similarities via dot product
    base_sims = s_vecs.dot(query_vec)

    N = len(contexts)
    cluster_boost = np.zeros(N, dtype=np.float32)

    if cluster_metadata:
        centroids = np.array(cluster_metadata.get("centroids", []), dtype=np.float32)
        assignments = np.array(cluster_metadata.get("cluster_assignments", [-1] * N))
        if centroids.size:
            cluster_sims = centroids.dot(query_vec)
            C = centroids.shape[0]
            top_c = min(top_c, C)
            top_ids = np.argsort(cluster_sims)[-top_c:]
            for cid in top_ids:
                # assign boost to members of this centroid
                mask = (assignments == int(cid))
                cluster_boost[mask] = np.maximum(cluster_boost[mask], cluster_sims[cid])

            maxb = cluster_boost.max() if cluster_boost.max() > 0 else 1.0
            cluster_boost = (cluster_boost / maxb) * alpha

    final = base_sims + cluster_boost
    top_idx = np.argsort(final)[-k:][::-1]
    return [(contexts[int(i)], float(final[int(i)])) for i in top_idx]


def jaccard(a: List[int], b: List[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    inter = sa.intersection(sb)
    uni = sa.union(sb)
    return float(len(inter)) / float(len(uni)) if uni else 0.0


def save_json(path: str, payload: Dict[str, Any]):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_md_report(md_path: str, video_title: str, results: Dict[str, Any], config: Dict[str, Any]):
    lines = []
    lines.append(f"# Retrieval Comparison: {video_title}\n")
    lines.append("## Config\n")
    for k, v in config.items():
        lines.append(f"- **{k}**: {v}\n")

    # cluster counts if available
    info = results.get("cluster_info")
    if info:
        lines.append("\n## Cluster counts\n")
        lines.append(f"- KMeans: {info.get('kmeans', 0)} clusters\n")
        lines.append(f"- HDBSCAN: {info.get('hdbscan', 0)} clusters\n")

    lines.append("## Summary\n")
    # Summary table
    lines.append("| Method | Avg Time (s) | Avg Jaccard vs Flat | Notes |\n")
    lines.append("|---|---:|---:|---:|\n")
    for method, rec in results.get("summary", {}).items():
        lines.append(f"| {method} | {rec.get('avg_time', 0):.6f} | {rec.get('avg_jaccard_vs_flat', 0):.3f} | {rec.get('notes','')} |\n")

    lines.append("\n## Per-query details\n")
    for qrec in results.get("queries", []):
        lines.append(f"### Query: {qrec['query']}\n")
        for method, out in qrec["results"].items():
            lines.append(f"- **{method}**: time={out['time']:.6f}s, top_indices={out['top_indices']}, scores={[round(s,3) for s in out['scores']]}\n")
        lines.append("\n")

    save_json(md_path.replace('.md', '.json'), results)
    folder = os.path.dirname(md_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

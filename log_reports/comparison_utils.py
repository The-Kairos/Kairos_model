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


def find_optimal_k_elbow(embeddings: List[List[float]], max_k: int = 20, random_state: int = 42) -> int:
    """Find optimal number of clusters using the elbow method.
    
    Computes KMeans inertia for K in [2, min(len(embeddings)//2, max_k)].
    Returns K with highest elbow (steepest drop in derivative).
    """
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

    X = np.array(embeddings, dtype=np.float32)
    n = X.shape[0]
    if n < 3:
        return 1  # too few points
    
    # test range of K values
    max_test = min(n // 2, max_k)
    if max_test < 2:
        return 1
    
    inertias = []
    k_values = list(range(2, max_test + 1))
    
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    
    # compute differences (how much each K reduces inertia)
    diffs = np.diff(inertias)
    # compute second derivative (acceleration of inertia drop)
    accel = np.diff(diffs)
    
    # elbow is where acceleration is maximal (steepest change in slope)
    if len(accel) > 0:
        elbow_idx = np.argmax(accel) + 1  # +1 because we lost one value in diff
        optimal_k = k_values[elbow_idx]
    else:
        optimal_k = k_values[0]
    
    return max(2, optimal_k)


def compute_kmeans_clusters(embeddings: List[List[float]], num_clusters: Optional[int] = None, random_state: int = 42) -> Dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

    X = np.array(embeddings, dtype=np.float32)
    if X.shape[0] == 0:
        return {}
    
    # if num_clusters not specified, find optimal K using elbow method
    if num_clusters is None:
        num_clusters = find_optimal_k_elbow(embeddings, max_k=20, random_state=random_state)
    
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
        if assignments.shape[0] != N:
            print(f"Warning: cluster_assignments length ({assignments.shape[0]}) != contexts length ({N}), skipping cluster boost")
            cluster_metadata = None  # disable boosting
        elif centroids.size:
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
    # Clip indices to valid range and warn if any were invalid
    valid_top_idx = []
    for idx in top_idx:
        if 0 <= idx < N:
            valid_top_idx.append(idx)
        else:
            print(f"Warning: Invalid scene index {idx} (valid range: 0-{N-1}), clipping to valid indices")
    if len(valid_top_idx) < k:
        print(f"Warning: Only {len(valid_top_idx)} valid indices found out of {k} requested")
    top_idx = valid_top_idx[:k]  # Take up to k valid indices
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


def write_md_report(md_path: str, video_title: str, results: Dict[str, Any], config: Dict[str, Any], checkpoint: Optional[Dict[str, Any]] = None):
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

        # Add scene details table for this query
        if checkpoint and "scenes" in checkpoint:
            scene_list = checkpoint["scenes"]
            scene_info = {s["scene_index"]: {"timestamp": s["start_timecode"], "description": s.get("llm_scene_description", "")[:150]} for s in scene_list}
            
            # Collect unique scene indices from all methods
            unique_indices = set()
            for method, out in qrec["results"].items():
                unique_indices.update(out["top_indices"])
            
            if unique_indices:
                lines.append("#### Scene Details\n")
                lines.append("| Scene Index | Timestamp | Description |\n")
                lines.append("|---|---|---|\n")
                for idx in sorted(unique_indices):
                    if idx in scene_info:
                        desc = scene_info[idx]["description"].replace("\n", " ").strip()
                        lines.append(f"| {idx} | {scene_info[idx]['timestamp']} | {desc} |\n")
                lines.append("\n")

    save_json(md_path.replace('.md', '.json'), results)
    folder = os.path.dirname(md_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

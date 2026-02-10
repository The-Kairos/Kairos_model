import numpy as np
from typing import List, Dict
from index import InMemoryIndex
from timing import timed

def semantic_search(
    query_embedding: np.ndarray,
    index: InMemoryIndex,
    top_k: int = 5,
    timings: dict | None = None,
) -> List[Dict]:
    if not index.is_ready():
        raise RuntimeError("Index not built")

    if timings is None:
        timings = {}

    with timed("similarity_search_ms", timings):
        scores = index.embeddings @ query_embedding  # (N,)
        top_idxs = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idxs:
        scene = index.scenes[idx]
        results.append({
            "scene_index": scene["scene_index"],
            "score": float(scores[idx]),
            "description": scene["description"],
        })

    return results, timings

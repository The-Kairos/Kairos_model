import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Minimal load_prompt to avoid dependency issues
def load_prompt(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompts_dir = REPO_ROOT / "prompts"
    prompt_path = prompts_dir / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

from src.rag_convo import create_answer, _get_gemini_client
from TEST_QUERIES_MAP import TEST_QUERIES_MAP


def load_rag_embeddings(path: str) -> Dict[str, Any]:
    """Lightweight loader for RAG embedding JSON files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG embedding file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_question_via_gemini(question: str):
    """Embed a question using Gemini API. Requires GEMINI_API_KEY env var."""
    try:
        from src.rag_convo import embed_question
        res = embed_question(question)
    except ImportError as e:
        raise RuntimeError(f"Could not import embed_question from src.rag_convo: {e}")

    emb = res[0] if isinstance(res, list) and res else res
    if hasattr(emb, "values"):
        return np.array(emb.values, dtype=np.float32)
    if isinstance(emb, dict) and "values" in emb:
        return np.array(emb["values"], dtype=np.float32)
    return np.array(emb, dtype=np.float32)


def normalize_name(name: str) -> str:
    """Normalize video name for matching against TEST_QUERIES_MAP."""
    s = name.lower().strip()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def find_optimal_k_elbow(embeddings: List[List[float]], max_k: int = 20, random_state: int = 42) -> int:
    """Find optimal number of clusters using the elbow method."""
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

    X = np.array(embeddings, dtype=np.float32)
    n = X.shape[0]
    if n < 3:
        return 1

    max_test = min(n // 2, max_k)
    if max_test < 2:
        return 1

    inertias = []
    k_values = list(range(2, max_test + 1))
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)

    diffs = np.diff(inertias)
    accel = np.diff(diffs)
    if len(accel) > 0:
        elbow_idx = np.argmax(accel) + 1
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

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)

    centroids = []
    unique = sorted(set([int(l) for l in labels if l >= 0]))
    for l in unique:
        members = X[labels == l]
        centroids.append(np.mean(members, axis=0).tolist())

    return {
        "algorithm": "hdbscan",
        "num_clusters": len(unique),
        "cluster_assignments": [int(x) for x in labels],
        "centroids": centroids,
        "cluster_id_map": {i: cid for i, cid in enumerate(unique)},
    }


def _to_vector(e) -> np.ndarray:
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
    base_sims = s_vecs.dot(query_vec)

    N = len(contexts)
    cluster_boost = np.zeros(N, dtype=np.float32)

    if cluster_metadata:
        centroids = np.array(cluster_metadata.get("centroids", []), dtype=np.float32)
        assignments = np.array(cluster_metadata.get("cluster_assignments", [-1] * N))
        if assignments.shape[0] != N:
            print(f"Warning: cluster_assignments length ({assignments.shape[0]}) != contexts length ({N}), skipping cluster boost")
            cluster_metadata = None
        elif centroids.size:
            cluster_sims = centroids.dot(query_vec)
            C = centroids.shape[0]
            top_c = min(top_c, C)
            top_ids = np.argsort(cluster_sims)[-top_c:]
            for cid in top_ids:
                mask = (assignments == int(cid))
                cluster_boost[mask] = np.maximum(cluster_boost[mask], cluster_sims[cid])

            maxb = cluster_boost.max() if cluster_boost.max() > 0 else 1.0
            cluster_boost = (cluster_boost / maxb) * alpha

    final = base_sims + cluster_boost
    top_idx = np.argsort(final)[-k:][::-1]
    valid_top_idx = []
    for idx in top_idx:
        if 0 <= idx < N:
            valid_top_idx.append(idx)
        else:
            print(f"Warning: Invalid scene index {idx} (valid range: 0-{N-1}), clipping to valid indices")
    if len(valid_top_idx) < k:
        print(f"Warning: Only {len(valid_top_idx)} valid indices found out of {k} requested")
    top_idx = valid_top_idx[:k]
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


def perform_flat_retrieval(
    question_vec: np.ndarray,
    embeddings: List[Any],
    contexts: List[str],
    k: int = 10
) -> List[Tuple[str, float]]:
    """Perform flat retrieval using direct cosine similarity."""
    s_vecs = np.array([np.array(e, dtype=np.float32) for e in embeddings], dtype=np.float32)
    similarities = np.dot(s_vecs, question_vec)
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(contexts[i], float(similarities[i])) for i in top_indices]


def perform_kmeans_retrieval(
    question_vec: np.ndarray,
    embeddings: List[Any],
    contexts: List[str],
    cluster_metadata: Dict[str, Any],
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3
) -> List[Tuple[str, float]]:
    """Perform KMeans hierarchical retrieval."""
    return merge_retrieval(question_vec, embeddings, contexts, cluster_metadata, k, top_c, alpha)


def perform_hdbscan_retrieval(
    question_vec: np.ndarray,
    embeddings: List[Any],
    contexts: List[str],
    cluster_metadata: Dict[str, Any],
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3
) -> List[Tuple[str, float]]:
    """Perform HDBSCAN hierarchical retrieval."""
    return merge_retrieval(question_vec, embeddings, contexts, cluster_metadata, k, top_c, alpha)


def generate_answer_for_chunks(
    question: str,
    chunks: List[Tuple[str, float]],
    client=None,
    model="gemini-2.5-pro"
) -> str:
    """Generate an answer using the provided chunks."""
    if client is None:
        client = _get_gemini_client()

    # Extract just the text from chunks (ignore scores for generation)
    context_texts = [text for text, _ in chunks]
    return create_answer(question, [(text, 0.0) for text in context_texts], client=client, model=model)


def compute_chunk_overlap(chunks_a: List[Tuple[str, float]], chunks_b: List[Tuple[str, float]]) -> float:
    """Compute Jaccard overlap between two sets of retrieved chunks."""
    texts_a = set(text for text, _ in chunks_a)
    texts_b = set(text for text, _ in chunks_b)
    return jaccard(list(texts_a), list(texts_b))


def run_hierarchical_retrieval_comparison(
    video_folder_path: str,
    queries: List[str],
    config: Dict[str, Any],
    output_dir: str = "./log_reports/comparison_results",
    client = None
) -> Dict[str, Any]:
    """
    Run retrieval comparison for a single video.

    Args:
        video_folder_path: Path to the processed video folder
        queries: List of questions to test
        config: Configuration dict with k, top_c, alpha
        output_dir: Where to save results

    Returns:
        Results dictionary with all comparison data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract video name from folder path
    video_name = os.path.basename(video_folder_path)

    # Load checkpoint.json for reference
    checkpoint_path = os.path.join(video_folder_path, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint.json in {video_folder_path}")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    # Load rag_embedding.json
    rag_path = os.path.join(video_folder_path, "rag_embedding.json")
    if not os.path.exists(rag_path):
        raise FileNotFoundError(f"No rag_embedding.json in {video_folder_path}")
    rag_data = load_rag_embeddings(rag_path)
    contexts = rag_data.get("contexts", [])
    embeddings = rag_data.get("embeddings", [])

    if not contexts or not embeddings:
        raise ValueError(f"RAG embedding file {rag_path} is missing contexts or embeddings")

    # Prepare cluster metadata file paths
    kmeans_meta_path = os.path.join(video_folder_path, "rag_embedding_kmeans_clusters.json")
    hdbscan_meta_path = os.path.join(video_folder_path, "rag_embedding_hdbscan_clusters.json")

    # Load/compute kmeans clusters
    if os.path.exists(kmeans_meta_path):
        with open(kmeans_meta_path, "r", encoding="utf-8") as f:
            kmeans_meta = json.load(f)
    else:
        kmeans_meta = compute_kmeans_clusters(embeddings, num_clusters=None)
        if kmeans_meta:
            save_json(kmeans_meta_path, kmeans_meta)

    # Load/compute hdbscan clusters
    if os.path.exists(hdbscan_meta_path):
        with open(hdbscan_meta_path, "r", encoding="utf-8") as f:
            hdbscan_meta = json.load(f)
    else:
        hdbscan_meta = compute_hdbscan_clusters(embeddings)
        if hdbscan_meta:
            save_json(hdbscan_meta_path, hdbscan_meta)

    # Get cluster counts for reporting
    km_count = kmeans_meta.get("num_clusters", 0) if kmeans_meta else 0
    hb_count = hdbscan_meta.get("num_clusters", 0) if hdbscan_meta else 0

    print(f"Processing video: {video_name}")
    print(f"  KMeans clusters: {km_count}")
    print(f"  HDBSCAN clusters: {hb_count}")
    print(f"  Total chunks: {len(contexts)}")

    # Initialize results structure
    results = {
        "video": video_name,
        "config": config,
        "cluster_info": {"kmeans": km_count, "hdbscan": hb_count},
        "queries": [],
        "summary": {
            "flat": {"times": [], "chunk_counts": []},
            "kmeans": {"times": [], "chunk_counts": [], "overlaps_vs_flat": []},
            "hdbscan": {"times": [], "chunk_counts": [], "overlaps_vs_flat": []}
        }
    }

    if client is None:
        # Get Gemini client for answer generation
        client = _get_gemini_client()

    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}/{len(queries)}: {query[:50]}{'...' if len(query) > 50 else ''}")

        qrec = {"query": query, "results": {}}

        # Embed question once
        try:
            q_vec = embed_question_via_gemini(query)
            q_vec = np.array(q_vec, dtype=np.float32)
        except Exception as e:
            print(f"    Failed to embed query: {e}")
            continue

        # Flat retrieval
        t_start = time.perf_counter()
        flat_chunks = perform_flat_retrieval(q_vec, embeddings, contexts, k=config["k"])
        t_retrieval = time.perf_counter() - t_start

        t_start = time.perf_counter()
        flat_answer = generate_answer_for_chunks(query, flat_chunks, client=client)
        t_generation = time.perf_counter() - t_start

        qrec["results"]["flat"] = {
            "retrieval_time": t_retrieval,
            "generation_time": t_generation,
            "total_time": t_retrieval + t_generation,
            "chunks": flat_chunks,
            "answer": flat_answer
        }

        results["summary"]["flat"]["times"].append(t_retrieval + t_generation)
        results["summary"]["flat"]["chunk_counts"].append(len(flat_chunks))

        # KMeans hierarchical retrieval
        if kmeans_meta:
            t_start = time.perf_counter()
            kmeans_chunks = perform_kmeans_retrieval(
                q_vec, embeddings, contexts, kmeans_meta,
                k=config["k"], top_c=config["top_c"], alpha=config["alpha"]
            )
            t_retrieval = time.perf_counter() - t_start

            t_start = time.perf_counter()
            kmeans_answer = generate_answer_for_chunks(query, kmeans_chunks, client=client)
            t_generation = time.perf_counter() - t_start

            overlap = compute_chunk_overlap(flat_chunks, kmeans_chunks)

            qrec["results"]["kmeans"] = {
                "retrieval_time": t_retrieval,
                "generation_time": t_generation,
                "total_time": t_retrieval + t_generation,
                "chunks": kmeans_chunks,
                "answer": kmeans_answer,
                "overlap_vs_flat": overlap
            }

            results["summary"]["kmeans"]["times"].append(t_retrieval + t_generation)
            results["summary"]["kmeans"]["chunk_counts"].append(len(kmeans_chunks))
            results["summary"]["kmeans"]["overlaps_vs_flat"].append(overlap)
        else:
            qrec["results"]["kmeans"] = {"error": "No KMeans clusters available"}

        # HDBSCAN hierarchical retrieval
        if hdbscan_meta:
            t_start = time.perf_counter()
            hdbscan_chunks = perform_hdbscan_retrieval(
                q_vec, embeddings, contexts, hdbscan_meta,
                k=config["k"], top_c=config["top_c"], alpha=config["alpha"]
            )
            t_retrieval = time.perf_counter() - t_start

            t_start = time.perf_counter()
            hdbscan_answer = generate_answer_for_chunks(query, hdbscan_chunks, client=client)
            t_generation = time.perf_counter() - t_start

            overlap = compute_chunk_overlap(flat_chunks, hdbscan_chunks)

            qrec["results"]["hdbscan"] = {
                "retrieval_time": t_retrieval,
                "generation_time": t_generation,
                "total_time": t_retrieval + t_generation,
                "chunks": hdbscan_chunks,
                "answer": hdbscan_answer,
                "overlap_vs_flat": overlap
            }

            results["summary"]["hdbscan"]["times"].append(t_retrieval + t_generation)
            results["summary"]["hdbscan"]["chunk_counts"].append(len(hdbscan_chunks))
            results["summary"]["hdbscan"]["overlaps_vs_flat"].append(overlap)
        else:
            qrec["results"]["hdbscan"] = {"error": "No HDBSCAN clusters available"}

        results["queries"].append(qrec)

    # Compute summary statistics
    for method in ["flat", "kmeans", "hdbscan"]:
        times = results["summary"][method]["times"]
        counts = results["summary"][method]["chunk_counts"]
        overlaps = results["summary"][method].get("overlaps_vs_flat", [])

        results["summary"][method]["avg_time"] = float(np.mean(times)) if times else 0.0
        results["summary"][method]["avg_chunk_count"] = float(np.mean(counts)) if counts else 0.0
        results["summary"][method]["avg_overlap_vs_flat"] = float(np.mean(overlaps)) if overlaps else 0.0

    # Write markdown report
    safe_name = video_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    md_path = os.path.join(output_dir, f"{safe_name}_comparison.md")
    write_comparison_report(md_path, results, checkpoint)

    # Write JSON results
    json_path = os.path.join(output_dir, f"{safe_name}_comparison.json")
    save_json(json_path, results)

    print(f"  Wrote results to: {md_path}")
    return results


def format_query_section(query_data: Dict[str, Any]) -> str:
    """Format a single query's results for the markdown report."""
    lines = []
    query = query_data["query"]
    lines.append(f"## Query: {query}\n")

    methods = ["flat", "kmeans", "hdbscan"]

    for method in methods:
        if method not in query_data["results"]:
            continue

        result = query_data["results"][method]
        if "error" in result:
            lines.append(f"### {method.upper()} Retrieval\n")
            lines.append(f"**Error**: {result['error']}\n\n")
            continue

        lines.append(f"### {method.upper()} Retrieval\n")

        # Retrieved chunks
        lines.append("**Retrieved Chunks**:\n")
        for i, (chunk_text, score) in enumerate(result["chunks"], 1):
            # Truncate very long chunks for readability
            truncated = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            lines.append(f"{i}. [{score:.3f}] {truncated}\n")
        lines.append("\n")

        # Generated answer
        lines.append("**Generated Answer**:\n")
        answer = result["answer"].strip()
        # Format answer with proper markdown
        lines.append(f"{answer}\n\n")

        # Timing
        lines.append("**Timing**:\n")
        lines.append(f"- Retrieval: {result['retrieval_time']:.3f}s\n")
        lines.append(f"- Generation: {result['generation_time']:.3f}s\n")
        lines.append(f"- Total: {result['total_time']:.3f}s\n")
        if "overlap_vs_flat" in result:
            lines.append(f"- Overlap vs Flat: {result['overlap_vs_flat']:.1%}\n")
        lines.append("\n")

        lines.append("---\n\n")

    return "".join(lines)


def write_comparison_report(md_path: str, results: Dict[str, Any], checkpoint: Optional[Dict[str, Any]] = None):
    """Write the complete comparison report in markdown format."""
    lines = []
    video_name = results["video"]
    config = results["config"]
    cluster_info = results["cluster_info"]

    lines.append(f"# Retrieval Comparison: {video_name}\n\n")

    # Configuration
    lines.append("## Configuration\n\n")
    lines.append(f"- **k** (top chunks): {config['k']}\n")
    lines.append(f"- **top_c** (top clusters): {config['top_c']}\n")
    lines.append(f"- **alpha** (cluster boost): {config['alpha']}\n")
    lines.append(f"- **KMeans clusters**: {cluster_info['kmeans']}\n")
    lines.append(f"- **HDBSCAN clusters**: {cluster_info['hdbscan']}\n")
    lines.append(f"- **Total queries**: {len(results['queries'])}\n\n")

    # Summary table
    lines.append("## Summary\n\n")
    lines.append("| Method | Avg Time (s) | Avg Chunks | Avg Overlap vs Flat |\n")
    lines.append("|--------|-------------:|-----------:|-------------------:|\n")

    for method in ["flat", "kmeans", "hdbscan"]:
        summary = results["summary"][method]
        avg_time = summary.get("avg_time", 0.0)
        avg_chunks = summary.get("avg_chunk_count", 0.0)
        avg_overlap = summary.get("avg_overlap_vs_flat", 0.0)

        method_name = method.upper()
        lines.append(f"| {method_name} | {avg_time:.3f} | {avg_chunks:.1f} | {avg_overlap:.1%} |\n")

    lines.append("\n")

    # Per-query details
    lines.append("## Per-Query Results\n\n")
    for qrec in results["queries"]:
        lines.append(format_query_section(qrec))

    # Write the file
    folder = os.path.dirname(md_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def run_comparison_for_test_queries(
    input_dir: str = "./_processed",
    output_dir: str = "./log_reports/comparison_results",
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3,
    n_queries: int = 0,
    client = None
):
    """
    Run comparison for all videos in TEST_QUERIES_MAP found in input_dir.

    Args:
        input_dir: Directory containing processed video folders
        output_dir: Where to save comparison results
        k: Number of top chunks to retrieve
        top_c: Number of top clusters for hierarchical retrieval
        alpha: Weight for cluster boosting in hierarchical retrieval
        n_queries: Max queries per video (0 = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize map keys
    norm_map = {normalize_name(k): v for k, v in TEST_QUERIES_MAP.items()}

    if not os.path.isdir(input_dir):
        print(f"Error: input directory {input_dir} not found")
        return

    config = {"k": k, "top_c": top_c, "alpha": alpha}

    # Iterate over video subdirectories
    for video_folder in os.listdir(input_dir):
        video_folder_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(video_folder_path):
            continue

        # Extract video name and normalize
        video_name = video_folder
        nname = normalize_name(os.path.splitext(video_name)[0])

        if nname not in norm_map:
            continue  # Video not in TEST_QUERIES_MAP

        queries = norm_map[nname]
        if n_queries > 0:
            queries = queries[:n_queries]

        try:
            run_hierarchical_retrieval_comparison(
                video_folder_path=video_folder_path,
                queries=queries,
                config=config,
                output_dir=output_dir,
                client = client
            )
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue

    print(f"\nComparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    run_comparison_for_test_queries(
        input_dir="./_processed",
        output_dir="./log_reports/comparison_results",
        k=10,
        top_c=3,
        alpha=0.3,
        n_queries=0  # All queries
    )

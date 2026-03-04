import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, List

import numpy as np

# Add parent directory to path so we can import src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from TEST_QUERIES_MAP import TEST_QUERIES_MAP
from comparison_utils import (
    normalize_name,
    resolve_rag_path,
    compute_kmeans_clusters,
    compute_hdbscan_clusters,
    merge_retrieval,
    _to_vector,
    save_json,
    write_md_report,
    jaccard,
    load_rag_embeddings,
    embed_question_via_gemini,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="./_processed", help="Base folder containing video subdirectories")
    p.add_argument("--output-dir", default="./log_reports/comparison_results")
    p.add_argument("--n", type=int, default=0, help="max queries per video (0 = all)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--top-c", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--cluster-k", type=int, default=8)
    return p.parse_args()


def run_benchmark(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # normalize map keys
    norm_map = {normalize_name(k): v for k, v in TEST_QUERIES_MAP.items()}

    # Scan for video subdirectories in input_dir
    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory {args.input_dir} not found")
        return

    # Iterate over subdirectories (each is a video folder)
    for video_folder in os.listdir(args.input_dir):
        video_folder_path = os.path.join(args.input_dir, video_folder)
        if not os.path.isdir(video_folder_path):
            continue

        # Extract video name from folder (e.g., "Young Sheldon - First Day of High School.mp4")
        video_name = video_folder
        # Normalize to match against TEST_QUERIES_MAP keys
        nname = normalize_name(os.path.splitext(video_name)[0])

        if nname not in norm_map:
            # Video folder not in TEST_QUERIES_MAP, skip it
            continue

        queries = norm_map[nname]
        if args.n > 0:
            queries = queries[: args.n]

        # Load checkpoint.json from video folder
        checkpoint_path = os.path.join(video_folder_path, "checkpoint.json")
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint.json in {video_folder_path}; skipping")
            continue

        try:
            with open(checkpoint_path, "r", encoding="utf-8", errors="replace") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}; skipping")
            continue

        # Load rag_embedding.json from video folder
        rag_path = os.path.join(video_folder_path, "rag_embedding.json")
        if not os.path.exists(rag_path):
            print(f"No rag_embedding.json in {video_folder_path}; skipping")
            continue

        rag = load_rag_embeddings(rag_path)
        contexts = rag.get("contexts", [])
        embeddings = rag.get("embeddings", [])

        # Prepare cluster metadata file paths
        kmeans_meta_path = os.path.join(video_folder_path, "rag_embedding_kmeans_clusters.json")
        hdbscan_meta_path = os.path.join(video_folder_path, "rag_embedding_hdbscan_clusters.json")

        # compute/load kmeans clusters
        if os.path.exists(kmeans_meta_path):
            with open(kmeans_meta_path, "r", encoding="utf-8") as f:
                kmeans_meta = json.load(f)
        else:
            kmeans_meta = compute_kmeans_clusters(embeddings, num_clusters=args.cluster_k)
            if kmeans_meta:
                save_json(kmeans_meta_path, kmeans_meta)
        # determine number of clusters (use explicit key or deduce from assignments)
        km_count = kmeans_meta.get("num_clusters")
        if km_count is None and "cluster_assignments" in kmeans_meta:
            km_count = len(set(kmeans_meta.get("cluster_assignments", [])))
        km_count = int(km_count) if km_count is not None else 0
        print(f"KMeans clusters: {km_count} clusters")
        
        # compute/load hdbscan clusters (optional)
        hdbscan_meta = {}
        try:
            if os.path.exists(hdbscan_meta_path):
                with open(hdbscan_meta_path, "r", encoding="utf-8") as f:
                    hdbscan_meta = json.load(f)
            else:
                hdbscan_meta = compute_hdbscan_clusters(embeddings)
                if hdbscan_meta:
                    save_json(hdbscan_meta_path, hdbscan_meta)
        except Exception:
            hdbscan_meta = {}
        hb_count = hdbscan_meta.get("num_clusters")
        if hb_count is None and "cluster_assignments" in hdbscan_meta:
            hb_count = len(set(hdbscan_meta.get("cluster_assignments", [])))
        hb_count = int(hb_count) if hb_count is not None else 0
        print(f"HDBSCAN clusters: {hb_count} clusters")
        
        # stash counts for report
        cluster_info = {"kmeans": km_count, "hdbscan": hb_count}


        print("Running benchmark for video:", video_name)
        # prepare result container, include cluster counts
        results = {"video": video_name, "queries": [], "cluster_info": cluster_info}
        summary = {"flat": {"times": [], "jaccard": []}, "kmeans": {"times": [], "jaccard": []}, "hdbscan": {"times": [], "jaccard": []}}

        s_vecs = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)

        for q in queries:
            qrec = {"query": q, "results": {}}

            # embed question
            try:
                q_vec = embed_question_via_gemini(q)  # already returns numpy vector
            except Exception as e:
                print(f"Failed to embed query '{q}': {e}")
                continue
            print("Query embedded successfully")

            # Baseline flat retrieval (compute similarities directly)
            t_start = time.perf_counter()
            base_sims = s_vecs.dot(q_vec)
            top_idx = np.argsort(base_sims)[-args.k:][::-1]
            t_end = time.perf_counter()
            top_indices = [int(i) for i in top_idx]
            top_scores = [float(base_sims[int(i)]) for i in top_idx]
            qrec["results"]["flat"] = {"time": t_end - t_start, "top_indices": top_indices, "scores": top_scores}
            summary["flat"]["times"].append(t_end - t_start)

            # KMeans hierarchical
            t_start = time.perf_counter()
            merged = merge_retrieval(q_vec, embeddings, contexts, cluster_metadata=kmeans_meta, k=args.k, top_c=args.top_c, alpha=args.alpha)
            t_end = time.perf_counter()
            k_idx = []
            k_scores = []
            # find indices by matching contexts (safe since contexts are unique per scene)
            ctx_to_idx = {c: i for i, c in enumerate(contexts)}
            for ctx, score in merged:
                idx = ctx_to_idx.get(ctx, None)
                if idx is not None:
                    k_idx.append(int(idx))
                    k_scores.append(float(score))
            qrec["results"]["kmeans"] = {"time": t_end - t_start, "top_indices": k_idx, "scores": k_scores}
            summary["kmeans"]["times"].append(t_end - t_start)
            summary["kmeans"]["jaccard"].append(jaccard(top_indices, k_idx))

            # HDBSCAN hierarchical (if available)
            if hdbscan_meta:
                t_start = time.perf_counter()
                merged_h = merge_retrieval(q_vec, embeddings, contexts, cluster_metadata=hdbscan_meta, k=args.k, top_c=args.top_c, alpha=args.alpha)
                t_end = time.perf_counter()
                h_idx = []
                h_scores = []
                for ctx, score in merged_h:
                    idx = ctx_to_idx.get(ctx, None)
                    if idx is not None:
                        h_idx.append(int(idx))
                        h_scores.append(float(score))
                qrec["results"]["hdbscan"] = {"time": t_end - t_start, "top_indices": h_idx, "scores": h_scores}
                summary["hdbscan"]["times"].append(t_end - t_start)
                summary["hdbscan"]["jaccard"].append(jaccard(top_indices, h_idx))
            else:
                qrec["results"]["hdbscan"] = {"time": 0.0, "top_indices": [], "scores": []}

            results["queries"].append(qrec)

        # aggregate summary stats
        out_summary = {}
        for method in ["flat", "kmeans", "hdbscan"]:
            times = summary[method]["times"]
            jacs = summary[method].get("jaccard", [])
            out_summary[method] = {
                "avg_time": float(np.mean(times)) if times else 0.0,
                "avg_jaccard_vs_flat": float(np.mean(jacs)) if jacs else 0.0,
            }

        results["summary"] = out_summary

        # write outputs
        # Use video name (with extension removed) as safe name
        safe_name = os.path.splitext(video_name)[0].replace(' ', '_')
        md_path = os.path.join(args.output_dir, f"{safe_name}_comparison.md")
        write_md_report(md_path, video_name, results, config={"k": args.k, "top_c": args.top_c, "alpha": args.alpha, "cluster_k": args.cluster_k})
        print(f"Wrote results for {video_name} -> {md_path}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)

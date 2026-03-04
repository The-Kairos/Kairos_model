# RAG Retrieval Benchmark (log_reports)

Files added:

- `TEST_QUERIES_MAP.py` — placeholder mapping of video base-names to query lists.
- `comparison_utils.py` — helper utilities for clustering, merging, metrics, and report writing.
- `rag_retrieval_benchmark.py` — main runner to execute flat, K-Means, and HDBSCAN retrieval strategies and produce per-video reports.

Quick start

1. Populate `log_reports/TEST_QUERIES_MAP.py` with your test queries. Keys must be the video base filename (no extension) as found in `logs/_processed/<checkpoint>.json` `video_path` field.
2. Ensure embeddings exist in each checkpoint RAG file (the script reads `rag_embedding.rag_path` or falls back to `rag_embedding.json` next to the checkpoint).
3. (Optional) Set your Gemini API key as an environment variable if you want the runner to call `embed_question()` at runtime:

```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

Dependencies

- Python packages: `numpy`, `scikit-learn`.
- Optional for HDBSCAN variant: `hdbscan` (and optionally `umap-learn` for dimensionality reduction). Install via pip or conda.

Example runs

- Run against all videos listed in `TEST_QUERIES_MAP` (use first N queries per video if `--n` set):

```bash
python log_reports/rag_retrieval_benchmark.py --n 5 --k 10 --top-c 3 --alpha 0.3
```

CLI options (important ones)

- `--input-dir` : folder with processed checkpoints (default `./logs/_processed`).
- `--output-dir` : where per-video reports and raw JSON are written (default `./log_reports/comparison_results`).
- `--n` : max queries per video (0 = all)
- `--k` : top-K scenes to retrieve
- `--top-c` : number of clusters to use for boosting
- `--cluster-k` : number of K-Means clusters to compute
- `--alpha` : cluster boost strength (0..1)

Outputs

- Per-video Markdown report: `log_reports/comparison_results/<video_base>_comparison.md`
- Per-video raw JSON: `log_reports/comparison_results/<video_base>_raw.json`
- Cluster metadata stored next to rag embeddings: `<rag_base>_kmeans_clusters.json` and `<rag_base>_hdbscan_clusters.json` (if computed).

Notes & tips

- The script is non-destructive: it reads existing checkpoints and RAG files and writes reports under `log_reports/comparison_results`.
- If `hdbscan` is not installed the HDBSCAN variant will be skipped gracefully.
- Matching between `TEST_QUERIES_MAP` keys and processed checkpoints is case-insensitive and normalized (underscores → spaces, collapses whitespace). Use exact base filenames to avoid fuzzy matching.
- HDBSCAN may label some scenes as noise (`-1`); noise scenes receive no cluster boost.

If you want, I can run a short smoke test for one video (requires GEMINI API key present to embed queries). Otherwise, populate `TEST_QUERIES_MAP.py` and run the example command above.

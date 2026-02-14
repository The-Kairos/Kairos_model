"""
Comprehensive Comparison: Semantic Search vs RAG with Multiple Embedding Methods

This script benchmarks:
1. Semantic Search with Gemini Embedding
2. Semantic Search with SentenceTransformer Embedding
3. RAG System with Gemini Embedding
4. RAG System with SentenceTransformer Embedding

Metrics captured:
- Index/Store building time
- Description embedding time
- Query embedding time
- Retrieval time
- Overall time per query

Output: Detailed markdown report in log_reports/
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np

# =====================================================================
# CONFIGURATION (MODIFY THESE VARIABLES)
# =====================================================================

VIDEO_LOG_PATH = "logs/.action/messi_20260210_190050"  # Change this to your video log
NUM_QUERIES = 5  # Number of test queries to run
TEST_QUERIES = [
    "the scene where the person washes their hands",
    "the scene where they are rolling dough",
    "the scene where flour is being measured",
    "when they serve pasta",
    "a cooking demonstration"
][:NUM_QUERIES]  # Use first N queries

# =====================================================================
# SETUP PATHS AND IMPORTS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SEMANTIC_SEARCH_DIR = PROJECT_ROOT / "semantic_search"
RAG_DIR = PROJECT_ROOT / "rag"
LOG_REPORTS_DIR = PROJECT_ROOT / "log_reports"

# Add paths to sys.path
sys.path.insert(0, str(SEMANTIC_SEARCH_DIR))
sys.path.insert(0, str(RAG_DIR))

# Import semantic search components
from semantic_search.loader import load_scenes
from semantic_search.embedder import TextEmbedder, format_embedding_text
from semantic_search.index import InMemoryIndex
from semantic_search.search import semantic_search

# Import RAG components
from rag.embedding import EmbeddingEngine, format_embedding_text as rag_format_embedding_text
from rag.vector_store import VectorStore


# =====================================================================
# RESULT TRACKING STRUCTURES
# =====================================================================

class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.index_time_ms = 0.0
        self.embedding_time_ms = 0.0
        self.query_times = []  # List of per-query timings
        self.query_results = []  # List of retrieved results per query
        
    def total_time_ms(self):
        return self.index_time_ms + self.embedding_time_ms + sum(self.query_times)
    
    def avg_query_time_ms(self):
        return np.mean(self.query_times) if self.query_times else 0.0


# =====================================================================
# SEMANTIC SEARCH BENCHMARK
# =====================================================================

def benchmark_semantic_search(embedding_method: str, log_path: str, queries: List[str]) -> BenchmarkResult:
    """
    Benchmark semantic search with specified embedding method.
    """
    print(f"\n[SEMANTIC SEARCH - {embedding_method.upper()}]")
    result = BenchmarkResult(f"Semantic Search ({embedding_method})")
    
    # Load scenes
    print("  Loading scenes...")
    scenes = load_scenes(f"{log_path}.json")
    
    # Format texts
    print("  Formatting texts...")
    texts = format_embedding_text(scenes)
    
    # Create embedder
    print(f"  Creating embedder ({embedding_method})...")
    embedder = TextEmbedder(method=embedding_method)
    
    # Embed descriptions (time this)
    print("  Embedding descriptions...")
    emb_start = time.perf_counter()
    embeddings = embedder.embed_texts(texts)
    result.embedding_time_ms = (time.perf_counter() - emb_start) * 1000
    
    # Build index (time this)
    print("  Building index...")
    idx_start = time.perf_counter()
    index = InMemoryIndex()
    index.build(scenes, embeddings)
    result.index_time_ms = (time.perf_counter() - idx_start) * 1000
    
    # Query benchmark
    print(f"  Running {len(queries)} queries...")
    for query in queries:
        query_start = time.perf_counter()
        
        # Query embedding
        q_emb = embedder.embed_query(query)
        
        # Retrieve
        query_results, _ = semantic_search(q_emb, index, top_k=5)
        
        query_time = (time.perf_counter() - query_start) * 1000
        result.query_times.append(query_time)
        result.query_results.append({
            "query": query,
            "results": query_results,
            "time_ms": query_time
        })
    
    print(f"  ✓ Completed (Index: {result.index_time_ms:.2f}ms, "
          f"Embeddings: {result.embedding_time_ms:.2f}ms, "
          f"Avg Query: {result.avg_query_time_ms():.2f}ms)")
    
    return result


# =====================================================================
# RAG BENCHMARK
# =====================================================================

def benchmark_rag(embedding_method: str, log_path: str, queries: List[str]) -> BenchmarkResult:
    """
    Benchmark RAG retrieval with specified embedding method.
    """
    print(f"\n[RAG - {embedding_method.upper()}]")
    result = BenchmarkResult(f"RAG ({embedding_method})")
    
    # Load scenes
    print("  Loading scenes...")
    with open(f"{log_path}.json", "r", encoding="utf-8") as f:
        logs = json.load(f)
    
    scenes = logs["scenes"]
    
    # Format texts
    print("  Formatting texts...")
    texts = rag_format_embedding_text(scenes)
    
    # Create embedding engine
    print(f"  Creating embedding engine ({embedding_method})...")
    engine = EmbeddingEngine(method=embedding_method)
    
    # Embed descriptions (time this)
    print("  Embedding descriptions...")
    emb_start = time.perf_counter()
    embeddings = engine.embed_texts(texts)
    result.embedding_time_ms = (time.perf_counter() - emb_start) * 1000
    
    # Build vector store (time this)
    print("  Building vector store...")
    idx_start = time.perf_counter()
    store = VectorStore()
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        store.add(embedding=emb, text=text, meta={"scene_id": i})
    result.index_time_ms = (time.perf_counter() - idx_start) * 1000
    
    # Query benchmark
    print(f"  Running {len(queries)} queries...")
    for query in queries:
        query_start = time.perf_counter()
        
        # Query embedding
        q_vec = engine.embed_query(query)
        
        # Retrieve
        retrieve_results = store.search(q_vec, k=5)
        
        query_time = (time.perf_counter() - query_start) * 1000
        result.query_times.append(query_time)
        
        # Convert to compatible format
        formatted_results = []
        for res in retrieve_results:
            formatted_results.append({
                "score": res["score"],
                "text": res["text"],
                "meta": res["meta"]
            })
        
        result.query_results.append({
            "query": query,
            "results": formatted_results,
            "time_ms": query_time
        })
    
    print(f"  ✓ Completed (Index: {result.index_time_ms:.2f}ms, "
          f"Embeddings: {result.embedding_time_ms:.2f}ms, "
          f"Avg Query: {result.avg_query_time_ms():.2f}ms)")
    
    return result


# =====================================================================
# MARKDOWN REPORT GENERATION
# =====================================================================

def generate_markdown_report(results: List[BenchmarkResult], queries: List[str], log_path: str) -> str:
    """
    Generate comprehensive markdown report.
    """
    report = []
    report.append("# Retrieval System Benchmark Report\n")
    
    # Header info
    report.append(f"**Video Log:** {log_path}\n")
    report.append(f"**Number of Queries:** {len(queries)}\n")
    report.append(f"**Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Queries Tested:**\n")
    for i, q in enumerate(queries, 1):
        report.append(f"  {i}. \"{q}\"\n")
    
    report.append("\n---\n")
    
    # Overall Summary
    report.append("## Overall Summary\n")
    report.append("| System | Index Time (ms) | Embedding Time (ms) | Avg Query Time (ms) | Total Time (ms) |\n")
    report.append("|--------|-----------------|----------------------|-------------------|------------------|\n")
    
    for res in results:
        total = res.total_time_ms()
        report.append(f"| {res.name} | {res.index_time_ms:.2f} | {res.embedding_time_ms:.2f} | {res.avg_query_time_ms():.2f} | {total:.2f} |\n")
    
    report.append("\n")
    
    # Component Timing Comparison
    report.append("## Component Timing Breakdown\n")
    
    report.append("### Index Building Time\n")
    report.append("| System | Time (ms) |\n")
    report.append("|--------|----------|\n")
    for res in results:
        report.append(f"| {res.name} | {res.index_time_ms:.2f} |\n")
    report.append("\n")
    
    report.append("### Description Embedding Time\n")
    report.append("| System | Time (ms) |\n")
    report.append("|--------|----------|\n")
    for res in results:
        report.append(f"| {res.name} | {res.embedding_time_ms:.2f} |\n")
    report.append("\n")
    
    report.append("### Average Query Time\n")
    report.append("| System | Time (ms) |\n")
    report.append("|--------|----------|\n")
    for res in results:
        report.append(f"| {res.name} | {res.avg_query_time_ms():.2f} |\n")
    report.append("\n")
    
    # Per-query results
    report.append("---\n\n")
    report.append("## Per-Query Results\n")
    
    for query_idx, query in enumerate(queries, 1):
        report.append(f"### Query {query_idx}: \"{query}\"\n\n")
        
        for res in results:
            query_data = res.query_results[query_idx - 1]
            report.append(f"#### {res.name}\n")
            report.append(f"**Query Time:** {query_data['time_ms']:.2f} ms\n\n")
            
            report.append("**Top Results:**\n")
            report.append("| Rank | Scene Index | Score | Scene Description |\n")
            report.append("|------|-------------|-------|---------------|\n")
            
            for rank, res_item in enumerate(query_data["results"][:5], 1):
                if "text" in res_item:
                    # RAG format
                    text_preview = res_item["text"].replace("\n", " ")
                    score = res_item["score"]
                    scene_idx = res_item.get("meta", {}).get("scene_id", "N/A")
                else:
                    # Semantic search format
                    text_preview = res_item.get("description", "N/A").replace("\n", " ")
                    score = res_item.get("score", 0)
                    scene_idx = res_item.get("scene_index", "N/A")
                
                report.append(f"| {rank} | {scene_idx} | {score:.4f} | {text_preview}... |\n")
            
            report.append("\n")
    
    # Analysis & Insights
    report.append("---\n\n")
    report.append("## Analysis & Insights\n\n")
    
    # Find fastest system
    fastest = min(results, key=lambda r: r.total_time_ms())
    slowest = max(results, key=lambda r: r.total_time_ms())
    
    report.append(f"**Fastest System:** {fastest.name} ({fastest.total_time_ms():.2f} ms total)\n")
    report.append(f"**Slowest System:** {slowest.name} ({slowest.total_time_ms():.2f} ms total)\n")
    report.append(f"**Speed Difference:** {((slowest.total_time_ms() - fastest.total_time_ms()) / fastest.total_time_ms() * 100):.1f}% slower\n\n")
    
    # Embedding cost analysis
    report.append("### Embedding Method Comparison\n")
    ss_gemini = next((r for r in results if "Semantic Search" in r.name and "gemini" in r.name.lower()), None)
    ss_st = next((r for r in results if "Semantic Search" in r.name and "sentence-transformer" in r.name.lower()), None)
    rag_gemini = next((r for r in results if "RAG" in r.name and "gemini" in r.name.lower()), None)
    rag_st = next((r for r in results if "RAG" in r.name and "sentence-transformer" in r.name.lower()), None)
    
    if ss_gemini and ss_st:
        diff = ((ss_gemini.embedding_time_ms - ss_st.embedding_time_ms) / ss_st.embedding_time_ms * 100)
        faster = "SentenceTransformer" if diff > 0 else "Gemini"
        report.append(f"- **Semantic Search Embedding Speed:** {faster} is {abs(diff):.1f}% faster\n")
    
    if rag_gemini and rag_st:
        diff = ((rag_gemini.embedding_time_ms - rag_st.embedding_time_ms) / rag_st.embedding_time_ms * 100)
        faster = "SentenceTransformer" if diff > 0 else "Gemini"
        report.append(f"- **RAG Embedding Speed:** {faster} is {abs(diff):.1f}% faster\n")
    
    if ss_gemini and rag_gemini:
        diff = ((ss_gemini.embedding_time_ms - rag_gemini.embedding_time_ms) / rag_gemini.embedding_time_ms * 100)
        faster = "RAG" if diff > 0 else "Semantic Search"
        report.append(f"- **Gemini Embedding (Semantic Search vs RAG):** Same API but slight differences in preprocessing\n")
    
    report.append("\n")
    report.append("### Retrieval Performance by Method\n")
    for res in results:
        report.append(f"- **{res.name}** average query latency: {res.avg_query_time_ms():.2f} ms\n")
    
    report.append("\n")
    report.append("### Key Findings\n")
    report.append("- Index building is typically dominated by initial setup overhead\n")
    report.append("- Query embedding time depends on the selected embedding method\n")
    report.append("- SentenceTransformer is generally faster as it runs locally\n")
    report.append("- Gemini API has network latency but may offer better semantic understanding\n")
    
    return "".join(report)


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    print("=" * 70)
    print("RETRIEVAL SYSTEM COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Video Log: {VIDEO_LOG_PATH}")
    print(f"  Number of Queries: {NUM_QUERIES}")
    print(f"  Embedding Methods: Gemini, SentenceTransformer")
    print(f"  Systems: Semantic Search (2 methods), RAG (2 methods) = 4 total")
    
    # Ensure log reports directory exists
    LOG_REPORTS_DIR.mkdir(exist_ok=True)
    
    # Run benchmarks
    results = []
    
    try:
        # Semantic Search with Gemini
        results.append(benchmark_semantic_search("gemini", VIDEO_LOG_PATH, TEST_QUERIES))
        
        # Semantic Search with SentenceTransformer
        results.append(benchmark_semantic_search("sentence-transformer", VIDEO_LOG_PATH, TEST_QUERIES))
        
        # RAG with Gemini
        results.append(benchmark_rag("gemini", VIDEO_LOG_PATH, TEST_QUERIES))
        
        # RAG with SentenceTransformer
        results.append(benchmark_rag("sentence-transformer", VIDEO_LOG_PATH, TEST_QUERIES))
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        return
    
    # Generate report
    print("\n[GENERATING REPORT]")
    markdown_report = generate_markdown_report(results, TEST_QUERIES, VIDEO_LOG_PATH)
    
    # Save report
    report_path = LOG_REPORTS_DIR / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"✓ Report saved to: {report_path}")
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    # Print summary to console
    print("\n### QUICK SUMMARY ###\n")
    for res in results:
        print(f"{res.name}:")
        print(f"  Total Time: {res.total_time_ms():.2f} ms")
        print(f"  Index: {res.index_time_ms:.2f} ms | Embeddings: {res.embedding_time_ms:.2f} ms | Avg Query: {res.avg_query_time_ms():.2f} ms")
        print()


if __name__ == "__main__":
    main()

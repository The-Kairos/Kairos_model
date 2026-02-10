from loader import load_scenes
from embedder import TextEmbedder, format_embedding_text
from index import InMemoryIndex
from search import semantic_search
from timing import timed

import time

def main():
    vid_desc_path = "logs/pasta_20260206_121638"
    scenes = load_scenes(f"{vid_desc_path}.json")
    # contextual text instead of just descriptions
    texts = format_embedding_text(scenes)

    embedder = TextEmbedder(method="gemini")
    embeddings = embedder.embed_texts(texts)

    index = InMemoryIndex()
    index.build(scenes, embeddings)

    print(f"Indexed {len(scenes)} scenes.")

    while True:
        print("Enter query (type 'exit' to quit):")
        query = input("> ").strip()
        if not query:
            continue
        
        if query.lower() == "exit":
            break
        
        timings = {}
        total_start = time.perf_counter()

        with timed("query_embedding_ms", timings):
            q_emb = embedder.embed_query(query)

        results, timings = semantic_search(
            q_emb,
            index,
            top_k=5,
            timings=timings,
        )

        total_end = time.perf_counter()
        timings["total_latency_ms"] = (total_end - total_start) * 1000

        print("\nResults:")
        for r in results:
            print("-" * 60)
            print(f"scene_index: {r['scene_index']}")
            print(f"score: {r['score']:.4f}")
            print(f"description: {r['description']}")

        print("\nTiming:")
        for k, v in timings.items():
            print(f"{k}: {v:.2f} ms")
        print()

if __name__ == "__main__":
    main()
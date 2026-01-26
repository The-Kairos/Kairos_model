from vector_store import VectorStore
from embedding import embed_texts, embed_query

def main():
    store = VectorStore()

    # Mock scene texts
    scenes = [
        "A sponge character is sitting at a table holding a piece of paper.",
        "A dog is running outside in a park.",
        "An airplane is flying across the sky."
    ]

    # Embed scenes
    scene_embeddings = embed_texts(scenes)

    # Store them
    for text, emb in zip(scenes, scene_embeddings):
        store.add(emb, text)

    # Query
    question = "What is the character holding?"
    query_vec = embed_query(question)

    # Retrieve top 2
    results = store.search(query_vec, k=2)

    print("\nTop results:\n")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Text: {r['text']}\n")

if __name__ == "__main__":
    main()

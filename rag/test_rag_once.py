from vector_store import VectorStore
from embedding import embed_texts, embed_query
from generate import generate_answer

def main():
    store = VectorStore()

    # Mock video scenes
    scenes = [
        "From 00:01 to 00:05, a sponge character sits at a table holding a piece of paper.",
        "From 00:06 to 00:10, an airplane flies across the sky."
    ]

    # Embed and store scenes
    embeddings = embed_texts(scenes)
    for text, emb in zip(scenes, embeddings):
        store.add(emb, text)

    # Question that SHOULD be answerable
    # question = "What is the character holding?"
    question ="what color is the car"

    # Retrieve
    query_vec = embed_query(question)
    retrieved = store.search(query_vec, k=2)

    print("\nRetrieved context:\n")
    for r in retrieved:
        print(f"- {r['text']} (score={r['score']:.4f})")

    # Generate answer
    answer = generate_answer(question, retrieved)

    print("\nGenerated answer:\n")
    print(answer)

if __name__ == "__main__":
    main()
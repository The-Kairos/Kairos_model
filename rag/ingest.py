import json
from embedding import format_embedding_text, embed_texts
from vector_store import VectorStore

def ingest_video(log_path: str, store: VectorStore):
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    texts = format_embedding_text(logs["scenes"])
    embeddings = embed_texts(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        store.add(
            embedding=emb,
            text=text,
            meta={"scene_id": i}
        )

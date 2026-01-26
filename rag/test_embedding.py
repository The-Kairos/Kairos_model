from embedding import embed_texts, embed_query
import numpy as np

texts = [
    "A cartoon character sits at a table holding paper.",
    "An airplane flies across the sky."
]

scene_embs = embed_texts(texts)
query_emb = embed_query("What is the character holding?")

print("Scene embedding count:", len(scene_embs))
print("Embedding dimension:", scene_embs[0].shape)
print("Query dimension:", query_emb.shape)

# Norm checks
print("Scene norm:", np.linalg.norm(scene_embs[0]))
print("Query norm:", np.linalg.norm(query_emb))

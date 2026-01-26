import numpy as np

class VectorStore:
    """
    DROP-IN REPLACEMENT for Cosmos DB later.
    """

    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, embedding, text, meta=None):
        self.vectors.append(embedding)
        self.texts.append(text)
        self.metadata.append(meta or {})

    def search(self, query_vec, k=5):
        sims = np.dot(self.vectors, query_vec)
        top_idx = np.argsort(sims)[::-1][:k]

        return [
            {
                "text": self.texts[i],
                "score": float(sims[i]),
                "meta": self.metadata[i]
            }
            for i in top_idx
        ]
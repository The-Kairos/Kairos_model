import numpy as np
from typing import List, Dict

class InMemoryIndex:
    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.scenes: List[Dict] = []

    def build(self, scenes: List[Dict], embeddings: np.ndarray | list):
        # CHANGED: Handle embeddings as either numpy array or list
        if isinstance(embeddings, list):
            embeddings = np.stack(embeddings) if embeddings else np.array([])
        
        if len(scenes) != embeddings.shape[0]:
            raise ValueError("Scenes and embeddings count mismatch")

        self.scenes = scenes
        self.embeddings = embeddings

    def is_ready(self) -> bool:
        return self.embeddings is not None

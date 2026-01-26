import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# =========================================================
# Environment & Client Setup
# =========================================================

from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

EMBEDDING_MODEL = "gemini-embedding-001"

client = genai.Client(
    vertexai=True,   # set to False if not using Vertex-backed key
    api_key=GEMINI_API_KEY
)

# =========================================================
# Utilities
# =========================================================

def _normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector for cosine similarity.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# =========================================================
# Scene → Text Conversion
# =========================================================

def format_embedding_text(scenes: list[dict]) -> list[str]:
    """
    Convert scene metadata into embedding-ready text chunks.
    One chunk = one scene.
    """

    embedding_texts = []

    for scene in scenes:
        start_timecode = scene.get("start_timecode", "unknown")
        end_timecode = scene.get("end_timecode", "unknown")

        audio_speech = scene.get("audio_speech", "")
        audio_natural = scene.get("audio_natural", "")
        llm_scene_description = scene.get("llm_scene_description", "")

        # Collect unique YOLO object labels
        yolo_objects = scene.get("yolo_detections", {})
        object_labels = {
            obj.get("label")
            for frame_objs in yolo_objects.values()
            for obj in frame_objs
            if obj.get("label")
        }

        objects_str = ", ".join(sorted(object_labels)) if object_labels else "none"

        text = (
            f"From {start_timecode} to {end_timecode}, "
            f"{llm_scene_description}. "
            f"Visible objects include {objects_str}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )

        embedding_texts.append(text)

    return embedding_texts

# =========================================================
# Embedding Functions
# =========================================================

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Embed a list of texts (scenes).
    Returns a list of normalized numpy vectors.
    """

    if not texts:
        return []

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    embeddings = []
    for emb in result.embeddings:
        vec = np.array(emb.values, dtype=np.float32)
        embeddings.append(_normalize(vec))

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a normalized numpy vector.
    """

    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query
    )

    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    return _normalize(vec)

# =========================================================
# (Optional) Local Similarity Debug Helper
# =========================================================

def cosine_similarity_matrix(query_vec: np.ndarray, vectors: list[np.ndarray]):
    """
    Debug helper for local similarity checks (NOT used in final RAG).
    """
    mat = np.stack(vectors)
    return np.dot(mat, query_vec)

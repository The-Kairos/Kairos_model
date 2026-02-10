import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =========================================================
# CHANGED: Environment & API Setup (was: local SentenceTransformer)
# =========================================================

load_dotenv(Path(__file__).parent / "../.env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

EMBEDDING_MODEL = "gemini-embedding-001"

client = genai.Client(
    vertexai=True,
    api_key=GEMINI_API_KEY
)

# =========================================================
# CHANGED: Normalization utility (was: built-in to SentenceTransformer)
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
# CHANGED: Rich text formatting (was: no text preparation)
# =========================================================

def format_embedding_text(scenes: list[dict]) -> list[str]:
    """
    Convert scene metadata into embedding-ready text chunks.
    CHANGED: Now includes timecodes, objects, and audio in addition to description.
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
# CHANGED: Embedding functions (was: SentenceTransformer.encode)
# Now: Supports both Google Gemini API and SentenceTransformer
# =========================================================

class TextEmbedder:
    def __init__(self, method: str = "gemini"):
        """
        CHANGED: Constructor now accepts method parameter.
        Methods:
            - "gemini": Uses Google Gemini API (gemini-embedding-001)
            - "sentence-transformer": Uses SentenceTransformer (all-MiniLM-L6-v2)
        """
        if method not in ["gemini", "sentence-transformer"]:
            raise ValueError(f"Unknown method: {method}. Must be 'gemini' or 'sentence-transformer'")
        
        self.method = method
        self.model = None
        
        if method == "sentence-transformer":
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_texts(self, texts):
        """
        Embed a list of texts using selected method.
        CHANGED: Now supports both Gemini API and SentenceTransformer.
        Returns a 2D numpy array.
        """
        if not texts:
            return np.array([])

        if self.method == "gemini":
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts
            )

            embeddings = []
            for emb in result.embeddings:
                vec = np.array(emb.values, dtype=np.float32)
                embeddings.append(_normalize(vec))

            return np.stack(embeddings) if embeddings else np.array([])
        
        else:  # sentence-transformer
            return self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

    def embed_query(self, query: str):
        """
        Embed a single query string using selected method.
        CHANGED: Now supports both Gemini API and SentenceTransformer.
        Returns a normalized numpy vector (1D).
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if self.method == "gemini":
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=query
            )

            vec = np.array(result.embeddings[0].values, dtype=np.float32)
            return _normalize(vec)
        
        else:  # sentence-transformer
            return self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype(np.float32)

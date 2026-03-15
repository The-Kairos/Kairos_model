"""RAG: embed scenes/synopsis, retrieve top-k, generate answers."""

import json
import os

import numpy as np

from kairos.core.utils import load_prompt, print_prefixed
from kairos.llm.client import get_embedding_client
from kairos.llm.synopsis.render import _extract_timed_entry

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-pro")


def _ensure_embedding_client(client):
    """Return the given client, or build a default embedding client if None."""
    return client if client is not None else get_embedding_client()


# Formatting helpers


def format_scene_embedding(scenes: list) -> list:
    texts = []
    for scene in scenes:
        start_tc = scene.get("start_timecode")
        end_tc = scene.get("end_timecode")
        audio_speech = scene.get("audio_speech")
        audio_natural = scene.get("audio_natural")
        llm_desc = scene.get("llm_scene_description")

        yolo_objects = scene.get("yolo_detections", {})
        labels = set()
        if isinstance(yolo_objects, list):
            for obj in yolo_objects:
                label = obj.get("label")
                if label:
                    labels.add(label)
        elif isinstance(yolo_objects, dict):
            for yolo_scene in yolo_objects.values():
                for obj in yolo_scene:
                    label = obj.get("label")
                    if label:
                        labels.add(label)
        objects = ", ".join(sorted(labels)) or "none"

        texts.append(
            f"From {start_tc} to {end_tc}, {llm_desc}. "
            f"Visible objects include {objects}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )
    return texts


def format_paragraph_embedding(paragraphs) -> list:
    if not paragraphs:
        return []
    if isinstance(paragraphs, list):
        return [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
    return [p.strip() for p in paragraphs.split("\n\n") if p.strip()]


def format_synopsis_embedding(synopsis) -> list:
    if not synopsis:
        return []
    if isinstance(synopsis, dict):
        contexts = []
        summary = synopsis.get("summary")
        if isinstance(summary, str) and summary.strip():
            contexts.append(f"summary: {summary.strip()}")

        for key, text_key in [
            ("video_highlights", "highlight"),
            ("video_timeline", "event"),
            ("suggested_clips", "description"),
        ]:
            items_list = synopsis.get(key, [])
            if not isinstance(items_list, list):
                continue
            items = []
            for entry in items_list:
                if isinstance(entry, str) and entry.strip():
                    items.append(entry.strip())
                    continue
                ts, val = _extract_timed_entry(entry, text_key)
                if isinstance(val, str) and val.strip():
                    items.append(
                        f"{ts.strip()} - {val.strip()}"
                        if isinstance(ts, str) and ts.strip()
                        else val.strip()
                    )
            if items:
                contexts.append(f"{key}: " + " | ".join(items))

        questions = synopsis.get("questions", [])
        if isinstance(questions, list):
            items = []
            for qa in questions:
                if not isinstance(qa, dict):
                    continue
                q = qa.get("question")
                a = qa.get("answer")
                if isinstance(q, str) and q.strip():
                    items.append(
                        f"Q: {q.strip()} A: {a.strip()}"
                        if isinstance(a, str) and a.strip()
                        else f"Q: {q.strip()}"
                    )
            if items:
                contexts.append("questions: " + " | ".join(items))
        return contexts
    return format_paragraph_embedding(synopsis)


def build_contexts(checkpoint: dict) -> list:
    scenes = format_scene_embedding(checkpoint.get("scenes", []))
    synopsis = format_synopsis_embedding(checkpoint.get("synopsis", ""))
    return [c for c in (scenes + synopsis) if c and c.strip()]


# Embedding

MAX_EMBED_BATCH = 250


def embed_contexts(
    contexts: list, client=None, model=EMBEDDING_MODEL, batch_size=MAX_EMBED_BATCH
):
    client = _ensure_embedding_client(client)
    if not contexts:
        return []
    embeddings = []
    for start in range(0, len(contexts), batch_size):
        result = client.models.embed_content(
            model=model, contents=contexts[start : start + batch_size]
        )
        embeddings.extend([e.values for e in result.embeddings])
    return embeddings


def embed_question(question: str, client=None, model=EMBEDDING_MODEL):
    client = _ensure_embedding_client(client)
    result = client.models.embed_content(model=model, contents=question)
    return result.embeddings


def _embedding_values(embedding):
    """Extract raw values from an embedding object, dict, or passthrough."""
    if hasattr(embedding, "values"):
        return embedding.values
    if isinstance(embedding, dict) and "values" in embedding:
        return embedding["values"]
    return embedding


def _to_vector(e) -> np.ndarray:
    """Convert an embedding to a numpy float32 vector."""
    return np.array(_embedding_values(e), dtype=np.float32)


# Clustering


def find_optimal_k_elbow(
    embeddings: list, max_k: int = 20, random_state: int = 42
) -> int:
    from sklearn.cluster import KMeans

    X = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)
    n = X.shape[0]
    if n < 3:
        return 1
    max_test = min(n // 2, max_k)
    if max_test < 2:
        return 1
    inertias = []
    k_values = list(range(2, max_test + 1))
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    diffs = np.diff(inertias)
    accel = np.diff(diffs)
    optimal_k = k_values[np.argmax(accel) + 1] if len(accel) > 0 else k_values[0]
    return max(2, optimal_k)


def compute_kmeans_clusters(
    embeddings: list, num_clusters: int | None = None, random_state: int = 42
) -> dict:
    from sklearn.cluster import KMeans

    X = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)
    if X.shape[0] == 0:
        return {}
    if num_clusters is None:
        num_clusters = find_optimal_k_elbow(
            embeddings, max_k=20, random_state=random_state
        )
    k = min(num_clusters, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return {
        "algorithm": "kmeans",
        "num_clusters": int(k),
        "cluster_assignments": labels.tolist(),
        "centroids": km.cluster_centers_.tolist(),
    }


# Retrieval


def merge_retrieval(
    query_vec,
    scene_embeddings,
    contexts,
    cluster_metadata=None,
    k=10,
    top_c=3,
    alpha=0.3,
):
    s_vecs = np.array([_to_vector(e) for e in scene_embeddings], dtype=np.float32)
    base_sims = s_vecs.dot(query_vec)
    N = len(contexts)
    cluster_boost = np.zeros(N, dtype=np.float32)

    if cluster_metadata:
        centroids = np.array(cluster_metadata.get("centroids", []), dtype=np.float32)
        assignments = np.array(cluster_metadata.get("cluster_assignments", [-1] * N))
        if assignments.shape[0] == N and centroids.size:
            cluster_sims = centroids.dot(query_vec)
            top_ids = np.argsort(cluster_sims)[-min(top_c, centroids.shape[0]) :]
            for cid in top_ids:
                mask = assignments == int(cid)
                cluster_boost[mask] = np.maximum(cluster_boost[mask], cluster_sims[cid])
            maxb = cluster_boost.max() if cluster_boost.max() > 0 else 1.0
            cluster_boost = (cluster_boost / maxb) * alpha

    final = base_sims + cluster_boost
    top_idx = np.argsort(final)[-k:][::-1]
    return [(contexts[int(i)], float(final[int(i)])) for i in top_idx]


def get_top_k_similar(
    question_embedding,
    embeddings,
    contexts,
    k=5,
    debug=False,
    cluster_metadata=None,
    top_c=3,
    alpha=0.3,
):
    if isinstance(question_embedding, list):
        question_embedding = question_embedding[0]
    q_vec = np.array(_embedding_values(question_embedding), dtype=np.float32)
    if cluster_metadata is None:
        cluster_metadata = compute_kmeans_clusters(embeddings)
    top_matches = merge_retrieval(
        q_vec,
        embeddings,
        contexts,
        cluster_metadata=cluster_metadata,
        k=k,
        top_c=top_c,
        alpha=alpha,
    )
    if debug:
        for text, score in top_matches:
            print_prefixed("(RAG)", f"Score: {score:.4f} | Text: {text}")
    return top_matches


def create_answer(question, top_matches, client=None, model=GENERATION_MODEL):
    context = "\n".join([text for text, _ in top_matches])
    template = load_prompt("generate_answer.txt")
    prompt = template.format(context=context, question=question)
    if client is not None:
        return client.generate(prompt)
    raw_client = _ensure_embedding_client(None)
    response = raw_client.models.generate_content(model=model, contents=prompt)
    return response.text


# Persistence


def save_rag_embeddings(
    path, contexts, embeddings, model=EMBEDDING_MODEL, kmeans_clusters=None
):
    payload = {
        "model": model,
        "context_count": len(contexts),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "contexts": contexts,
        "embeddings": embeddings,
    }
    if kmeans_clusters:
        payload["kmeans_clusters"] = kmeans_clusters
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_rag_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG embedding file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_embedding(
    checkpoint: dict, output_path: str, model=EMBEDDING_MODEL, embedding_client=None
):
    contexts = build_contexts(checkpoint)
    if not contexts:
        raise ValueError("No contexts found in checkpoint to embed.")
    embedding_client = _ensure_embedding_client(embedding_client)
    embeddings = embed_contexts(contexts, client=embedding_client, model=model)
    kmeans_clusters = compute_kmeans_clusters(embeddings)
    payload = save_rag_embeddings(
        output_path, contexts, embeddings, model=model, kmeans_clusters=kmeans_clusters
    )
    return {
        "rag_path": output_path,
        "context_count": payload["context_count"],
        "embedding_dim": payload["embedding_dim"],
        "model": payload["model"],
    }

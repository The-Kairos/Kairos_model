"""RAG: embed scenes/synopsis, retrieve top-k, generate answers."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from kairos.core.utils import load_prompt, print_prefixed
from kairos.llm.client import get_embedding_client
from kairos.llm.synopsis.render import _extract_timed_entry

EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL: str = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-pro")


def _ensure_embedding_client(client: Any | None) -> Any:
    """Return the given client, or build a default embedding client if *None*.

    Args:
        client: An existing embedding client instance, or ``None``.

    Returns:
        Any: The supplied *client* when it is not ``None``, otherwise a
            freshly-built default embedding client from
            :func:`~kairos.llm.client.get_embedding_client`.
    """
    return client if client is not None else get_embedding_client()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_scene_embedding(scenes: list[dict[str, Any]]) -> list[str]:
    """Convert a list of scene dicts into human-readable embedding texts.

    Each scene is rendered as a single sentence that includes the
    timecodes, the LLM description, detected objects, background audio,
    and spoken dialogue.

    Args:
        scenes: A list of scene dictionaries. Expected keys include
            ``"start_timecode"``, ``"end_timecode"``,
            ``"llm_scene_description"``, ``"yolo_detections"``,
            ``"audio_speech"``, and ``"audio_natural"``.

    Returns:
        list[str]: One formatted string per scene.
    """
    texts: list[str] = []
    for scene in scenes:
        start_tc: str | None = scene.get("start_timecode")
        end_tc: str | None = scene.get("end_timecode")
        audio_speech: str | None = scene.get("audio_speech")
        audio_natural: str | None = scene.get("audio_natural")
        llm_desc: str | None = scene.get("llm_scene_description")

        yolo_objects: dict[str, Any] | list[Any] = scene.get("yolo_detections", {})
        labels: set[str] = set()
        if isinstance(yolo_objects, list):
            for obj in yolo_objects:
                label: str | None = obj.get("label")
                if label:
                    labels.add(label)
        elif isinstance(yolo_objects, dict):
            for yolo_scene in yolo_objects.values():
                for obj in yolo_scene:
                    label = obj.get("label")
                    if label:
                        labels.add(label)
        objects: str = ", ".join(sorted(labels)) or "none"

        texts.append(
            f"From {start_tc} to {end_tc}, {llm_desc}. "
            f"Visible objects include {objects}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )
    return texts


def format_paragraph_embedding(paragraphs: str | list[str] | None) -> list[str]:
    """Split paragraph text(s) into a list of non-empty strings.

    Accepts either a single newline-separated string or a pre-split
    list.  Empty / whitespace-only entries are discarded.

    Args:
        paragraphs: A multi-paragraph string, a list of paragraph
            strings, or ``None``.

    Returns:
        list[str]: Cleaned paragraph strings ready for embedding.
    """
    if not paragraphs:
        return []
    if isinstance(paragraphs, list):
        return [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
    return [p.strip() for p in paragraphs.split("\n\n") if p.strip()]


def format_synopsis_embedding(synopsis: dict[str, Any] | str | None) -> list[str]:
    """Format a synopsis (dict or raw string) into embeddable context chunks.

    When *synopsis* is a dictionary the function extracts the summary,
    video highlights, timeline events, suggested clips, and Q&A pairs,
    each rendered as a labelled string.  A plain string is forwarded to
    :func:`format_paragraph_embedding`.

    Args:
        synopsis: A synopsis dictionary with optional keys ``"summary"``,
            ``"video_highlights"``, ``"video_timeline"``,
            ``"suggested_clips"``, and ``"questions"``; a raw string; or
            ``None``.

    Returns:
        list[str]: Context strings suitable for embedding.
    """
    if not synopsis:
        return []
    if isinstance(synopsis, dict):
        contexts: list[str] = []
        summary: str | None = synopsis.get("summary")
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
            items: list[str] = []
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

        questions: list[dict[str, str]] = synopsis.get("questions", [])
        if isinstance(questions, list):
            items = []
            for qa in questions:
                if not isinstance(qa, dict):
                    continue
                q: str | None = qa.get("question")
                a: str | None = qa.get("answer")
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


def build_contexts(checkpoint: dict[str, Any]) -> list[str]:
    """Build a flat list of embedding-ready context strings from a checkpoint.

    Combines scene-level and synopsis-level formatted texts, filtering
    out any empty strings.

    Args:
        checkpoint: A checkpoint dictionary that may contain
            ``"scenes"`` and/or ``"synopsis"`` keys.

    Returns:
        list[str]: Non-empty context strings ready for embedding.
    """
    scenes: list[str] = format_scene_embedding(checkpoint.get("scenes", []))
    synopsis: list[str] = format_synopsis_embedding(checkpoint.get("synopsis", ""))
    return [c for c in (scenes + synopsis) if c and c.strip()]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

MAX_EMBED_BATCH: int = 250


def embed_contexts(
    contexts: list[str],
    client: Any | None = None,
    model: str = EMBEDDING_MODEL,
    batch_size: int = MAX_EMBED_BATCH,
) -> list[list[float]]:
    """Embed a list of context strings in batches.

    Long context lists are split into batches of *batch_size* and
    submitted to the Gemini embedding endpoint.

    Args:
        contexts: The text strings to embed.
        client: An optional pre-built embedding client.  When ``None`` a
            default client is created via
            :func:`_ensure_embedding_client`. Defaults to ``None``.
        model: The embedding model identifier. Defaults to
            :data:`EMBEDDING_MODEL`.
        batch_size: Maximum number of texts per API call.  Defaults to
            :data:`MAX_EMBED_BATCH` (250).

    Returns:
        list[list[float]]: A list of embedding vectors (each a list of
            floats), one per input context.
    """
    client = _ensure_embedding_client(client)
    if not contexts:
        return []
    embeddings: list[list[float]] = []
    for start in range(0, len(contexts), batch_size):
        result = client.models.embed_content(
            model=model, contents=contexts[start : start + batch_size]
        )
        embeddings.extend([e.values for e in result.embeddings])
    return embeddings


def embed_question(
    question: str,
    client: Any | None = None,
    model: str = EMBEDDING_MODEL,
) -> Any:
    """Embed a single question string.

    Args:
        question: The question text to embed.
        client: An optional pre-built embedding client.  When ``None`` a
            default client is created. Defaults to ``None``.
        model: The embedding model identifier. Defaults to
            :data:`EMBEDDING_MODEL`.

    Returns:
        Any: The embeddings object returned by the Gemini SDK (typically
            a list of embedding wrapper objects).
    """
    client = _ensure_embedding_client(client)
    result = client.models.embed_content(model=model, contents=question)
    return result.embeddings


def _embedding_values(embedding: Any) -> list[float]:
    """Extract raw float values from an embedding object, dict, or list.

    Supports objects with a ``.values`` attribute, dictionaries with a
    ``"values"`` key, or plain lists which are passed through unchanged.

    Args:
        embedding: An embedding in any of the supported formats.

    Returns:
        list[float]: The raw numeric embedding vector.
    """
    if hasattr(embedding, "values"):
        return embedding.values
    if isinstance(embedding, dict) and "values" in embedding:
        return embedding["values"]
    return embedding


def _to_vector(e: Any) -> np.ndarray:
    """Convert an embedding to a NumPy float32 vector.

    Args:
        e: An embedding in any format accepted by
            :func:`_embedding_values`.

    Returns:
        np.ndarray: A 1-D ``float32`` array.
    """
    return np.array(_embedding_values(e), dtype=np.float32)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def find_optimal_k_elbow(
    embeddings: list[Any],
    max_k: int = 20,
    random_state: int = 42,
) -> int:
    """Find the optimal number of clusters using the elbow method.

    Computes K-Means inertia for *k* = 2 … *max_k* (capped at
    ``n // 2``), then picks the *k* that maximises the second-order
    difference (acceleration) of the inertia curve.

    Args:
        embeddings: Embedding vectors in any format accepted by
            :func:`_to_vector`.
        max_k: Upper bound for the number of clusters to test.
            Defaults to ``20``.
        random_state: Random seed for reproducibility. Defaults to ``42``.

    Returns:
        int: The optimal cluster count (always ≥ 1).
    """
    from sklearn.cluster import KMeans

    X: np.ndarray = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)
    n: int = X.shape[0]
    if n < 3:
        return 1
    max_test: int = min(n // 2, max_k)
    if max_test < 2:
        return 1
    inertias: list[float] = []
    k_values: list[int] = list(range(2, max_test + 1))
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    diffs: np.ndarray = np.diff(inertias)
    accel: np.ndarray = np.diff(diffs)
    optimal_k: int = k_values[np.argmax(accel) + 1] if len(accel) > 0 else k_values[0]
    return max(2, optimal_k)


def compute_kmeans_clusters(
    embeddings: list[Any],
    num_clusters: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run K-Means clustering over embedding vectors.

    If *num_clusters* is ``None`` the optimal *k* is chosen automatically
    via :func:`find_optimal_k_elbow`.

    Args:
        embeddings: Embedding vectors in any format accepted by
            :func:`_to_vector`.
        num_clusters: Number of clusters. When ``None``, it is determined
            automatically. Defaults to ``None``.
        random_state: Random seed for reproducibility. Defaults to ``42``.

    Returns:
        dict[str, Any]: A dictionary with keys ``"algorithm"``,
            ``"num_clusters"``, ``"cluster_assignments"`` (list of ints),
            and ``"centroids"`` (list of lists).  Returns an empty dict
            when *embeddings* is empty.
    """
    from sklearn.cluster import KMeans

    X: np.ndarray = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)
    if X.shape[0] == 0:
        return {}
    if num_clusters is None:
        num_clusters = find_optimal_k_elbow(
            embeddings, max_k=20, random_state=random_state
        )
    k: int = min(num_clusters, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels: np.ndarray = km.fit_predict(X)
    return {
        "algorithm": "kmeans",
        "num_clusters": int(k),
        "cluster_assignments": labels.tolist(),
        "centroids": km.cluster_centers_.tolist(),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def merge_retrieval(
    query_vec: np.ndarray,
    scene_embeddings: list[Any],
    contexts: list[str],
    cluster_metadata: dict[str, Any] | None = None,
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3,
) -> list[tuple[str, float]]:
    """Retrieve the top-*k* contexts using cosine similarity + cluster boost.

    Base similarity scores (dot product between the query vector and
    each scene embedding) are optionally augmented with a cluster-level
    boost derived from the *top_c* closest cluster centroids.

    Args:
        query_vec: The query embedding as a 1-D float32 NumPy array.
        scene_embeddings: Embedding vectors for each context.
        contexts: The context strings corresponding to
            *scene_embeddings*.
        cluster_metadata: Optional K-Means metadata dict (as returned by
            :func:`compute_kmeans_clusters`). Defaults to ``None``.
        k: Number of top results to return. Defaults to ``10``.
        top_c: Number of closest cluster centroids to use for boosting.
            Defaults to ``3``.
        alpha: Weight of the cluster boost relative to the base
            similarity. Defaults to ``0.3``.

    Returns:
        list[tuple[str, float]]: A list of ``(context_text, score)``
            tuples sorted by descending score, with at most *k* entries.
    """
    s_vecs: np.ndarray = np.array(
        [_to_vector(e) for e in scene_embeddings], dtype=np.float32
    )
    base_sims: np.ndarray = s_vecs.dot(query_vec)
    N: int = len(contexts)
    cluster_boost: np.ndarray = np.zeros(N, dtype=np.float32)

    if cluster_metadata:
        centroids: np.ndarray = np.array(
            cluster_metadata.get("centroids", []), dtype=np.float32
        )
        assignments: np.ndarray = np.array(
            cluster_metadata.get("cluster_assignments", [-1] * N)
        )
        if assignments.shape[0] == N and centroids.size:
            cluster_sims: np.ndarray = centroids.dot(query_vec)
            top_ids: np.ndarray = np.argsort(cluster_sims)[
                -min(top_c, centroids.shape[0]) :
            ]
            for cid in top_ids:
                mask: np.ndarray = assignments == int(cid)
                cluster_boost[mask] = np.maximum(
                    cluster_boost[mask], cluster_sims[cid]
                )
            maxb: float = cluster_boost.max() if cluster_boost.max() > 0 else 1.0
            cluster_boost = (cluster_boost / maxb) * alpha

    final: np.ndarray = base_sims + cluster_boost
    top_idx: np.ndarray = np.argsort(final)[-k:][::-1]
    return [(contexts[int(i)], float(final[int(i)])) for i in top_idx]


def get_top_k_similar(
    question_embedding: Any,
    embeddings: list[Any],
    contexts: list[str],
    k: int = 5,
    debug: bool = False,
    cluster_metadata: dict[str, Any] | None = None,
    top_c: int = 3,
    alpha: float = 0.3,
) -> list[tuple[str, float]]:
    """Find the *k* most similar contexts to a question embedding.

    Wraps :func:`merge_retrieval` with automatic cluster computation
    when *cluster_metadata* is not supplied.

    Args:
        question_embedding: The question embedding (a single embedding
            object/list, or a one-element list of embeddings).
        embeddings: Embedding vectors for all contexts.
        contexts: The context strings corresponding to *embeddings*.
        k: Number of results to return. Defaults to ``5``.
        debug: If ``True``, prints each match to stdout.
            Defaults to ``False``.
        cluster_metadata: Optional pre-computed K-Means metadata.
            Defaults to ``None`` (computed on the fly).
        top_c: Number of top cluster centroids for boosting.
            Defaults to ``3``.
        alpha: Cluster-boost weight. Defaults to ``0.3``.

    Returns:
        list[tuple[str, float]]: ``(context_text, score)`` tuples sorted
            by descending score.
    """
    if isinstance(question_embedding, list):
        question_embedding = question_embedding[0]
    q_vec: np.ndarray = np.array(
        _embedding_values(question_embedding), dtype=np.float32
    )
    if cluster_metadata is None:
        cluster_metadata = compute_kmeans_clusters(embeddings)
    top_matches: list[tuple[str, float]] = merge_retrieval(
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


def create_answer(
    question: str,
    top_matches: list[tuple[str, float]],
    client: Any | None = None,
    model: str = GENERATION_MODEL,
) -> str:
    """Generate a natural-language answer from retrieved context.

    Loads the ``generate_answer.txt`` prompt template, fills it with the
    concatenated context and *question*, then calls either the provided
    *client* or falls back to a raw Gemini client.

    Args:
        question: The user's question.
        top_matches: ``(context_text, score)`` tuples as returned by
            :func:`get_top_k_similar`.
        client: An optional :class:`~kairos.llm.client.LLMClient`.
            When ``None``, a raw Gemini client is used directly.
            Defaults to ``None``.
        model: Model name used when falling back to the raw Gemini
            client. Defaults to :data:`GENERATION_MODEL`.

    Returns:
        str: The LLM-generated answer.
    """
    context: str = "\n".join([text for text, _ in top_matches])
    template: str = load_prompt("generate_answer.txt")
    prompt: str = template.format(context=context, question=question)
    if client is not None:
        return client.generate(prompt)
    raw_client: Any = _ensure_embedding_client(None)
    response = raw_client.models.generate_content(model=model, contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_rag_embeddings(
    path: str,
    contexts: list[str],
    embeddings: list[list[float]],
    model: str = EMBEDDING_MODEL,
    kmeans_clusters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save RAG embeddings and contexts to a JSON file.

    Creates parent directories as needed.

    Args:
        path: Destination file path.
        contexts: The context strings that were embedded.
        embeddings: The embedding vectors (lists of floats).
        model: The embedding model identifier stored as metadata.
            Defaults to :data:`EMBEDDING_MODEL`.
        kmeans_clusters: Optional K-Means cluster metadata to persist
            alongside the embeddings. Defaults to ``None``.

    Returns:
        dict[str, Any]: The full payload dictionary that was written to
            disk.
    """
    payload: dict[str, Any] = {
        "model": model,
        "context_count": len(contexts),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "contexts": contexts,
        "embeddings": embeddings,
    }
    if kmeans_clusters:
        payload["kmeans_clusters"] = kmeans_clusters
    folder: str = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_rag_embeddings(path: str) -> dict[str, Any]:
    """Load previously-saved RAG embeddings from a JSON file.

    Args:
        path: Path to the JSON file produced by
            :func:`save_rag_embeddings`.

    Returns:
        dict[str, Any]: The deserialised payload containing ``"model"``,
            ``"contexts"``, ``"embeddings"``, and optionally
            ``"kmeans_clusters"``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG embedding file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_embedding(
    checkpoint: dict[str, Any],
    output_path: str,
    model: str = EMBEDDING_MODEL,
    embedding_client: Any | None = None,
) -> dict[str, Any]:
    """Build, cluster, and persist RAG embeddings for a checkpoint.

    End-to-end convenience function that extracts contexts from a
    checkpoint dictionary, embeds them, clusters the embeddings with
    K-Means, and writes everything to *output_path*.

    Args:
        checkpoint: A checkpoint dictionary containing ``"scenes"``
            and/or ``"synopsis"`` data.
        output_path: File path where the embedding JSON will be saved.
        model: Embedding model identifier. Defaults to
            :data:`EMBEDDING_MODEL`.
        embedding_client: Optional pre-built embedding client.  When
            ``None``, one is created automatically.
            Defaults to ``None``.

    Returns:
        dict[str, Any]: A summary dictionary with keys ``"rag_path"``,
            ``"context_count"``, ``"embedding_dim"``, and ``"model"``.

    Raises:
        ValueError: If no embeddable contexts can be extracted from
            *checkpoint*.
    """
    contexts: list[str] = build_contexts(checkpoint)
    if not contexts:
        raise ValueError("No contexts found in checkpoint to embed.")
    embedding_client = _ensure_embedding_client(embedding_client)
    embeddings: list[list[float]] = embed_contexts(
        contexts, client=embedding_client, model=model
    )
    kmeans_clusters: dict[str, Any] = compute_kmeans_clusters(embeddings)
    payload: dict[str, Any] = save_rag_embeddings(
        output_path, contexts, embeddings, model=model, kmeans_clusters=kmeans_clusters
    )
    return {
        "rag_path": output_path,
        "context_count": payload["context_count"],
        "embedding_dim": payload["embedding_dim"],
        "model": payload["model"],
    }

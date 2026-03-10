import json
import os
import time
import textwrap
import numpy as np
from src.debug_utils import load_prompt
from dotenv import load_dotenv
from google import genai

load_dotenv("././.env")

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-pro"


def _get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables.")
    return genai.Client(vertexai=True, api_key=api_key)


def format_scene_embedding(scenes: list):
    embedding_texts = []
    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        end_timecode = scene.get("end_timecode")

        audio_speech = scene.get("audio_speech")
        audio_natural = scene.get("audio_natural")
        llm_scene_description = scene.get("llm_scene_description")

        yolo_objects = scene.get("yolo_detections", {})
        labels = set()

        if isinstance(yolo_objects, list):
            # New format: list of track summaries
            for obj in yolo_objects:
                label = obj.get("label")
                if label:
                    labels.add(label)
        elif isinstance(yolo_objects, dict):
            # Legacy format: dict of per-frame detections
            for yolo_scene in yolo_objects.values():
                for obj in yolo_scene:
                    label = obj.get("label")
                    if label:
                        labels.add(label)

        objects = ", ".join(sorted(labels))
        if not objects:
            objects = "none"

        embedding_texts.append(
            f"From {start_timecode} to {end_timecode}, {llm_scene_description}. "
            f"Visible objects include {objects}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )

    return embedding_texts


def format_paragraph_embedding(paragraphs):
    if not paragraphs:
        return []
    if isinstance(paragraphs, list):
        return [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
    return [p.strip() for p in paragraphs.split("\n\n") if p.strip()]


def _extract_timed_entry(item, text_key: str):
    if not isinstance(item, dict):
        return None, None
    if "timestamp" in item:
        return item.get("timestamp"), item.get(text_key)
    if len(item) == 1:
        timestamp, value = next(iter(item.items()))
        return timestamp, value
    return item.get("timestamp"), item.get(text_key)


def format_synopsis_embedding(synopsis):
    if not synopsis:
        return []
    if isinstance(synopsis, dict):
        contexts = []

        summary = synopsis.get("summary")
        if isinstance(summary, str) and summary.strip():
            contexts.append(f"summary: {summary.strip()}")

        highlights = synopsis.get("video_highlights", [])
        if isinstance(highlights, list):
            items = []
            for entry in highlights:
                if isinstance(entry, str) and entry.strip():
                    items.append(entry.strip())
                    continue
                ts, highlight = _extract_timed_entry(entry, "highlight")
                if isinstance(highlight, str) and highlight.strip():
                    if isinstance(ts, str) and ts.strip():
                        items.append(f"{ts.strip()} - {highlight.strip()}")
                    else:
                        items.append(highlight.strip())
            if items:
                contexts.append("highlights: " + " | ".join(items))

        timeline = synopsis.get("video_timeline", [])
        if isinstance(timeline, list):
            items = []
            for entry in timeline:
                ts, event = _extract_timed_entry(entry, "event")
                if isinstance(event, str) and event.strip():
                    if isinstance(ts, str) and ts.strip():
                        items.append(f"{ts.strip()} - {event.strip()}")
                    else:
                        items.append(event.strip())
            if items:
                contexts.append("timeline: " + " | ".join(items))

        clips = synopsis.get("suggested_clips", [])
        if isinstance(clips, list):
            items = []
            for entry in clips:
                ts, desc = _extract_timed_entry(entry, "description")
                if isinstance(desc, str) and desc.strip():
                    if isinstance(ts, str) and ts.strip():
                        items.append(f"{ts.strip()} - {desc.strip()}")
                    else:
                        items.append(desc.strip())
            if items:
                contexts.append("suggested_clips: " + " | ".join(items))

        questions = synopsis.get("questions", [])
        if isinstance(questions, list):
            items = []
            for qa in questions:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("question")
                answer = qa.get("answer")
                if isinstance(question, str) and question.strip():
                    if isinstance(answer, str) and answer.strip():
                        items.append(f"Q: {question.strip()} A: {answer.strip()}")
                    else:
                        items.append(f"Q: {question.strip()}")
            if items:
                contexts.append("questions: " + " | ".join(items))

        return contexts

    return format_paragraph_embedding(synopsis)


def build_contexts(checkpoint: dict):
    scenes = format_scene_embedding(checkpoint.get("scenes", []))
    synopsis = format_synopsis_embedding(checkpoint.get("synopsis", ""))
    return [c for c in (scenes + synopsis) if c and c.strip()]


MAX_EMBED_BATCH = 250  # Vertex AI embed_content supports up to 250 items per request.


def embed_contexts(contexts: list, client=None, model=EMBEDDING_MODEL, batch_size=MAX_EMBED_BATCH):
    if client is None:
        client = _get_gemini_client()
    if not contexts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    embeddings = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start:start + batch_size]
        result = client.models.embed_content(
            model=model,
            contents=batch,
        )
        embeddings.extend([embedding.values for embedding in result.embeddings])

    return embeddings


def embed_question(question: str, client=None, model=EMBEDDING_MODEL):
    if client is None:
        client = _get_gemini_client()
    result = client.models.embed_content(
        model=model,
        contents=question
    )
    return result.embeddings


def _embedding_values(embedding):
    if hasattr(embedding, "values"):
        return embedding.values
    return embedding


def _to_vector(e) -> np.ndarray:
    if hasattr(e, "values"):
        try:
            return np.array(e.values, dtype=np.float32)
        except Exception:
            pass
    if isinstance(e, dict) and "values" in e:
        return np.array(e["values"], dtype=np.float32)
    return np.array(e, dtype=np.float32)


def find_optimal_k_elbow(embeddings: list, max_k: int = 20, random_state: int = 42) -> int:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

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
    if len(accel) > 0:
        elbow_idx = np.argmax(accel) + 1
        optimal_k = k_values[elbow_idx]
    else:
        optimal_k = k_values[0]

    return max(2, optimal_k)


def compute_kmeans_clusters(embeddings: list, num_clusters: int = None, random_state: int = 42):
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required for KMeans clustering") from e

    X = np.array([_to_vector(e) for e in embeddings], dtype=np.float32)
    if X.shape[0] == 0:
        return {}

    if num_clusters is None:
        num_clusters = find_optimal_k_elbow(embeddings, max_k=20, random_state=random_state)

    k = min(num_clusters, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_.tolist()

    return {
        "algorithm": "kmeans",
        "num_clusters": int(k),
        "cluster_assignments": labels.tolist(),
        "centroids": centroids,
    }


def merge_retrieval(
    query_vec: np.ndarray,
    scene_embeddings: list,
    contexts: list,
    cluster_metadata: dict = None,
    k: int = 10,
    top_c: int = 3,
    alpha: float = 0.3,
):
    s_vecs = np.array([_to_vector(e) for e in scene_embeddings], dtype=np.float32)
    base_sims = s_vecs.dot(query_vec)

    N = len(contexts)
    cluster_boost = np.zeros(N, dtype=np.float32)

    if cluster_metadata:
        centroids = np.array(cluster_metadata.get("centroids", []), dtype=np.float32)
        assignments = np.array(cluster_metadata.get("cluster_assignments", [-1] * N))
        if assignments.shape[0] != N:
            print(f"Warning: cluster_assignments length ({assignments.shape[0]}) != contexts length ({N}), skipping cluster boost")
            cluster_metadata = None
        elif centroids.size:
            cluster_sims = centroids.dot(query_vec)
            C = centroids.shape[0]
            top_c = min(top_c, C)
            top_ids = np.argsort(cluster_sims)[-top_c:]
            for cid in top_ids:
                mask = (assignments == int(cid))
                cluster_boost[mask] = np.maximum(cluster_boost[mask], cluster_sims[cid])

            maxb = cluster_boost.max() if cluster_boost.max() > 0 else 1.0
            cluster_boost = (cluster_boost / maxb) * alpha

    final = base_sims + cluster_boost
    top_idx = np.argsort(final)[-k:][::-1]
    return [(contexts[int(i)], float(final[int(i)])) for i in top_idx]


def get_top_k_similar(question_embedding, embeddings, contexts, k=5, debug=False, cluster_metadata=None, top_c=3, alpha=0.3):
    if isinstance(question_embedding, list):
        question_embedding = question_embedding[0]
    q_vec = np.array(_embedding_values(question_embedding), dtype=np.float32)

    if cluster_metadata is None:
        cluster_metadata = compute_kmeans_clusters(embeddings, num_clusters=None)

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
            print(f"Score: {score:.4f} | Text: {text}\n")

    return top_matches


def create_answer(question, top_matches, client=None, model=GENERATION_MODEL):
    if client is None:
        client = _get_gemini_client()

    context = "\n".join([text for text, _ in top_matches])
    template = load_prompt("generate_answer.txt")
    prompt = template.format(context=context, question=question)

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text


def save_rag_embeddings(path, contexts, embeddings, model=EMBEDDING_MODEL, kmeans_clusters=None):
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_embedding(checkpoint: dict, output_path: str, model=EMBEDDING_MODEL):
    contexts = build_contexts(checkpoint)
    if not contexts:
        raise ValueError("No contexts found in checkpoint to embed.")

    client = _get_gemini_client()
    embeddings = embed_contexts(contexts, client=client, model=model)
    kmeans_clusters = compute_kmeans_clusters(embeddings, num_clusters=None)
    payload = save_rag_embeddings(output_path, contexts, embeddings, model=model, kmeans_clusters=kmeans_clusters)

    return {
        "rag_path": output_path,
        "context_count": payload["context_count"],
        "embedding_dim": payload["embedding_dim"],
        "model": payload["model"],
    }


def _ensure_parent_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _load_conversation(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("history"), list):
                return data["history"]
            if isinstance(data.get("items"), list):
                return data["items"]
    except json.JSONDecodeError:
        return []
    return []


def _write_conversation(path, items):
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def ask_rag(
    rag_path,
    show_k_context=False,
    k=10,
    generation_model=GENERATION_MODEL,
    conv_path=None,
    log_source=None,
    show_timings=False,
):
    data = load_rag_embeddings(rag_path)
    contexts = data.get("contexts", [])
    embeddings = data.get("embeddings", [])
    kmeans_clusters = data.get("kmeans_clusters")
    if kmeans_clusters is None:
        kmeans_clusters = compute_kmeans_clusters(embeddings, num_clusters=None)

    if not contexts or not embeddings:
        raise ValueError("RAG embedding file is missing contexts or embeddings.")

    client = _get_gemini_client()
    print("RAG ready. Ask questions (type 'exit' to quit).")

    conversation = None
    if conv_path:
        _ensure_parent_dir(conv_path)
        conversation = _load_conversation(conv_path)
        if not os.path.exists(conv_path):
            _write_conversation(conv_path, conversation)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        t0 = time.perf_counter()
        question_embedding = embed_question(question, client=client)
        t1 = time.perf_counter()

        top_matches = get_top_k_similar(
            question_embedding,
            embeddings,
            contexts,
            k=k,
            debug=False,
            cluster_metadata=kmeans_clusters,
            top_c=3,
            alpha=0.3,
        )
        t2 = time.perf_counter()

        answer = create_answer(question, top_matches, client=client, model=generation_model)
        t3 = time.perf_counter()

        print("=" * 80)
        print("Answer:")
        print(answer)

        if show_k_context:
            print("-" * 80)
            print("Top contexts:")
            for idx, (text, score) in enumerate(top_matches, 1):
                snippet = text.strip()
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                wrapped = textwrap.fill(snippet, width=96, subsequent_indent="   ")
                print(f"{idx}. score={score:.4f}")
                print(f"   {wrapped}")

        if show_timings:
            print("-" * 80)
            print(
                "Timings (sec): "
                f"embed={t1 - t0:.3f} | search={t2 - t1:.3f} | gen={t3 - t2:.3f}"
            )

        print("=" * 80)

        if conv_path:
            if conversation is None:
                conversation = _load_conversation(conv_path)
            entry = {
                "timeDate": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": question,
                "rag_answer": answer,
                "top_k_similar": [(float(score), text) for text, score in top_matches],
                "durations": {
                    "question_embedding": round(t1 - t0, 4),
                    "context_search": round(t2 - t1, 4),
                    "llm_generation": round(t3 - t2, 4),
                },
            }
            if log_source:
                entry["source"] = log_source

            conversation.append(entry)
            _write_conversation(conv_path, conversation)


def test():
    log_path = r".batch2\sheldon\checkpoint.json"
    if not os.path.exists(log_path):
        print(f"Demo checkpoint not found: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    output_path = os.path.join(os.path.dirname(log_path), "rag_embedding.json")
    make_embedding(logs, output_path)
    ask_rag(output_path, show_k_context=True)


# if __name__ == "__main__":
#     test()

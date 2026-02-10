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



def build_contexts(checkpoint: dict):
    scenes = format_scene_embedding(checkpoint.get("scenes", []))

    synopsis_text = checkpoint.get("synopsis", "")
    synopsis = format_paragraph_embedding(synopsis_text)

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


def get_top_k_similar(question_embedding, embeddings, contexts, k=5, debug=False):
    if isinstance(question_embedding, list):
        question_embedding = question_embedding[0]
    q_vec = np.array(_embedding_values(question_embedding), dtype=np.float32)
    s_vecs = np.array([_embedding_values(s) for s in embeddings], dtype=np.float32)
    similarities = np.dot(s_vecs, q_vec)

    top_indices = np.argsort(similarities)[::-1][:k]
    top_matches = [(contexts[i], similarities[i]) for i in top_indices]

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


def save_rag_embeddings(path, contexts, embeddings, model=EMBEDDING_MODEL):
    payload = {
        "model": model,
        "context_count": len(contexts),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "contexts": contexts,
        "embeddings": embeddings,
    }

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
    payload = save_rag_embeddings(output_path, contexts, embeddings, model=model)

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
            question_embedding, embeddings, contexts, k=k, debug=False
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

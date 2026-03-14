"""Interactive RAG question-answering session."""

import json
import os
import textwrap
import time

from kairos.llm.client import get_embedding_client
from kairos.llm.rag import (
    GENERATION_MODEL,
    compute_kmeans_clusters,
    create_answer,
    embed_question,
    get_top_k_similar,
    load_rag_embeddings,
)


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
    generation_client=None,
):
    data = load_rag_embeddings(rag_path)
    contexts = data.get("contexts", [])
    embeddings = data.get("embeddings", [])
    kmeans_clusters = data.get("kmeans_clusters")
    if kmeans_clusters is None:
        kmeans_clusters = compute_kmeans_clusters(embeddings)
    if not contexts or not embeddings:
        raise ValueError("RAG embedding file is missing contexts or embeddings.")

    embedding_client = get_embedding_client()
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
        question_embedding = embed_question(question, client=embedding_client)
        t1 = time.perf_counter()

        top_matches = get_top_k_similar(
            question_embedding,
            embeddings,
            contexts,
            k=k,
            cluster_metadata=kmeans_clusters,
        )
        t2 = time.perf_counter()

        answer = create_answer(
            question, top_matches, client=generation_client, model=generation_model
        )
        t3 = time.perf_counter()

        print("=" * 80)
        print("Answer:")
        print(answer)

        if show_k_context:
            print("-" * 80)
            print("Top contexts:")
            for idx, (text, score) in enumerate(top_matches, 1):
                snippet = (
                    text.strip()[:237] + "..."
                    if len(text.strip()) > 240
                    else text.strip()
                )
                print(f"{idx}. score={score:.4f}")
                print(f"   {textwrap.fill(snippet, width=96, subsequent_indent='   ')}")

        if show_timings:
            print("-" * 80)
            print(
                f"Timings (sec): embed={t1 - t0:.3f} "
                f"| search={t2 - t1:.3f} "
                f"| gen={t3 - t2:.3f}"
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

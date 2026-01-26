from vector_store import VectorStore
from ingest import ingest_video
from retrieve import retrieve_context
from generate import generate_answer

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "car_pyscene_blip_yolo_ASR_AST_GeminiPro25_20251123_180938.json"

def main():
    store = VectorStore()

    print("Ingesting video scenes...")
    ingest_video(LOG_PATH, store)

    print("RAG ready!!!!!!! Ask questions.\n")

    while True:
        question = input("Question (or 'exit'): ")
        if question.lower() == "exit":
            break

        contexts = retrieve_context(question, store)
        answer = generate_answer(question, contexts)
        print("\nAnswer:\n", answer, "\n")

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv

load_dotenv("../../env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# RAG settings
TOP_K = 5
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-pro"

# Placeholder for Cosmos
VECTOR_DB_BACKEND = "cosmos"  # future-proof

from embedding import embed_query
from config import TOP_K

def retrieve_context(question, store):
    q_vec = embed_query(question)
    results = store.search(q_vec, k=TOP_K)
    return results

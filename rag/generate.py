from google import genai
from config import GEMINI_API_KEY, GENERATION_MODEL

client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)

def generate_answer(question, retrieved_chunks):
    context = "\n".join([r["text"] for r in retrieved_chunks])

    prompt = f"""
You are answering questions about a video.
Use ONLY the information provided below.
If the answer is not present, say "Not shown in the video."

Video scenes:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text
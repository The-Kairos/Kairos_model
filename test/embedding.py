import json
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# https://ai.google.dev/gemini-api/docs/embeddings#generate-embeddings
load_dotenv(".env")
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(vertexai=True, api_key=api_key) # put "vertexai=True" if you're using Dr. Oussama's key

def format_embedding_text(scenes: list):
    embedding_texts = []
    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        end_timecode = scene.get("end_timecode")

        audio_speech = scene.get("audio_speech")
        audio_natural = scene.get("audio_natural")
        llm_scene_description = scene.get("llm_scene_description")
        
        yolo_objects = scene.get("yolo_detections", {})
        objects = ', '.join({obj.get('label') for yolo_scene in yolo_objects.values() for obj in yolo_scene})

        embedding_texts.append(
            f"From {start_timecode} to {end_timecode}, {llm_scene_description}. "
            f"Visible objects include {objects}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )
        
    return embedding_texts

def embed_scenes(scenes: list):
    # https://ai.google.dev/gemini-api/docs/embeddings#generate-embeddings
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=scenes
    )
    return [embedding for embedding in result.embeddings]

def embed_question(question: str):
    # https://ai.google.dev/gemini-api/docs/embeddings#generate-embeddings
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    return result.embeddings

def get_top_k_similar(question, embeddings, scenes, k=5, debug=False):
    q_vec = np.array(question[0].values)
    s_vecs = np.array([s.values for s in embeddings])
    similarities = np.dot(s_vecs, q_vec)

    top_indices = np.argsort(similarities)[::-1][:k]
    top_matches =  [(scenes[i], similarities[i]) for i in top_indices]
    
    if debug:
        for text, score in top_matches:
            print(f"Score: {score:.4f} | Text: {text}\n")
    
    return top_matches

def create_answer(question, top_matches):
    context = "\n".join([text for text, _ in top_matches])
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
        model="gemini-2.5-pro",
        contents=prompt
    )

    return response.text

log_path= r"logs\car_pyscene_blip_yolo_ASR_AST_GeminiPro25_20251123_180938.json"
with open(log_path, "r", encoding="utf-8") as f:
    logs = json.load(f)
scenes = format_embedding_text(logs.get("scenes"))
scene_embeddings = embed_scenes(scenes)

while True:
    question = input("Give questions about cartastrophy: ")
    question_embedding = embed_question(question)
    top_matches = get_top_k_similar(question_embedding, scene_embeddings, scenes, k=5)
    answer = create_answer(question, top_matches)
    print(answer)
    print()
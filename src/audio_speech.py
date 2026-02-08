# src/audio_speech.py
import os
import time
from openai import AzureOpenAI

_client = None

def get_azure_client():
    global _client
    if _client is None:
        key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not key or not endpoint:
            print("[WARNING] AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT not set in environment.")
            return None
            
        _client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=version
        )
    return _client

def extract_speech_asr_api(audio_path: str, enable_logs=True):
    client = get_azure_client()
    if client is None:
        return "Missing credentials", {"error": "Azure credentials not found"}
    
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        return "Missing deployment name", {"error": "AZURE_OPENAI_DEPLOYMENT not set"}

    timings = {}
    t0 = time.time()

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=deployment,
                file=audio_file
            )
        transcription = response.text
    except Exception as e:
        print(f"[ERROR] Azure Whisper failed: {e}")
        return f"Error: {str(e)}", {"error": str(e)}

    timings["asr_duration_sec"] = time.time() - t0
    
    if enable_logs:
        print(f"[Azure Whisper] {audio_path} -> {transcription[:60]}...")
        print("ASR timings:", timings)

    return transcription, timings

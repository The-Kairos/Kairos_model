# src/audio_asr_api.py
import os
import time
from openai import AzureOpenAI

AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)

def extract_speech_asr_api(audio_path: str, enable_logs=True):
    timings = {}
    t0 = time.time()

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=AZURE_DEPLOYMENT,
            file=audio_file
        )

    transcription = response.text
    timings["asr_duration_sec"] = time.time() - t0

    if enable_logs:
        print(f"[Azure Whisper] {audio_path} -> {transcription[:60]}...")
        print("ASR timings:", timings)

    return transcription, timings

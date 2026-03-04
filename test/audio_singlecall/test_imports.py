import sys
import time
print("Importing time, sys...")
import torch
print("Importing torch...")
import numpy as np
print("Importing numpy...")
import av
print("Importing av...")
import whisper
print("Importing whisper...")
import librosa
print("Importing librosa...")
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
print("Importing transformers...")

print("All imports successful. Testing Silero load...")
_silero_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
print("Silero load successful.")

print("Testing Whisper load...")
model = whisper.load_model("small")
print("Whisper load successful.")

print("Testing AST load...")
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
fe = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)
print("AST load successful.")

print("DONE.")

"""Small connectivity and inference checks before the unattended pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src.path_utils import load_kairos_env
load_kairos_env()
import torch
import main
import src.audio_detector
import src.audio_MIT_ast_parallel
import src.audio_whisper_parallel
import src.frame_captioning_blip
from src.rag_convo import embed_contexts

print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('GPU tensor test:', (torch.ones(4, device='cuda') * 2).sum().item(), flush=True)
response = main.get_llm_client().chat.completions.create(model=main.deployment,
    messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=5, temperature=0, timeout=40)
print('generation:', response.choices[0].message.content, 'model:', response.model, flush=True)
print('embedding dimensions:', len(embed_contexts(['Pilot connectivity check.'])[0]), flush=True)
# Exercise ASR on a short actual movie sample, rather than discovering credentials
# are unusable after hours of visual processing.
import imageio_ffmpeg
import numpy as np
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-ss', '600', '-i',
       str(ROOT / 'evaluation/pilot/movies/Yes Man (2008).mkv'), '-t', '20', '-vn',
       '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
audio = np.frombuffer(subprocess.check_output(cmd), dtype=np.float32)
segments = src.audio_whisper_parallel.transcribe_via_api(audio, 16000)
print('ASR sample:', json.dumps(segments, ensure_ascii=True), flush=True)
print('PREFLIGHT PASSED', flush=True)

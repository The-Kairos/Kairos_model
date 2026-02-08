import torch
import sys

# CRITICAL PATCH: Must patch BEFORE any transformers imports
# transformers_stream_generator (used by Qwen) tries to import BeamSearchScorer
# In newer transformers, it's in a different location
class MockBeamSearchScorer:
    """Mock BeamSearchScorer for compatibility with transformers_stream_generator"""
    pass

# Inject into transformers module namespace BEFORE it's imported
if 'transformers' not in sys.modules:
    import transformers
    transformers.BeamSearchScorer = MockBeamSearchScorer
else:
    sys.modules['transformers'].BeamSearchScorer = MockBeamSearchScorer

# Also try the real import path if it exists
try:
    from transformers.generation import BeamSearchScorer
    sys.modules['transformers'].BeamSearchScorer = BeamSearchScorer
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from PIL import Image

def load_vlm_model(model_id="Qwen/Qwen-VL-Chat"):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, fp16=True).eval()
    return model, tokenizer

def caption_image(model, tokenizer, pil_image, prompt="Describe the scene in detail."):
    temp_path = "temp_qwen.jpg"
    pil_image.save(temp_path)
    query = tokenizer.from_list([{'image': temp_path}, {'text': prompt}])
    inputs = tokenizer(query, return_tensors='pt').to(model.device)
    out = model.generate(**inputs)
    caption = tokenizer.decode(out[0], skip_special_tokens=True)
    if os.path.exists(temp_path): os.remove(temp_path)
    return caption.replace(prompt, "").strip()

def run_inference(model, tokenizer, scenes, video_path):
    import cv2
    vlm_scenes = []
    for scene in scenes:
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        mid_frame_idx = int(((scene["start_seconds"] + scene["end_seconds"]) / 2) * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = video.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = caption_image(model, tokenizer, pil_img)
            new_scene = dict(scene); new_scene["frame_captions"] = [caption]; vlm_scenes.append(new_scene)
        else:
            new_scene = dict(scene); new_scene["frame_captions"] = ["None"]; vlm_scenes.append(new_scene)
        video.release()
    return vlm_scenes

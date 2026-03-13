import os
import sys
import time
import json
import torch
import cv2
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_from_clip
from test_heavy_vlms.benchmark_utils import benchmark_inference

def load_vlm(model_name):
    """
    Loads one of the pre-defined heavy VLMs.
    """
    if model_name == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
        
        def infer(image):
            prompt = "[INST] <image>\nDescribe the scene in detail. Focus only on what is visually observable. Do not assume intentions or unseen events. Mention actions, objects, and interactions. [/INST]"
            inputs = processor(prompt, image, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=200)
            return processor.decode(output[0], skip_special_tokens=True)
        return infer

    elif model_name == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model_id = "OpenGVLab/InternVL-Chat-V1-5"
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        def infer(image):
            pixel_values = model.extract_feature(image)
            question = "Describe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions."
            response, history = model.chat(tokenizer, pixel_values, question, generation_config={"max_new_tokens": 200})
            return response
        return infer

    elif model_name == "qwenvl":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "Qwen/Qwen-VL-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, fp16=True).eval()

        def infer(image):
            # Qwen-VL needs a file path, we'll save the image temporarily
            temp_path = "temp_vlm_frame.jpg"
            image.save(temp_path)
            query = tokenizer.from_list_format([
                {'image': temp_path},
                {'text': 'Describe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions.'},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)
            return response
        return infer
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_video_test(video_path, model_name, num_scenes_to_test=3):
    """
    Runs the VLM test on the first few scenes of a video.
    """
    print(f"\n--- Testing {video_path.name} with {model_name} ---")
    
    # 1. Detect Scenes
    scenes = get_scene_list(str(video_path))
    scenes_to_test = scenes[:num_scenes_to_test]
    
    # 2. Load Model
    infer_func = load_vlm(model_name)
    
    results = []
    
    from PIL import Image

    for scene in scenes_to_test:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        mid = (start + end) / 2
        
        # 3. Sample Frame
        frames = sample_from_clip(str(video_path), idx, mid, mid + 0.1, num_frames=1, new_size=336)
        if not frames:
            continue
            
        frame_pil = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        
        # 4. Inference
        print(f"  Processing Scene {idx} ({start:.1f}s - {end:.1f}s)...")
        caption, metrics = benchmark_inference(infer_func, frame_pil)
        
        results.append({
            "scene_index": idx,
            "timestamp": mid,
            "caption": caption,
            "metrics": metrics
        })
        
    return results

if __name__ == "__main__":
    VIDEOS_DIR = PROJECT_ROOT / "Videos"
    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    
    # Example: Run on the first video with LLaVA
    if not videos:
        print("No videos found in Videos/ directory.")
    else:
        # Default to the first video and llava for a quick test
        # In a real scenario, we might loop through all or take arguments
        target_video = videos[0]
        model_to_test = "llava" # Change to "internvl" or "qwenvl" as needed
        
        try:
            test_results = run_video_test(target_video, model_to_test)
            print("\nFinal Results Summary:")
            print(json.dumps(test_results, indent=2))
        except Exception as e:
            print(f"Error during video test: {e}")

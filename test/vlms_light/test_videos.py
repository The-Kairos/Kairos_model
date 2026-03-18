"""
Quick per-video test for light VLMs (first N scenes). Same structure as test_heavy_vlms/test_videos.py.
Uses same Videos directory as heavy VLMs.
"""
import os
import sys
import json
import torch
import cv2
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_from_clip
from test_light_vlms.benchmark_utils import benchmark_inference


def load_vlm(model_name):
    """Load one of the light VLMs (same interface as heavy: returns infer(image) -> caption)."""
    if model_name == "siglip":
        import test_light_vlms.test_siglip as m
        model, processor = m.load_vlm_model()
        def infer(image):
            return m.caption_image(model, processor, image)
        return infer
    elif model_name == "mobilevlm":
        import test_light_vlms.test_mobilevlm as m
        model, processor = m.load_vlm_model()
        def infer(image):
            return m.caption_image(model, processor, image)
        return infer
    elif model_name == "tinyllava":
        import test_light_vlms.test_tinyllava as m
        model, processor = m.load_vlm_model()
        def infer(image):
            return m.caption_image(model, processor, image)
        return infer
    elif model_name == "blip2":
        import test_light_vlms.test_blip2 as m
        model, processor = m.load_vlm_model()
        def infer(image):
            return m.caption_image(model, processor, image)
        return infer
    else:
        raise ValueError(f"Unknown light model: {model_name}")


def run_video_test(video_path, model_name, num_scenes_to_test=3):
    """Run light VLM on first num_scenes_to_test scenes of the video."""
    print(f"\n--- Testing {video_path.name} with {model_name} ---")
    scenes = get_scene_list(str(video_path))
    scenes_to_test = scenes[:num_scenes_to_test]
    infer_func = load_vlm(model_name)
    results = []
    for scene in scenes_to_test:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        mid = (start + end) / 2
        frames = sample_from_clip(str(video_path), idx, mid, mid + 0.1, num_frames=1, new_size=336)
        if not frames:
            continue
        frame_pil = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        print(f"  Processing Scene {idx} ({start:.1f}s - {end:.1f}s)...")
        caption, metrics = benchmark_inference(infer_func, frame_pil)
        results.append({
            "scene_index": idx,
            "timestamp": mid,
            "caption": caption,
            "metrics": metrics,
        })
    return results


if __name__ == "__main__":
    VIDEOS_DIR = PROJECT_ROOT / "Videos"
    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    if not videos:
        print("No videos found in Videos/ directory.")
    else:
        target_video = videos[0]
        model_to_test = "siglip"
        try:
            test_results = run_video_test(target_video, model_to_test)
            print("\nFinal Results Summary:")
            print(json.dumps(test_results, indent=2))
        except Exception as e:
            print(f"Error during video test: {e}")

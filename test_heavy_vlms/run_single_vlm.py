#!/usr/bin/env python3
"""
Run a SINGLE VLM on a SINGLE video, save captions, then EXIT.
This forces OS-level memory cleanup.

Usage: python run_single_vlm.py <vlm_name> <video_path> <base_data_path> <output_dir>
"""
import sys
import os
import json
import time
import torch
import gc
from pathlib import Path
from PIL import Image
import cv2

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.frame_sampling import sample_from_clip

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def run_vlm_isolated(vlm_name, video_path, base_data_path, output_dir):
    """Run ONE VLM on ONE video, save results, exit."""
    
    print(f"\n{'='*80}")
    print(f"ISOLATED VLM RUN: {vlm_name} on {Path(video_path).stem}")
    print(f"{'='*80}\n")
    
    # Load base data (scenes with ASR/AST/YOLO)
    with open(base_data_path, 'r') as f:
        base_data = json.load(f)
    scenes = base_data['scenes']
    
    # Import VLM module
    if vlm_name == "llava":
        import test_llava_1_6 as vlm_module
    elif vlm_name == "internvl":
        import test_internvl as vlm_module
    elif vlm_name == "qwenvl":
        import test_qwenvl as vlm_module
    else:
        print(f"ERROR: Unknown VLM {vlm_name}")
        sys.exit(1)
    
    # Clear GPU memory before loading model
    print(f"[0/3] Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"    GPU memory cleared. Available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
    
    # Load model
    print(f"[1/3] Loading {vlm_name} model...")
    try:
        model, processor = vlm_module.load_vlm_model()
        print(f"    Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run inference on all scenes
    print(f"[2/3] Running inference on {len(scenes)} scenes...")
    captions = []
    
    for i, scene in enumerate(scenes):
        try:
            mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
            frames = sample_from_clip(str(video_path), scene["scene_index"], mid, mid+0.1, num_frames=1, new_size=336)
            
            if frames and len(frames) > 0 and frames[0] is not None:
                pil_img = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
                
                # Double-check PIL image is valid
                if pil_img is None or pil_img.size[0] == 0:
                    raise ValueError("Invalid PIL image")
                
                caption = vlm_module.caption_image(model, processor, pil_img)
                
                # Clean up the caption (remove prompt artifacts)
                if "ASSISTANT:" in caption:
                    caption = caption.split("ASSISTANT:")[-1].strip()
                
                captions.append({
                    "scene_index": scene["scene_index"],
                    "start_seconds": scene["start_seconds"],
                    "end_seconds": scene["end_seconds"],
                    "caption": caption
                })
                print(f"  Scene {i+1}/{len(scenes)}: {caption[:80]}...")
            else:
                captions.append({
                    "scene_index": scene["scene_index"],
                    "start_seconds": scene["start_seconds"],
                    "end_seconds": scene["end_seconds"],
                    "caption": "[No valid frame extracted]"
                })
                print(f"  Scene {i+1}/{len(scenes)}: No valid frame")
        except Exception as e:
            print(f"  Scene {i+1} FAILED: {e}")
            captions.append({
                "scene_index": scene["scene_index"],
                "start_seconds": scene["start_seconds"],
                "end_seconds": scene["end_seconds"],
                "caption": f"ERROR: {str(e)}"
            })
    
    # Save captions IMMEDIATELY
    print(f"[3/3] Saving captions to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = Path(output_dir) / "vlm_captions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "vlm": vlm_name,
            "video": Path(video_path).stem,
            "timestamp": time.time(),
            "captions": captions
        }, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"✓ Saved {len(captions)} captions to {output_file}")
    
    # Cleanup
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print(f"✓ {vlm_name} completed successfully. Exiting to free memory.\n")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_single_vlm.py <vlm_name> <video_path> <base_data_path> <output_dir>")
        sys.exit(1)
    
    vlm_name = sys.argv[1]
    video_path = sys.argv[2]
    base_data_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    run_vlm_isolated(vlm_name, video_path, base_data_path, output_dir)

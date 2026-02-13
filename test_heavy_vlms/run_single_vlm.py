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

    # Check if final result already exists
    os.makedirs(output_dir, exist_ok=True)
    caption_file = Path(output_dir) / "vlm_captions.json"
    if caption_file.exists():
        print(f"✓ Final results already exist at {caption_file}. Skipping.")
        sys.exit(0)
    
    # Load base data (scenes with ASR/AST/YOLO)
    print(f"[0.5/3] Loading base data from: {base_data_path}")
    with open(base_data_path, 'r') as f:
        base_data = json.load(f)
    scenes = base_data['scenes']
    
    # Import VLM module
    if vlm_name == "llava":
        import test_llava_1_6 as vlm_module
    elif vlm_name == "phi3v":
        import test_phi3v as vlm_module
    elif vlm_name == "instructblip":
        import test_instructblip as vlm_module
    elif vlm_name == "llava_mistral":
        import test_llava_1_6_mistral as vlm_module
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
    
    # Track metrics
    from src.system_metrics import get_system_usage
    start_metrics = get_system_usage()
    print(f"    Start Metrics: RAM={start_metrics['ram_used_gb']:.1f}GB, GPU={start_metrics.get('gpu_used_gb', 0):.1f}GB")
    
    captions = []
    processed_indices = set()
    
    # Check for partial progress
    tmp_file = Path(output_dir) / "vlm_captions_partial.json"
    if tmp_file.exists():
        try:
            with open(tmp_file, 'r') as f:
                partial_data = json.load(f)
                captions = partial_data.get("captions", [])
                processed_indices = {c["scene_index"] for c in captions}
                print(f"    Resuming from partial progress: {len(processed_indices)} scenes already processed.")
        except Exception as e:
            print(f"    Warning: Could not load partial file ({e}). Starting fresh.")

    for i, scene in enumerate(scenes):
        scene_idx = scene["scene_index"]
        if scene_idx in processed_indices:
            print(f"  Scene {i+1}/{len(scenes)}: [Skipping - already in partial results]")
            continue
            
        try:
            # Sample 2 frames per scene for LLaVA to avoid context overflow
            frames = sample_from_clip(
                str(video_path), 
                scene["scene_index"], 
                scene["start_seconds"], 
                scene["end_seconds"], 
                num_frames=2, 
                new_size=336
            )
            
            if frames and len(frames) > 0:
                # Convert to PIL images
                pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames if f is not None]
                
                if not pil_frames:
                    raise ValueError("No valid frames after conversion")
                
                # Use caption_frames instead of caption_image
                caption = vlm_module.caption_frames(model, processor, pil_frames)
                
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
                
                # --- Intermediate Save ---
                os.makedirs(output_dir, exist_ok=True)
                tmp_file = Path(output_dir) / "vlm_captions_partial.json"
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump({"captions": captions}, f, indent=2, cls=CustomJSONEncoder)
            else:
                captions.append({
                    "scene_index": scene["scene_index"],
                    "start_seconds": scene["start_seconds"],
                    "end_seconds": scene["end_seconds"],
                    "caption": "[No valid frames extracted]"
                })
                print(f"  Scene {i+1}/{len(scenes)}: No valid frames")
        except Exception as e:
            print(f"  Scene {i+1} FAILED: {e}")
            captions.append({
                "scene_index": scene["scene_index"],
                "start_seconds": scene["start_seconds"],
                "end_seconds": scene["end_seconds"],
                "caption": f"ERROR: {str(e)}"
            })

    end_metrics = get_system_usage()
    total_duration = end_metrics['timestamp'] - start_metrics['timestamp']
    avg_per_scene = total_duration / len(scenes) if scenes else 0
    print(f"    End Metrics: RAM={end_metrics['ram_used_gb']:.1f}GB, GPU={end_metrics.get('gpu_used_gb', 0):.1f}GB")
    print(f"    Total Duration: {total_duration:.1f}s ({avg_per_scene:.2f}s per scene)")
    
    # Perform Manual Fusion
    print(f"[3/3] Fusing data and saving results...")
    
    # Create lookup for base data
    scene_map = {s["scene_index"]: s for s in scenes}
    
    # List to hold final fused results
    final_results = []
    
    for cap in captions:
        idx = cap["scene_index"]
        base = scene_map.get(idx, {})
        
        # Gather components
        vlm_text = cap["caption"]
        asr_text = base.get("audio_speech", "")
        # AST is a list of strings
        ast_list = base.get("audio_natural", [])
        ast_text = ", ".join(ast_list) if isinstance(ast_list, list) else str(ast_list)
        
        # YOLO: could be list of dicts or list of track summaries
        yolo_raw = base.get("objects", [])
        yolo_text = ""
        if isinstance(yolo_raw, list):
            # If it's track summaries (dicts with 'label', 'movement', etc.)
            if yolo_raw and "movement" in yolo_raw[0]:
                from src.frame_obj_d_yolo import format_track_summary
                lines = [format_track_summary(x, style="compact") for x in yolo_raw[:5]] # Top 5 tracks
                yolo_text = "; ".join(lines)
            elif yolo_raw and "class" in yolo_raw[0]: # simple detection
                counts = {}
                for obj in yolo_raw:
                    cls = obj.get("class", "unknown")
                    counts[cls] = counts.get(cls, 0) + 1
                yolo_text = ", ".join([f"{k} x{v}" for k,v in counts.items()])
        
        # Manual Fusion String
        parts = []
        if vlm_text: parts.append(f"Visual: {vlm_text}")
        if asr_text and not asr_text.startswith("[ASR") and "Missing credentials" not in asr_text: 
            parts.append(f"Speech: {asr_text}")
        if ast_text: parts.append(f"Sounds: {ast_text}")
        if yolo_text: parts.append(f"Objects: {yolo_text}")
        
        full_description = " | ".join(parts)
        
        # Add to final result
        final_results.append({
            "scene_index": idx,
            "start_seconds": cap["start_seconds"],
            "end_seconds": cap["end_seconds"],
            "vlm_caption": vlm_text,
            "asr": asr_text,
            "ast": ast_list,
            "yolo": yolo_raw,
            "scene_description": full_description
        })
    
    # Save captions AND fused results
    os.makedirs(output_dir, exist_ok=True)
    
    caption_file = Path(output_dir) / "vlm_captions.json"
    with open(caption_file, 'w', encoding='utf-8') as f:
        json.dump({
            "vlm": vlm_name,
            "video": Path(video_path).stem,
            "timestamp": time.time(),
            "metrics": {
                "start": start_metrics,
                "end": end_metrics,
                "total_duration_sec": total_duration,
                "avg_sec_per_scene": avg_per_scene
            },
            "captions": captions,
            "fused_results": final_results
        }, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"✓ Saved results to {caption_file}")
    
    # Remove partial file if it exists
    tmp_file = Path(output_dir) / "vlm_captions_partial.json"
    if tmp_file.exists():
        os.remove(tmp_file)
    
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

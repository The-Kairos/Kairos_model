#!/usr/bin/env python3
"""
Process base data for a video: scenes, ASR, AST, YOLO.
This runs once per video and is cached.
"""
import sys
import os
import json
import time
import traceback
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env file for Azure credentials
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.scene_cutting import get_scene_list
from src.audio_utils import extract_scene_audio_ffmpeg
from src.audio_speech import extract_speech_asr_api
from src.audio_natural import extract_sounds
from src.frame_sampling import sample_from_clip, sample_fps
from src.frame_obj_d_yolo import detect_object_yolo

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

def process_base_data(video_path, output_file):
    """Process base data for a video."""
    
    video_path = Path(video_path)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_file.parent / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"Processing base data for {video_path.name}...")
    
    # Timing metrics
    metrics = {}
    overall_start = time.time()
    
    # 1. Scene Detection
    print("  [1/4] Scene detection...")
    t1 = time.time()
    scenes = get_scene_list(str(video_path))
    metrics["scene_detection"] = time.time() - t1
    print(f"    Found {len(scenes)} scenes ({metrics['scene_detection']:.1f}s)")
    
    # 2. Audio Processing (ASR only first)
    print("  [2/4] Audio processing (ASR)...")
    t2_asr = time.time()
    for scene in scenes:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        wav_path = audio_dir / f"scene_{idx:02d}.wav"
        
        # Extract audio - correct argument order: (input_video, output_wav, start_sec, end_sec)
        extract_scene_audio_ffmpeg(str(video_path), str(wav_path), start, end)
        
        # ASR - returns (transcription, timings) tuple, but may fail if credentials missing
        try:
            transcription, _ = extract_speech_asr_api(str(wav_path), enable_logs=False)
            scene["audio_speech"] = transcription
        except Exception as e:
            scene["audio_speech"] = f"[ASR unavailable: {str(e)[:50]}]"
            
    metrics["asr"] = time.time() - t2_asr
    print(f"    ASR completed ({metrics['asr']:.1f}s, {metrics['asr']/max(len(scenes), 1):.1f}s per scene)")
    
    # 2.5. AST (Natural Sounds) - process all scenes at once
    print("  [2.5/4] Audio processing (AST - Natural Sounds)...")
    t2_ast = time.time()
    try:
        scenes = extract_sounds(str(video_path), scenes, debug=False)
        metrics["ast"] = time.time() - t2_ast
        print(f"    AST completed ({metrics['ast']:.1f}s)")
    except Exception as e:
        print(f"    Warning: AST failed: {e}")
        metrics["ast"] = 0
        for scene in scenes:
            if "audio_natural" not in scene:
                scene["audio_natural"] = []
    
    # 3. YOLO Object Detection
    print("  [3/4] YOLO object detection...")
    t3 = time.time()
    try:
        # Sample frames with meta data (needed for tracking/motion)
        scenes = sample_fps(str(video_path), scenes, fps=1.0, new_size=320, store_meta=True)
        
        # Run YOLO detection with tracking
        # We assume yolov8n.pt is available or will be downloaded
        yolo_model_path = "yolov8n.pt" 
        scenes = detect_object_yolo(scenes, model_size=yolo_model_path, summary_key="objects")
        
        metrics["yolo"] = time.time() - t3
        print(f"    YOLO completed ({metrics['yolo']:.1f}s)")
        
    except Exception as e:
        print(f"    Warning: YOLO failed: {e}")
        traceback.print_exc()
        metrics["yolo"] = 0
        for scene in scenes:
            if "objects" not in scene:
                scene["objects"] = []
    
    # Strip frames from scenes before saving (too large for JSON)
    serializable_scenes = []
    for scene in scenes:
        scene_copy = scene.copy()
        if "frames" in scene_copy:
            del scene_copy["frames"]
        serializable_scenes.append(scene_copy)
    
    # 4. Save
    metrics["total"] = time.time() - overall_start
    print("  [4/4] Saving base data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "video": video_path.name,
            "scenes": serializable_scenes,
            "metrics": metrics,
            "timestamp": time.time()
        }, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"✓ Base data saved to {output_file}")
    print(f"  Total base processing: {metrics['total']:.1f}s")
    print(f"    - Scene detection: {metrics['scene_detection']:.1f}s")
    print(f"    - ASR: {metrics['asr']:.1f}s")
    print(f"    - AST: {metrics['ast']:.1f}s")
    print(f"    - YOLO: {metrics['yolo']:.1f}s")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_base.py <video_path> <output_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_file = sys.argv[2]
    
    process_base_data(video_path, output_file)

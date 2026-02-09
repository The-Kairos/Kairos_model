#!/usr/bin/env python3
"""
Process base data for a video: scenes, ASR, AST, YOLO.
This runs once per video and is cached.
"""
import sys
import os
import json
import time
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
from src.frame_sampling import sample_from_clip
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
    
    # 1. Scene Detection
    print("  [1/4] Scene detection...")
    scenes = get_scene_list(str(video_path))
    print(f"    Found {len(scenes)} scenes")
    
    # 2. Audio Processing (ASR only first)
    print("  [2/4] Audio processing (ASR)...")
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
    
    # 2.5. AST (Natural Sounds) - process all scenes at once
    print("  [2.5/4] Audio processing (AST - Natural Sounds)...")
    try:
        from src.audio_natural import extract_sounds
        scenes = extract_sounds(str(video_path), scenes, debug=False)
    except Exception as e:
        print(f"    Warning: AST failed: {e}")
        for scene in scenes:
            if "audio_natural" not in scene:
                scene["audio_natural"] = []
    
    # 3. YOLO Object Detection
    print("  [3/4] YOLO object detection...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # Use nano model for speed
        
        for scene in scenes:
            mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
            frames = sample_from_clip(str(video_path), scene["scene_index"], mid, mid+0.1, num_frames=1)
            if frames:
                # Run YOLO on single frame
                results = yolo_model.predict(frames[0], conf=0.5, verbose=False)
                detections = []
                for r in results:
                    if hasattr(r, "boxes"):
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            label = yolo_model.names[cls]
                            conf_score = float(box.conf[0])
                            detections.append({
                                "class": label,
                                "confidence": conf_score
                            })
                scene["objects"] = detections
            else:
                scene["objects"] = []
    except Exception as e:
        print(f"    Warning: YOLO failed: {e}")
        for scene in scenes:
            if "objects" not in scene:
                scene["objects"] = []
    
    # 4. Save
    print("  [4/4] Saving base data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "video": video_path.name,
            "scenes": scenes,
            "timestamp": time.time()
        }, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"✓ Base data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_base.py <video_path> <output_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_file = sys.argv[2]
    
    process_base_data(video_path, output_file)

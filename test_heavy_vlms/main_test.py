import os
import sys
import time
import json
import torch
import cv2
import copy
import numpy as np
import gc
import torch
import google.auth
from google.auth.transport.requests import Request
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root before ANY other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add project root and src to sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Local Pipeline Imports
from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_from_clip, sample_fps
from src.audio_utils import extract_scene_audio_ffmpeg
from src.audio_speech import extract_speech_asr_api
from src.audio_natural import extract_sounds
from src.frame_obj_d_yolo import detect_object_yolo
from src.scene_description import describe_scenes
from src.system_metrics import get_system_usage

# SETTINGS
ENABLE_VLM_LLAVA = True
ENABLE_VLM_QWEN = True
ENABLE_VLM_INTERNVL = True
# Disabled per user request (Gemini 401 and policy blocks)
USE_LLM_FUSION = False 
SUMMARY_key = "llm_scene_description"

# VLM Module Imports (Delayed to avoid VRAM issues during init)
def get_vlm_module(vlm_name):
    if vlm_name == "llava":
        import test_heavy_vlms.test_llava_1_6 as vlm
    elif vlm_name == "internvl":
        import test_heavy_vlms.test_internvl as vlm
    elif vlm_name == "qwenvl":
        import test_heavy_vlms.test_qwenvl as vlm
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}")
    return vlm

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def get_base_video_data(video_path, base_results_dir):
    """
    Performs Step 1-3 (Scene Detection, ASR, AST, YOLO) once per video.
    Returns the processed scenes and base timing metrics.
    """
    video_name = video_path.stem
    video_base_dir = base_results_dir / video_name
    video_base_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = video_base_dir / "base_data.json"
    if cache_file.exists():
        print(f"  [Cache] Loading base data for {video_name}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["scenes"], data["metrics"]

    print(f"  [Base] Processing shared steps for {video_name}...")
    audio_dir = video_base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    base_metrics = {}
    
    # 1. Scene Detection
    print("    [Step 1] Scene Detection...")
    t1 = time.time()
    scenes = get_scene_list(str(video_path))
    base_metrics["duration_scene_detection"] = time.time() - t1
    base_metrics["scene_count"] = len(scenes)

    # 2. Audio Processing (ASR & AST)
    print("    [Step 2] Audio Processing (ASR + Local AST)...")
    t2_asr = time.time()
    for scene in scenes:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        wav_path = audio_dir / f"scene_{idx:02d}.wav"
        extract_scene_audio_ffmpeg(str(video_path), str(wav_path), start, end)
        speech, _ = extract_speech_asr_api(str(wav_path), enable_logs=False)
        scene["audio_speech"] = speech
    base_metrics["duration_asr"] = time.time() - t2_asr

    t2_ast = time.time()
    extract_sounds(str(video_path), scenes, debug=False)
    base_metrics["duration_ast"] = time.time() - t2_ast

    # 3. YOLO Detection
    print("    [Step 3] YOLO Object Detection...")
    t3 = time.time()
    scenes = sample_fps(str(video_path), scenes, fps=1.0, new_size=320, store_meta=True)
    yolo_model_path = PROJECT_ROOT / "model" / "yolov8s.pt"
    scenes = detect_object_yolo(scenes, model_size=str(yolo_model_path), summary_key="yolo_detections")
    base_metrics["duration_yolo"] = time.time() - t3

    # CRITICAL: Strip the raw 'frames' (ndarray) before saving to JSON to save space and avoid errors
    # We only save metadata in base_data.json
    serializable_scenes = []
    for sc in scenes:
        sc_copy = copy.deepcopy(sc)
        if "frames" in sc_copy:
            del sc_copy["frames"]
        serializable_scenes.append(sc_copy)

    # Save to cache using custom encoder for safety
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"scenes": serializable_scenes, "metrics": base_metrics}, f, indent=2, cls=CustomJSONEncoder)
        
    return serializable_scenes, base_metrics

def get_gcp_token():
    try:
        credentials, project = google.auth.default()
        credentials.refresh(Request())
        return credentials, project
    except Exception as e:
        print(f"      [Auth] Could not fetch VM credentials: {e}")
        return None, None

def cleanup_memory(model=None, tokenizer=None):
    if model is not None:
        try:
            model = model.to('cpu')
            del model
        except:
            pass
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("    [Memory] GPU and System Cache cleared.")

def run_vlm_on_base(video_path, vlm_name, scenes, base_metrics, results_dir):
    """
    Performs Step 4-6 (VLM Captioning, LLM Fusion) per VLM.
    """
    video_name = video_path.stem
    output_dir = results_dir / vlm_name / video_name
    
    # --- SESSION RECOVERY (Skip if exists) ---
    if (output_dir / "vlm_captions.json").exists():
        print(f"  [Skip] Results already exist for {vlm_name} on {video_name}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We work on a deep copy to avoid polluting base scenes for other VLMs
    vlm_scenes = copy.deepcopy(scenes)
    vlm_metrics = copy.deepcopy(base_metrics)
    
    print(f"  [VLM] Running {vlm_name} on {video_name}...")
    vlm_mod = get_vlm_module(vlm_name)
    
    model, processor = None, None
    try:
        model, processor = vlm_mod.load_vlm_model()
        
        t4 = time.time()
        for scene in vlm_scenes:
            mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
            # Sample frames
            frames = sample_from_clip(str(video_path), scene["scene_index"], mid, mid+0.1, num_frames=1, new_size=336)
            if frames:
                pil_img = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
                caption = vlm_mod.caption_image(model, processor, pil_img)
                scene["frame_captions"] = [caption]
            else:
                scene["frame_captions"] = ["None"]
    except Exception as e:
        print(f"      [Error] VLM Phase Failed for {vlm_name}: {e}")
        return None, None
    finally:
        # Aggressive cleanup even if it fails
        cleanup_memory(model, processor)
        del model, processor
    
    vlm_metrics["duration_vlm_inference"] = time.time() - t4
    vlm_metrics["gpu_mem_vlm_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    # --- STEP 4.5: Explicitly Save VLM Captions (Requested) ---
    vlm_captions_data = {
        "vlm": vlm_name,
        "video": video_name,
        "captions": [
            {"idx": s["scene_index"], "caption": s["frame_captions"][0]}
            for s in vlm_scenes
        ]
    }
    with open(output_dir / "vlm_captions.json", "w", encoding="utf-8") as f:
        json.dump(vlm_captions_data, f, indent=2, cls=CustomJSONEncoder)
    print(f"    [Save] Captions saved to {output_dir / 'vlm_captions.json'}")
    
    # --- MEMORY CLEANUP ---
    cleanup_memory()

    # Intermediate Save: Save captions before fusion in case fusion fails
    intermediate_data = {
        "vlm": vlm_name,
        "video": video_name,
        "metrics": vlm_metrics,
        "scenes": [
            {
                "idx": s["scene_index"],
                "start": s["start_seconds"],
                "end": s["end_seconds"],
                "vlm_caption": s["frame_captions"][0],
                "fusion_description": "PENDING"
            }
            for s in vlm_scenes
        ]
    }
    with open(output_dir / "pipeline_results_partial.json", "w", encoding="utf-8") as f:
        json.dump(intermediate_data, f, indent=2, cls=CustomJSONEncoder)

    del model
    del processor
    torch.cuda.empty_cache()

    # 5. Scene Description (LLM Fusion)
    t5 = time.time()
    if not USE_LLM_FUSION:
        print("    [Step 5] Scene Description (LLM Fusion) BYPASSED.")
        print("      [Manual] Creating structured scene descriptions from raw data...")
        # Manual fusion: Concatenate all available data sources
        for s in vlm_scenes:
            parts = []
            
            # 1. VLM Caption
            if "frame_captions" in s and s["frame_captions"]:
                parts.append(f"Visual: {s['frame_captions'][0]}")
            
            # 2. Audio Speech (ASR)
            if "audio_speech" in s and s["audio_speech"]:
                parts.append(f"Speech: {s['audio_speech']}")
            
            # 3. Natural Sounds (AST)
            if "audio_natural" in s and s["audio_natural"]:
                sounds = ", ".join(s["audio_natural"]) if isinstance(s["audio_natural"], list) else s["audio_natural"]
                parts.append(f"Sounds: {sounds}")
            
            # 4. Detected Objects (YOLO)
            if "objects" in s and s["objects"]:
                objects = ", ".join([obj.get("class", "unknown") for obj in s["objects"][:5]])  # Top 5 objects
                parts.append(f"Objects: {objects}")
            
            # Combine into structured description
            s[SUMMARY_key] = " | ".join(parts) if parts else "No data available"
    else:
        print("    [Step 5] Scene Description (LLM Fusion)...")
        import google.generativeai as genai_legacy
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Check for VM auth credentials to bypass 401 "API keys not supported"
        creds, proj = get_gcp_token()
        if creds:
            print("      [Auth] Using VM Service Account (OAuth2) for Gemini.")
            genai_legacy.configure(credentials=creds)
        else:
            if not gemini_key:
                print("      [Warning] GEMINI_API_KEY not found in .env")
            if gemini_key:
                genai_legacy.configure(api_key=gemini_key)
        
        # We now rely on the robust per-scene retry logic inside describe_scenes()
        try:
            vlm_scenes = describe_scenes(
                vlm_scenes,
                genai_legacy,
                FLIP_key="frame_captions",
                ASR_key="audio_speech",
                AST_key="audio_natural",
                model="gemini-1.5-flash"
            )
        except Exception as e:
            print(f"      [Critical Error] LLM Fusion call failed for complete video: {e}")
            # We don't crash here, we just use the default "Fusion Failed" status from describe_scenes if partial

    vlm_metrics["duration_llm_fusion"] = time.time() - t5
    vlm_metrics["total_duration_sec"] = sum(v for k,v in vlm_metrics.items() if k.startswith("duration_"))
    vlm_metrics["system_usage_final"] = get_system_usage()
    
    result_data = {
        "vlm": vlm_name,
        "video": video_name,
        "metrics": vlm_metrics,
        "scenes": [
            {
                "idx": s["scene_index"],
                "start": s["start_seconds"],
                "end": s["end_seconds"],
                "vlm_caption": s["frame_captions"][0],
                "fusion_description": s.get("llm_scene_description", "Fusion Failed")
            }
            for s in vlm_scenes
        ]
    }
    
    with open(output_dir / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, cls=CustomJSONEncoder)
    
    # Clean up partial file on success
    partial_file = output_dir / "pipeline_results_partial.json"
    if partial_file.exists():
        partial_file.unlink()
        
    return vlm_metrics

if __name__ == "__main__":
    RESULTS_DIR = Path(__file__).parent / "results"
    BASE_DIR = RESULTS_DIR / "_shared_base_data"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    VIDEOS_DIR = PROJECT_ROOT / "Videos"
    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    VLMS = ["llava", "qwenvl", "internvl"]
    
    all_metrics = {}

    for video in videos:
        print(f"\n>>> VIDEO: {video.name}")
        try:
            # Step 1-3: Done once per video
            base_scenes, base_metrics = get_base_video_data(video, BASE_DIR)
            
            # Step 4-6: Done per VLM
            for vlm in VLMS:
                if vlm not in all_metrics: all_metrics[vlm] = {}
                
                # Check for existing results to allow resume
                results_file = RESULTS_DIR / vlm / video.stem / "pipeline_results.json"
                if results_file.exists():
                    print(f"  [Skip] {vlm} results already exist.")
                    with open(results_file, "r") as f:
                        old_data = json.load(f)
                        all_metrics[vlm][video.name] = old_data.get("metrics", {})
                    continue

                try:
                    metrics = run_vlm_on_base(video, vlm, base_scenes, base_metrics, RESULTS_DIR)
                    all_metrics[vlm][video.name] = metrics
                except Exception as e:
                    print(f"    FAILED VLM: {vlm} | Error: {e}")
        except Exception as e:
            print(f"  FAILED VIDEO: {video.name} | Error: {e}")

    with open(Path(__file__).parent / "vlm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, cls=CustomJSONEncoder)
        
    print("\n\nBENCHMARKING COMPLETE. Results saved to test_heavy_vlms/results/")

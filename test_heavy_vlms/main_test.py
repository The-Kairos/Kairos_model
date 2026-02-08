import os
import sys
import time
import json
import torch
import cv2
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
from src.frame_sampling import sample_from_clip, sample_frames
from src.audio_utils import extract_scene_audio_ffmpeg
from src.audio_speech import extract_speech_asr_api
from src.audio_natural import extract_sounds
from src.frame_obj_d_yolo import detect_object_yolo
from src.scene_description import describe_scenes
from src.system_metrics import get_system_usage

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

def run_pipeline_with_vlm(video_path, vlm_name, results_dir, gcloud_json):
    """
    Runs the full pipeline replacing BLIP with the specified VLM.
    """
    video_name = Path(video_path).stem
    output_dir = results_dir / vlm_name / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> PROCESSING: {video_name} | VLM: {vlm_name}")
    
    pipeline_metrics = {}
    t_start = time.time()

    # 1. Scene Detection
    print("  [Step 1/6] Scene Detection...")
    scenes = get_scene_list(str(video_path))
    pipeline_metrics["scene_count"] = len(scenes)

    # 2. Audio Processing (ASR & AST)
    print("  [Step 2/6] Audio Processing (ASR + Local AST)...")
    # AST still uses local model for now
    for scene in scenes:
        idx = scene["scene_index"]
        start, end = scene["start_seconds"], scene["end_seconds"]
        wav_path = audio_dir / f"scene_{idx:02d}.wav"
        extract_scene_audio_ffmpeg(str(video_path), str(wav_path), start, end)
        speech, asr_timings = extract_speech_asr_api(str(wav_path), enable_logs=False)
        scene["audio_speech"] = speech

    # Local AST processes all scenes in one pass efficiently
    extract_sounds(str(video_path), scenes, debug=False)

    # 3. YOLO Detection
    print("  [Step 3/6] YOLO Object Detection...")
    # Sample some frames for YOLO (using existing sample_fps logic or similar)
    # For testing, we'll just run on 1 frame per scene to be faster
    from src.frame_sampling import sample_fps
    scenes = sample_fps(str(video_path), scenes, fps=1.0, new_size=320, store_meta=True)
    scenes = detect_object_yolo(scenes, model_size="model/yolov8s.pt", summary_key="yolo_detections")

    # 4. Heavy VLM Captioning
    print(f"  [Step 4/6] Heavy VLM Captioning ({vlm_name})...")
    vlm_mod = get_vlm_module(vlm_name)
    model, processor = vlm_mod.load_vlm_model()
    
    for scene in scenes:
        # Sample one high-res frame for the VLM
        mid = (scene["start_seconds"] + scene["end_seconds"]) / 2
        frames = sample_from_clip(str(video_path), scene["scene_index"], mid, mid+0.1, num_frames=1, new_size=336)
        if frames:
            pil_img = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            caption = vlm_mod.caption_image(model, processor, pil_img)
            scene["frame_captions"] = [caption] # List format for describe_scenes compatibility
        else:
            scene["frame_captions"] = ["None"]

    # Clear memory after VLM usage to allow LLM/next VLM
    del model
    del processor
    torch.cuda.empty_cache()

    # 5. Scene Description (LLM Fusion)
    print("  [Step 5/6] Scene Description (LLM Fusion)...")
    # Using Gemini/GPT for fusion as in main.py
    # We'll assume LLM client is configured in main_test
    # This part requires the GEMINI_API_KEY
    from google import genai
    gemini_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(vertexai=True, api_key=gemini_key)
    
    scenes = describe_scenes(
        scenes,
        client,
        FLIP_key="frame_captions",
        ASR_key="audio_speech",
        AST_key="audio_natural",
        model="gemini-2.5-flash"
    )

    # 6. Save Results
    print("  [Step 6/6] Saving Results...")
    pipeline_metrics["total_duration_sec"] = time.time() - t_start
    pipeline_metrics["system_usage"] = get_system_usage()
    
    result_data = {
        "vlm": vlm_name,
        "video": video_name,
        "metrics": pipeline_metrics,
        "scenes": [
            {
                "idx": s["scene_index"],
                "start": s["start_seconds"],
                "end": s["end_seconds"],
                "vlm_caption": s["frame_captions"][0],
                "fusion_description": s.get("llm_scene_description", "")
            }
            for s in scenes
        ]
    }
    
    with open(output_dir / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return pipeline_metrics

if __name__ == "__main__":
    # load_dotenv() is now at the top
    
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    VIDEOS_DIR = PROJECT_ROOT / "Videos"
    videos = [v for v in VIDEOS_DIR.glob("*.mp4") if not v.name.startswith("_")]
    
    VLMS = ["llava", "qwenvl", "internvl"]
    GCLOUD_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    all_metrics = {}

    for vlm in VLMS:
        all_metrics[vlm] = {}
        for video in videos:
            try:
                metrics = run_pipeline_with_vlm(video, vlm, RESULTS_DIR, GCLOUD_JSON)
                all_metrics[vlm][video.name] = metrics
            except Exception as e:
                print(f"FAILED: {vlm} on {video.name} | Error: {e}")

    with open(Path(__file__).parent / "vlm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    print("\n\nBENCHMARKING COMPLETE. Results saved to test_heavy_vlms/results/")

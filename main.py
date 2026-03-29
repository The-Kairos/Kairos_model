from src.path_utils import load_kairos_env

# Load environment variables from project root dynamically
load_kairos_env(override=True)

from src.debug_utils import *
from src.log_utils import *
from src.redo_utils import apply_redo, REDO_CHOICES
import argparse
import os
import time
import logging
import warnings
from pathlib import Path

# --- Mute harmless native warnings (FFmpeg / H264 / OpenCV) ---
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["AV_LOG_LEVEL"] = "quiet"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*mmco: unref short failure.*")

import gc
import torch
import psutil
from src.path_utils import load_kairos_env, is_low_mem
from src.storage_utils import StorageManager

# --- Resource Policy Detection ---
LOW_MEM_MODE = is_low_mem()

def purge_memory(force=False):
    """Clear RAM and GPU VRAM if in low memory mode or forced."""
    if not LOW_MEM_MODE and not force:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[Kairos] Resource purge completed (LowMem: {LOW_MEM_MODE})")

use_gemini = False
if use_gemini:
    # =============== GEMINI FLASH 2.5 ===============
    api_key = os.getenv("GEMINI_API_KEY")

    from google import genai
    model_name= "gemini-2.5-flash"
    client = genai.Client(vertexai=True, api_key=api_key) # vertexai=True is needed if youre Dr. Oussama's key
else:
    # =============== GPT 4o ===============
    model_name = "gpt-4o"
    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")
    api_version = os.getenv("GPT_VERSION")

    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

# =========================================================
improve_motion_detection = True
pyscene_threshold   = 27        # sensitivity (lower = more cuts)
pyscene_shortest    = 2         # minimum scene length  
frames_per_scene    = 3         # number of frames sampled in each scene
frame_resolution    = 320       # resolution on the longest axis
blip_start_prompt   = "a video frame of"
blip_caption_len    = 50        # max blip caption length
blip_min_length     = 15        # min blip caption length
blip_num_beams      = 1         # small beam for structure
blip_do_sample      = True      # allow variation
blip_top_p          = 0.85       # nucleus sampling
blip_temperature    = 0.65       # mild randomness
blip_length_penalty = 1.0
blip_no_repeat_ngram_size = 3
blip_repetition_penalty = 1.1
yolo_action_fps     = 4
yolo_conf_thres     = 0.8       # YOLO confidence threshold
yolo_iou_thres      = 0.5       # YOLO IoU threshold for NMS
ast_target_sr       = 16000     # audio target sample rate for AST
asr_model_size      = 'medium'
asr_use_vad         = True      # enable VAD for ASR (whatever that means)
asr_target_sr       = 16000     # audio target sample rate for ASR
llm_scene_history   = 5         # number of prior scenes in LLM context
llm_chunk_len       = 20000     # max char len of combined scenes for one chunk
llm_summary_len     = 50000     # max char len of final context for synopsis
llm_cooldown_sec    = 0         # LLM cooldown between scene calls
rag_top_k_context   = 10        # top-k RAG scenes to include
# =========================================================
improve_motion_detection    = False
prioritize_speed            = False
process_static_videos       = False

if improve_motion_detection:
    pyscene_threshold   = 15     # more sensitive pyscene
    pyscene_shortest    = 0.5    # the minimum scene length  
    frames_per_scene    = 5      # more frames sampled per scene
    yolo_action_fps     = 8      # more frames sampled per scene
if prioritize_speed:
    pyscene_threshold   = 40     # less sensitive pyscene
    frames_per_scene    = 1      # number of frames sampled in each scene
    llm_chunk_len       = 500000 # x10 bigger story chunks
    llm_summary_len     = 500000 # x10 bigger context for synopsis
if process_static_videos:
    pyscene_threshold   = 3      # more sensitive pyscene
    frames_per_scene    = 1      # number of frames sampled in each scene
    yolo_action_fps     = 0.5
# todo: if 0 scenes are found, decrease pyscene_threshold automatically
# =========================================================

params = {
    "improve_motion_detection": improve_motion_detection,
    "prioritize_speed": prioritize_speed,
    "process_static_videos": process_static_videos,
    "pyscene_threshold": pyscene_threshold,
    "pyscene_shortest": pyscene_shortest,
    "frames_per_scene": frames_per_scene,
    "frame_resolution": frame_resolution,
    "blip_start_prompt": blip_start_prompt,
    "blip_caption_len": blip_caption_len,
    "blip_min_length": blip_min_length,
    "blip_num_beams": blip_num_beams,
    "blip_do_sample": blip_do_sample,
    "blip_top_p": blip_top_p,
    "blip_temperature": blip_temperature,
    "blip_length_penalty": blip_length_penalty,
    "blip_no_repeat_ngram_size": blip_no_repeat_ngram_size,
    "blip_repetition_penalty": blip_repetition_penalty,
    "yolo_conf_thres": yolo_conf_thres,
    "yolo_iou_thres": yolo_iou_thres,
    "ast_target_sr": ast_target_sr,
    "asr_model_size": asr_model_size,
    "asr_use_vad": asr_use_vad,
    "asr_target_sr": asr_target_sr,
    "llm_scene_history": llm_scene_history,
    "llm_chunk_len": llm_chunk_len,
    "llm_summary_len": llm_summary_len,
    "llm_cooldown_sec": llm_cooldown_sec,
    "rag_top_k_context": rag_top_k_context,
}

# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Process videos or run RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process videos")
    process.add_argument("path", nargs="?", help="Direct path to a video file for easy one-off runs")
    process.add_argument("--video", action="append", help="Blob name or path (repeatable)")
    process.add_argument("--all", action="store_true", help="Process all catalog videos")
    process.add_argument(
        "--filter",
        choices=["short", "medium", "long", "extra"],
        help="Inclusive length filter",
    )
    process.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include videos with unknown length when filtering",
    )
    process.add_argument(
        "--redo",
        nargs="+",
        action="append",
        choices=REDO_CHOICES,
        help="Redo a processing step; dependents are redone by default (repeatable).",
    )
    process.add_argument(
        "--redo-only",
        nargs="*",
        choices=REDO_CHOICES,
        help=(
            "Redo only the specified steps and stop afterward (no dependents). "
            "Provide steps here or use with --redo."
        ),
    )
    process.add_argument("--chat-id", help="MongoDB Chat ID to update")
    process.add_argument("--mongo-uri", help="MongoDB Connection URI")

    rag = subparsers.add_parser("rag", help="Run RAG for a single video")
    rag.add_argument("--video", required=True, help="Blob name or path")

    return parser.parse_args()

def main():
    VIDEOS_DIR = Path("Videos")
    CATALOG_PATH = VIDEOS_DIR / "_all_videos.json"
    PROCESSED_ROOT = Path("_processed")
    args = parse_args()
    
    # Smart Defaults for MongoDB
    # If MONGODB_URI is in env but not CLI, we use the env one automatically.
    # If chat_id is missing, StorageManager will generate a fallback inside per-video loop.
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    
    # Initialize StorageManager (We will update local_path and video_name inside the loop)
    storage_manager = StorageManager(
        chat_id=args.chat_id,
        mongo_uri=mongo_uri,
        local_path=None
    )

    redo_only_raw = getattr(args, "redo_only", None)
    redo_only_flag = redo_only_raw is not None
    redo_only_steps = redo_only_raw or []
    redo_steps = []

    def _flatten(values):
        flat = []
        if not values:
            return flat
        for value in values:
            if isinstance(value, (list, tuple)):
                flat.extend(value)
            else:
                flat.append(value)
        return flat

    if redo_only_steps:
        redo_steps = redo_only_steps
    elif getattr(args, "redo", None):
        redo_steps = _flatten(args.redo)
    if redo_only_flag and not redo_steps:
        raise SystemExit("--redo-only requires at least one step (via --redo-only or --redo)")
    catalog = load_video_catalog(CATALOG_PATH)
    all_vids = []
    if getattr(args, "path", None):
        all_vids.append(Path(args.path))
    if getattr(args, "video", None):
        all_vids.extend([Path(p) for p in args.video])
    
    # If both provided, use them; if only positional provided, use it.
    selected_paths = list(dict.fromkeys(all_vids)) if all_vids else select_videos(args, catalog, VIDEOS_DIR)
    
    if not selected_paths:
        raise SystemExit("No videos selected. Use: python3 main.py process <path>")
    if args.command == "rag" and len(selected_paths) != 1:
        raise SystemExit("RAG supports exactly one video. Use --video to pick one.")

    test_videos = {make_output_dir(p, PROCESSED_ROOT): str(p) for p in selected_paths}
    rag_only = args.command == "rag"
    if redo_only_steps and getattr(args, "redo", None):
        redo_steps = list(dict.fromkeys(redo_steps + _flatten(args.redo)))
    redo_only = redo_only_flag

    for output_dir, test_video in test_videos.items():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = f"{output_dir}/checkpoint.json"
        
        # Ensure StorageManager is correctly context-switched for this specific video
        # We always reset chat_id to None if args.chat_id is missing, 
        # allowing deterministic generation per-video.
        storage_manager.__init__(
            chat_id=args.chat_id, 
            mongo_uri=mongo_uri, 
            local_path=Path(checkpoint_path),
            video_name=test_video
        )

        if rag_only:
            rag_path = f"{output_dir}/rag_embedding.json"
            checkpoint_path = f"{output_dir}/checkpoint.json"
            if not os.path.exists(rag_path):
                print(f"RAG embedding not found: {rag_path}. Run process first.")
                continue
            ask_rag(
                rag_path=rag_path,
                show_k_context=True,
                k=rag_top_k_context,
                conv_path=f"{output_dir}/conversation_history.json",
                log_source=checkpoint_path,
                show_timings=False,
            )
            continue

        log = initiate_log(
            video_path=test_video,
            run_description="Test run for video processing pipeline.",
            params=params,
        )

        # I added checkpoints so if you wanna redo the whole process,
        # youd have to delete the checkpoint json in the path below
        checkpoint = storage_manager.read_checkpoint() # if deleted it will return a {}
        checkpoint.setdefault("steps", {})
        step = checkpoint["steps"]
        if redo_steps:
            checkpoint, redo_info = apply_redo(
                checkpoint=checkpoint,
                output_dir=output_dir,
                redo_steps=redo_steps,
                redo_only=redo_only,
            )
            if redo_info.get("changed") and "scenes" in checkpoint:
                storage_manager.save_checkpoint(checkpoint)

        # Catch-up reporting for cached steps to ensure MongoDB status is current
        if storage_manager.is_remote and checkpoint.get("scenes"):
            last_scene = checkpoint["scenes"][-1]
            if "synopsis" in checkpoint:
                storage_manager.update_pipeline_state("synopsis_generation", 95)
            elif "narratives" in checkpoint:
                storage_manager.update_pipeline_state("narrative_synthesis", 90)
            elif "llm_scene_description" in last_scene:
                storage_manager.update_pipeline_state("scene_description", 85)
            elif "audio_natural" in last_scene:
                storage_manager.update_pipeline_state("sound_analysis", 75)
            elif "audio_speech" in last_scene:
                storage_manager.update_pipeline_state("speech_transcription", 65)
            elif "yolo_detections" in last_scene:
                storage_manager.update_pipeline_state("object_detection", 50)
            elif "frame_captions" in last_scene:
                storage_manager.update_pipeline_state("frame_captioning", 40)
            elif "scenes" in checkpoint:
                storage_manager.update_pipeline_state("scene_detection", 10)

        if not checkpoint.get("scenes"):
            print("")
            print_section("Running PysceneDetect...")
            checkpoint["scenes"], step['get_scene_list'] = get_scene_list_log(
                input_video_path=test_video,
                threshold = pyscene_threshold,
                min_scene_sec= pyscene_shortest,
            )
            storage_manager.update_pipeline_state("scene_detection", 10)
            see_scenes_cuts(df=checkpoint["scenes"])
            time.sleep(10)

            print("")
            print(f"Saving clips in: {output_dir}/.clips")
            checkpoint["scenes"], step['save_clips'] = save_clips_log(
                video_path=test_video,
                scenes=checkpoint["scenes"],
                output_dir=f"{output_dir}/.clips",
            )
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()

        if "frame_captions" not in checkpoint["scenes"][-1].keys():
            print(f"Saving sampled frames in: {output_dir}/.frames")
            checkpoint["scenes"], step['sample_frames'] = sample_frames_log(
                input_video_path=test_video,
                scenes=checkpoint["scenes"],
                num_frames = frames_per_scene,
                new_size = frame_resolution,
                output_dir=f"{output_dir}/.frames",
            )
            storage_manager.update_pipeline_state("frame_sampling", 30)
            time.sleep(10)

            print("")
            print_section("Running BLIP...")
            checkpoint["scenes"], step['caption_frames'] = caption_frames_log(
                scenes=checkpoint["scenes"],
                prompt= blip_start_prompt,
                max_length=blip_caption_len,
                min_length=blip_min_length,
                num_beams=blip_num_beams,
                do_sample=blip_do_sample,
                top_p=blip_top_p,
                temperature=blip_temperature,
                length_penalty=blip_length_penalty,
                no_repeat_ngram_size=blip_no_repeat_ngram_size,
                repetition_penalty=blip_repetition_penalty,
                debug=True,
            )
            storage_manager.update_pipeline_state("frame_captioning", 40)
            time.sleep(10)
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()
        

        if "yolo_detections" not in checkpoint["scenes"][-1].keys():
            if "yolo_frames" not in checkpoint["scenes"][-1].keys():
                print("")
                print(f"Saving sampled fps in: {output_dir}/.fps")
                checkpoint["scenes"], step['sample_fps'] = sample_fps_log(
                    input_video_path=test_video,
                    scenes=checkpoint["scenes"],
                    fps=yolo_action_fps,
                    new_size=frame_resolution,
                    output_dir=f"{output_dir}/.fps",
                    frames_key="yolo_frames",
                    frame_paths_key="yolo_frame_paths",
                )
            time.sleep(10)

            print("")
            print_section("Running YOLOv8...")
            checkpoint["scenes"], step['detect_object_yolo'] = detect_object_yolo_log(
                scenes=checkpoint["scenes"],
                model_size="model/yolov8s.pt",
                conf=yolo_conf_thres,
                iou=yolo_iou_thres,
                output_dir=f"{output_dir}/.yolo",
                frame_key="yolo_frames",
                summary_key="yolo_detections",
                debug=True,
            )
            storage_manager.update_pipeline_state("object_detection", 50)
            time.sleep(10)
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()

        scan_result = None
        def get_scan_result():
            print("")
            print_section("Running Audio Pre-Scan...")
            scan_result, step['audio_scan'] = scan_audio_log(
                video_path=test_video,
                scenes=checkpoint["scenes"],
                target_sr=asr_target_sr,
                debug=True,
            )
            time.sleep(10)
            return scan_result

        if "audio_speech" not in checkpoint["scenes"][-1].keys():
            if scan_result is None: scan_result = get_scan_result()
            print("")
            print_section("Running Whisper (Parallel)...")
            checkpoint["scenes"], step['asr_timings'] = extract_speech_log(
                scenes=checkpoint["scenes"],
                scan_result=scan_result,
                model_size=asr_model_size,
                use_vad=asr_use_vad,
                language=None,
                parallel=True,
                use_api=True,
                force_cpu=True,
                debug=True,
            )
            storage_manager.update_pipeline_state("speech_transcription", 65)
            time.sleep(10)
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()

        if "audio_natural" not in checkpoint["scenes"][-1].keys():
            if scan_result is None: scan_result = get_scan_result()

            # Dynamic Resource Scaling: Standardized via Environment Variables
            # Use MAX_KAIROS_WORKERS from .env.local to control concurrency.
            # Default to 2 for safety; scale up to 4+ on high-performance VMs.
            env_workers = os.environ.get("MAX_KAIROS_WORKERS")
            ast_workers = int(env_workers) if env_workers else 2

            if env_workers:
                print(f"[Kairos] Using {ast_workers} workers from environment override.")
            else:
                print(f"[Kairos] MAX_KAIROS_WORKERS not set. Defaulting to safe value: {ast_workers}")

            print("")
            print_section("Running MIT AST (Parallel)...")
            checkpoint["scenes"], step['ast_timings'] = extract_sounds_log(
                scenes=checkpoint["scenes"],
                scan_result=scan_result,
                max_workers=ast_workers,
                use_processes=True,
                force_cpu=False,
                debug=True,
            )
            storage_manager.update_pipeline_state("sound_analysis", 75)
            time.sleep(10)
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()

        if "llm_scene_description" not in checkpoint["scenes"][-1].keys():
            print("")
            print_section("Running GPT4o Scene Descriptions...")
            checkpoint["scenes"], step['describe_scenes'] = describe_scenes_log(
                scenes=checkpoint["scenes"],
                client=client,
                hist_size= llm_scene_history,
                YOLO_key="yolo_detections",
                FLIP_key="frame_captions",
                ASR_key="audio_speech",
                AST_key="audio_natural",
                SUMMARY_key="llm_scene_description",
                model=model_name,
                prompt_path="prompts/describe_scene.txt",
                cooldown_sec=llm_cooldown_sec,
                debug=True,
                video_path=test_video,
            )
            storage_manager.update_pipeline_state("scene_description", 85)
            time.sleep(10)
            storage_manager.save_checkpoint(checkpoint)
            purge_memory()
            

        if "narratives" not in checkpoint:
            print("")
            print_section("Running GPT4o Summary narrative...")
            checkpoint, step['summarize_scenes'] = summarize_scenes_log(
                client=client,
                deployment=deployment,
                scenes=checkpoint["scenes"],
                chunk_size = llm_chunk_len,
                summary_len = llm_summary_len,
                debug=True,
                output_dir=output_dir,
            )
            narratives = checkpoint.get("narratives", [])
            if narratives:
                last = narratives[-1]
                narrative_path = Path(output_dir) / f"narrative_{len(narratives)}_len_{last['narrative_len']}.txt"
                print(f"Saving narrative in: {narrative_path}")
            storage_manager.update_pipeline_state("narrative_synthesis", 90)
            storage_manager.save_checkpoint(checkpoint)

        if "synopsis" not in checkpoint:
            print("")
            print_section("Running GPT4o Synopsis generation...")
            checkpoint, step['synthesize_synopsis'] = synthesize_synopsis_log(
                client=client,
                deployment=deployment,
                data=checkpoint,
                debug=True,
                output_dir=output_dir
            )
            storage_manager.update_pipeline_state("synopsis_generation", 95)
            storage_manager.save_checkpoint(checkpoint=checkpoint)

        rag_path = Path(output_dir) / "rag_embedding.json"
        
        # 1. Generate embeddings if they don't exist
        if not rag_path.exists():
            checkpoint["rag_embedding"], step['make_embedding'] = make_embedding_log(
                checkpoint=checkpoint,
                output_path=str(rag_path),
            )
        else:
            # 2. Load existing embeddings for sync if not already in memory
            if "rag_embedding" not in checkpoint:
                try:
                    with open(rag_path, 'r') as f:
                        checkpoint["rag_embedding"] = json.load(f)
                except:
                    checkpoint["rag_embedding"] = []

        # 3. ALWAYS SYNC TO MONGODB IF CHAT_ID IS PROVIDED
        # This ensures that even if the video was already processed, the state is updated in DB
        if storage_manager.is_remote:
            storage_manager.save_final_results(
                checkpoint=checkpoint, 
                rag_embedding=checkpoint.get("rag_embedding")
            )

        # Final logs and checkpoint sync
        logpath = save_log(data=checkpoint, path=f"logs/{output_dir}.json")
        storage_manager.save_checkpoint(checkpoint=checkpoint)


if __name__ == '__main__':
    main()
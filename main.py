from dotenv import load_dotenv
load_dotenv()

from src.debug_utils import *
from src.log_utils import *
import argparse
import os
import time
from pathlib import Path

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
blip_caption_len    = 30        # max blip caption length
blip_num_beams      = 4         # beam search width (whatever that means)
blip_do_sample      = False     # sampling vs deterministic decoding
yolo_action_fps     = 4
yolo_conf_thres     = 0.8       # YOLO confidence threshold
yolo_iou_thres      = 0.5       # YOLO IoU threshold for NMS
ast_target_sr       = 16000     # audio target sample rate for AST
asr_model_size      = 'small'
asr_use_vad         = True      # enable VAD for ASR (whatever that means)
asr_target_sr       = 16000     # audio target sample rate for ASR
llm_scene_history   = 5         # number of prior scenes in LLM context
llm_chunk_len       = 50000     # max char len of combined scenes for one chunk
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
    "blip_num_beams": blip_num_beams,
    "blip_do_sample": blip_do_sample,
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

    rag = subparsers.add_parser("rag", help="Run RAG for a single video")
    rag.add_argument("--video", required=True, help="Blob name or path")

    return parser.parse_args()

VIDEOS_DIR = Path("Videos")
CATALOG_PATH = VIDEOS_DIR / "_all_videos.json"
PROCESSED_ROOT = Path("_processed")
args = parse_args()
catalog = load_video_catalog(CATALOG_PATH)
selected_paths = select_videos(args, catalog, VIDEOS_DIR)

if not selected_paths:
    raise SystemExit("No videos selected.")
if args.command == "rag" and len(selected_paths) != 1:
    raise SystemExit("RAG supports exactly one video. Use --video to pick one.")

test_videos = {make_output_dir(p, PROCESSED_ROOT): str(p) for p in selected_paths}
rag_only = args.command == "rag"

for output_dir, test_video in test_videos.items():
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    checkpoint_path = f"{output_dir}/checkpoint.json"
    checkpoint = read_json(json_path=checkpoint_path) # if deleted it will return a {}
    checkpoint.setdefault("steps", {})
    step = checkpoint["steps"]

    if not checkpoint.get("scenes"):
        print("")
        print_section("Running PysceneDetect...")
        checkpoint["scenes"], step['get_scene_list'] = get_scene_list_log(
            input_video_path=test_video,
            threshold = pyscene_threshold,
            min_scene_sec= pyscene_shortest,
        )
        see_scenes_cuts(df=checkpoint["scenes"])
        time.sleep(10)

        print("")
        print(f"Saving clips in: {output_dir}/.clips")
        checkpoint["scenes"], step['save_clips'] = save_clips_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            output_dir=f"{output_dir}/.clips",
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "frame_captions" not in checkpoint["scenes"][-1].keys():
        print(f"Saving sampled frames in: {output_dir}/.frames")
        checkpoint["scenes"], step['sample_frames'] = sample_frames_log(
            input_video_path=test_video,
            scenes=checkpoint["scenes"],
            num_frames = frames_per_scene,
            new_size = frame_resolution,
            output_dir=f"{output_dir}/.frames",
        )
        time.sleep(10)

        print("")
        print_section("Running BLIP...")
        checkpoint["scenes"], step['caption_frames'] = caption_frames_log(
            scenes=checkpoint["scenes"],
            prompt= blip_start_prompt,
            max_length=blip_caption_len,
            num_beams=blip_num_beams,
            do_sample=blip_do_sample,
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
    

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
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "audio_natural" not in checkpoint["scenes"][-1].keys():
        print("")
        print_section("Running MIT AST...")
        checkpoint["scenes"], step['ast_timings'] = extract_sounds_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            target_sr=ast_target_sr,
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


    if "audio_speech" not in checkpoint["scenes"][-1].keys():
        print("")
        print_section("Running Whisper...")
        checkpoint["scenes"], step['asr_timings'] = extract_speech_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            model=asr_model_size,
            use_vad=asr_use_vad,
            target_sr=asr_target_sr,
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "llm_scene_description" not in checkpoint["scenes"][-1].keys():
        print("")
        print_section("Running GPT4o Scene Descriptions...")
        checkpoint["scenes"], step['describe_scenes'] = describe_scenes_log(
            scenes=checkpoint["scenes"],
            client=client,
            hist_size= llm_scene_history,
            YOLO_key="yolo_detections",
            FLIP_key="frame_captions",
            ASR_key="audio_natural",
            AST_key="audio_speech",
            SUMMARY_key="llm_scene_description",
            model=model_name,
            prompt_path="prompts/describe_scene.txt",
            cooldown_sec=llm_cooldown_sec,
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
    #todo: if GPT4o responsibleAI error gets triggered, change prompt?

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
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
    #todo: if narrative is < final summary_len then go directly to synopsis
    if "synopsis" not in checkpoint:
        print("")
        print_section("Running GPT4o Synopsis generation...")
        checkpoint, step['synthesize_synopsis'] = synthesize_synopsis_log(
            client=client,
            deployment=deployment,
            data=checkpoint,
            debug=True,
            output_dir=output_dir,
        )

    rag_path = f"{output_dir}/rag_embedding.json"
    if not os.path.exists(rag_path):
        checkpoint["rag_embedding"], step['make_embedding'] = make_embedding_log(
            checkpoint=checkpoint,
            output_path=rag_path,
        )
        
        cleared_checkpoint = save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
        log = complete_log(
            log=log,
            steps=step,
            vid_len=checkpoint["scenes"][-1]["end_timecode"],
            scene_num=len(checkpoint["scenes"]),
            vid_df=cleared_checkpoint,
        )
        
        logpath = save_log(data=log, path=f"logs/{output_dir}.json")
        save_checkpoint(checkpoint=log, path=checkpoint_path)

    # todo: the RAG should be able to answer questions like "how long is the video"

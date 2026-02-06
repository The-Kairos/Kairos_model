from src.debug_utils import *
from src.log_utils import *
import time
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

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

# todo: in the future, i want to have different params ready for videos with motions, or params to run things for faster but assume no motion 

# =========================================================
# RAG IS ONLY ACTIVE WHEN THERES ONE VIDEO IN TESTVIDS
# =========================================================
run_folder = "./.runs"
test_videos = {
    f"{run_folder}/pasta": r"Videos\How to Make Pasta - Without a Machine.mp4",
    f"{run_folder}/messi": r"Videos\Argentina v France Full Penalty Shoot-out.mp4",
    f"{run_folder}/malala_long": r"Videos\.Malala Yousafzai FULL Nobel Peace Prize Lecture 2014.mp4",
    f"{run_folder}/web_summit": r"Videos\.Web Summit Qatar 2026 Day Three.mp4",
    # f"{run_folder}/grad_honors": r"Videos\.UDST honors graduation.mp4",
    # f"{run_folder}/sheldon2": r"Videos\Young Sheldon_ First Day of High School (Season 1 Episode 1 Clip) _ TBS.mp4",
}

for OUTPUT_DIR, test_video in test_videos.items():
    log = initiate_log(video_path=test_video, run_description="Test run for video processing pipeline.")
    output_dir = Path(OUTPUT_DIR)

    # I added checkpoints so if you wanna redo the whole process,
    # youd have to delete the checkpoint json in the path below
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = read_json(checkpoint_path) # if deleted it will return a {}
    checkpoint.setdefault("steps", {})
    step = checkpoint["steps"]

    if not checkpoint.get("scenes"):
        checkpoint["scenes"], step['get_scene_list'] = get_scene_list_log(test_video, min_scene_sec=2)
        see_scenes_cuts(checkpoint["scenes"])
        time.sleep(10)

        checkpoint["scenes"], step['save_clips'] = save_clips_log(
            test_video,
            checkpoint["scenes"],
            output_dir=output_dir / "clips",
        )
        save_checkpoint(checkpoint, checkpoint_path)

    if "frame_captions" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['sample_frames'] = sample_frames_log(
            input_video_path=test_video,
            scenes=checkpoint["scenes"],
            num_frames=3,
            new_size=320,
            output_dir=output_dir / "frames",
        )
        time.sleep(10)

        checkpoint["scenes"], step['caption_frames'] = caption_frames_log(
            scenes=checkpoint["scenes"],
            max_length=30,
            num_beams=4,
            do_sample=False,
            debug=True,
            prompt="a video frame of"
        )
        time.sleep(10)
        save_checkpoint(checkpoint, checkpoint_path)
    

    if "yolo_detections" not in checkpoint["scenes"][-1].keys():
        
        if "frames" not in checkpoint["scenes"][-1].keys():
            checkpoint["scenes"], step['sample_frames'] = sample_frames_log(
            input_video_path=test_video,
            scenes=checkpoint["scenes"],
            num_frames=3,
            new_size=320,
            output_dir=output_dir / "frames",
        )
        time.sleep(10)

        checkpoint["scenes"], step['detect_object_yolo'] = detect_object_yolo_log(
            scenes=checkpoint["scenes"],
            model_size="model/yolov8s.pt",
            conf=0.5,
            iou=0.45,
            output_dir=output_dir / "yolo",
        )
        time.sleep(10)
        save_checkpoint(checkpoint, checkpoint_path)

    if "audio_natural" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['ast_timings'] = extract_sounds_log(
            test_video,
            scenes=checkpoint["scenes"],
            debug=True
        )
        time.sleep(10)
        save_checkpoint(checkpoint, checkpoint_path)


    if "audio_speech" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['asr_timings'] = extract_speech_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            model="small",
            use_vad=True,
            target_sr=16000,
            debug=True
        )
        time.sleep(10)
        save_checkpoint(checkpoint, checkpoint_path)

    if "llm_scene_description" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['describe_scenes'] = describe_scenes_log(
            scenes=checkpoint["scenes"],
            client=client,
            hist_size=3,
            YOLO_key="yolo_detections",
            FLIP_key="frame_captions",
            ASR_key="audio_natural",
            AST_key="audio_speech",
            SUMMARY_key="llm_scene_description",
            debug=True,
            prompt_path="prompts/flash_scene_prompt_joy.txt",
            model=model_name,
        )
        time.sleep(10)
        save_checkpoint(checkpoint, checkpoint_path)

    if "narratives" not in checkpoint:
        checkpoint, step['summarize_scenes'] = summarize_scenes_log(
            client, deployment, checkpoint["scenes"], debug=True, output_dir=output_dir
        )
        save_checkpoint(checkpoint, checkpoint_path)

    if "synopsis" not in checkpoint:
        checkpoint, step['synthesize_synopsis'] = synthesize_synopsis_log(
            client, deployment, checkpoint, debug=True, output_dir=output_dir
        )

    rag_path = output_dir / "rag_embedding.json"
    if not rag_path.exists():
        checkpoint["rag_embedding"] , step['make_embedding'] = make_embedding_log(checkpoint, rag_path)
        
        cleared_checkpoint = save_checkpoint(checkpoint, checkpoint_path)
        log = complete_log(log, step, vid_len=checkpoint["scenes"][-1]["end_timecode"], scene_num=len(checkpoint["scenes"]), vid_df= cleared_checkpoint)
        
        logpath = save_log(log, path=Path("logs") / f"{output_dir}.json")
        save_checkpoint(log, checkpoint_path)

    # RAG IS ONLY ACTIVE WHEN THERES ONE VIDEO IN TESTVIDS
    if len(test_videos) == 1 and rag_path.exists():
        ask_rag(rag_path, k=10, show_k_context=True, 
                conv_path=output_dir / "conversation_history.json",
                log_source = checkpoint_path )
        
    # todo: integrate the RAG to have the synopsis and narratives as "long summary? the "key" is summary  
    # todo: the RAG should be able to answer questions like "how long is the video"

from src.debug_utils import *
from src.log_utils import *
import time
import os
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

# =========================================================
improve_motion_detection = True
pyscene_threshold   = 27        # sensitivity (lower = more cuts)
pyscene_shortest    = 2         # minimum scene length  
frames_per_scene    = 3         # number of frames sampled in each scene
frame_resolution    = 320       # resolution on the longest axis
blip_start_prompt   = "a video frame of"
blip_caption_len    = 30        # max blip caption length
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
llm_chunk_len       = 7000      # max char len of combined scenes for one chunk
llm_summary_len     = 20000     # max char len of final context for synopsis
rag_top_k_context   = 10        # top-k RAG scenes to include
# =========================================================
improve_motion_detection    = False
prioritize_speed            = False

if improve_motion_detection:
    pyscene_threshold   = 40     # sensitivity of scene detected
    pyscene_shortest    = 0.5    # the minimum scene length  
    yolo_action_fps     = 5
if prioritize_speed:
    frames_per_scene    = 1      # number of frames sampled in each scene
    llm_scene_history   = 3      # number of prior scenes in LLM context
    llm_chunk_len       = 128000 # max char len of combined scenes for one chunk 
    llm_summary_len     = 128000 # max char len of final context for synopsis
    rag_top_k_context   = 5      # top-k RAG scenes to include
# =========================================================

params = {
    "improve_motion_detection": improve_motion_detection,
    "prioritize_speed": prioritize_speed,
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
    "rag_top_k_context": rag_top_k_context,
}

# =========================================================
# RAG IS ONLY ACTIVE WHEN THERES ONE VIDEO IN TESTVIDS
# =========================================================
run_folder = "./.action"
test_videos = {
    f"{run_folder}/messi": r"Videos\Argentina v France Full Penalty Shoot-out.mp4",
    f"{run_folder}/pasta": r"Videos\How to Make Pasta - Without a Machine.mp4",
    f"{run_folder}/malala_long": r"Videos\.Malala Yousafzai FULL Nobel Peace Prize Lecture 2014.mp4",
    f"{run_folder}/sheldon": r"Videos\Young Sheldon_ First Day of High School (Season 1 Episode 1 Clip) _ TBS.mp4",
    f"{run_folder}/titanic": r"Videos\.Titanic.1997.NaijaPrey.com.mkv",
    # f"{run_folder}/grad_honors": r"Videos\.UDST honors graduation.mp4",
    # f"{run_folder}/web_summit": r"Videos\.Web Summit Qatar 2026 Day Three.mp4",
}

for output_dir, test_video in test_videos.items():
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
        checkpoint["scenes"], step['get_scene_list'] = get_scene_list_log(
            input_video_path=test_video,
            threshold = pyscene_threshold,
            min_scene_sec= pyscene_shortest,
        )
        see_scenes_cuts(df=checkpoint["scenes"])
        time.sleep(10)

        checkpoint["scenes"], step['save_clips'] = save_clips_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            output_dir=f"{output_dir}/clips",
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "frame_captions" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['sample_frames'] = sample_frames_log(
            input_video_path=test_video,
            scenes=checkpoint["scenes"],
            num_frames = frames_per_scene,
            new_size = frame_resolution,
            output_dir=f"{output_dir}/fps",
        )
        time.sleep(10)

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
            checkpoint["scenes"], step['sample_fps'] = sample_fps_log(
                input_video_path=test_video,
                scenes=checkpoint["scenes"],
                fps=yolo_action_fps,
                new_size=frame_resolution,
                output_dir=f"{output_dir}/frames",
                frames_key="yolo_frames",
                frame_paths_key="yolo_frame_paths",
            )
        time.sleep(10)

        checkpoint["scenes"], step['detect_object_yolo'] = detect_object_yolo_log(
            scenes=checkpoint["scenes"],
            model_size="model/yolov8s.pt",
            conf=yolo_conf_thres,
            iou=yolo_iou_thres,
            output_dir=f"{output_dir}/yolo",
            frame_key="yolo_frames",
            summary_key="yolo_detections",
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "audio_natural" not in checkpoint["scenes"][-1].keys():
        checkpoint["scenes"], step['ast_timings'] = extract_sounds_log(
            video_path=test_video,
            scenes=checkpoint["scenes"],
            target_sr=ast_target_sr,
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


    if "audio_speech" not in checkpoint["scenes"][-1].keys():
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
            prompt_path="prompts/flash_scene_prompt_joy.txt",
            debug=True,
        )
        time.sleep(10)
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "narratives" not in checkpoint:
        checkpoint, step['summarize_scenes'] = summarize_scenes_log(
            client=client,
            deployment=deployment,
            scenes=checkpoint["scenes"],
            chunk_size = llm_chunk_len,
            summary_len = llm_summary_len,
            debug=True,
            output_dir=output_dir,
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "synopsis" not in checkpoint:
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

    # RAG IS ONLY ACTIVE WHEN THERES ONE VIDEO IN TESTVIDS
    if len(test_videos) == 1 and os.path.exists(rag_path):
        ask_rag(
            rag_path=rag_path,
            show_k_context=True,
            k=rag_top_k_context,
            conv_path=f"{output_dir}/conversation_history.json",
            log_source=checkpoint_path,
            show_timings=False,
        )
        
    # todo: integrate the RAG to have the synopsis and narratives as "long summary? the "key" is summary  
    # todo: the RAG should be able to answer questions like "how long is the video"

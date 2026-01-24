from src.debug_utils import *
from src.log_utils import *
import time
from dotenv import load_dotenv
load_dotenv()

use_gemini = True

if use_gemini:
    # =============== GEMINI FLASH 2.5 ===============
    api_key = os.getenv("GEMINI_API_KEY")

    from google import genai
    model_name= "gemini-2.5-flash"
    client = genai.Client(vertexai=True, api_key=api_key) # vertexai=True is needed if youre Dr. Oussama's key
else:
    # =============== GPT 4o ===============
    endpoint = "https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
    model_name = "gpt-4o"
    deployment = os.getenv("GPT_DEPLOYMENT")

    subscription_key = os.getenv("SUPSCRIPTION_KEY")
    api_version = os.getenv("API_VERSION")

    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

test_videos = {
    f"car_hist3_{model_name}": r"Videos\.Cartastrophe.mp4",
    # f"malala_hist3_{model_name}": r"Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4",
    f"CCIT_hist3_{model_name}": r"Videos\.UDST CCIT graduation 30 mins.mp4",
    f"sponge_hist3_{model_name}": r"Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4",
    f"spain_hist3_{model_name}": r"Videos\.Spain Vlog.mp4",
}
OUTPUT_DIR = "spain"

for OUTPUT_DIR, test_video in test_videos.items():
    log = initiate_log(video_path=test_video, run_description="Test run for video processing pipeline.")
    step = {}

    scenes, step['get_scene_list'] = get_scene_list_log(test_video, min_scene_sec=2) 
    time.sleep(10)

    scenes, step['save_clips'] = save_clips_log(test_video, scenes, output_dir=f"./{OUTPUT_DIR}/clips")
    time.sleep(10)
    
    see_scenes_cuts(scenes)

    scenes_with_frames, step['sample_frames'] = sample_frames_log(
        input_video_path=test_video,
        scenes=scenes,
        num_frames=3,
        new_size = 320,
        output_dir=f"./{OUTPUT_DIR}/frames",
    )
    time.sleep(10)

    captioned_scenes, step['caption_frames'] = caption_frames_log(
        scenes=scenes_with_frames,
        max_length=30,
        num_beams=4,
        do_sample=False,
        debug=True,
        prompt="a video frame of"
    )
    time.sleep(10)

    detected_obj_scenes, step['detect_object_yolo'] = detect_object_yolo_log(
        scenes= captioned_scenes,
        model_size = "model/yolov8s",
        conf = 0.5,
        iou = 0.45,
        output_dir=f"./{OUTPUT_DIR}/yolo",
    )
    time.sleep(10)

    sound_audio, step['ast_timings'] = extract_sounds_log(
            test_video,
            scenes=detected_obj_scenes,
            debug=True
    )
    time.sleep(10)

    speech_audio, step['asr_timings'] = extract_speech_log(
            video_path = test_video, 
            scenes = sound_audio, 
            model="small",
            use_vad=True, 
            target_sr=16000,
            debug = True
        )
    time.sleep(10)

    described_scenes, step['describe_scenes'] = describe_scenes_log(
        scenes= speech_audio,
        client= client,
        hist_size = 3,
        YOLO_key="yolo_detections",
        FLIP_key="frame_captions",
        ASR_key= "audio_natural",
        AST_key= "audio_speech",
        SUMMARY_key = "llm_scene_description",
        debug= True,
        prompt_path= "prompts/flash_scene_prompt_manahil.txt",
        model= model_name,
    )
    time.sleep(10)

    save_safe_df = save_vid_df(described_scenes, f"{OUTPUT_DIR}/captioned_scenes.json")
    log = complete_log(log, step, vid_len=scenes[-1]["end_seconds"], scene_num=len(scenes), vid_df= save_safe_df)
    save_log(log, filename=OUTPUT_DIR)
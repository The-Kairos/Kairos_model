import json

def format_embedding_text(scenes: list):
    embedding_texts = []
    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        end_timecode = scene.get("end_timecode")

        audio_speech = scene.get("audio_speech")
        audio_natural = scene.get("audio_natural")
        llm_scene_description = scene.get("llm_scene_description")
        
        yolo_objects = scene.get("yolo_detections", {})
        objects = ', '.join({obj.get('label') for yolo_scene in yolo_objects.values() for obj in yolo_scene})

        embedding_texts.append(
            f"From {start_timecode} to {end_timecode}, {llm_scene_description}. "
            f"Visible objects include {objects}. "
            f"Background audio: {audio_natural}. "
            f"Spoken dialogue: {audio_speech}."
        )
        
    return embedding_texts

log_path= r"logs\car_pyscene_blip_yolo_ASR_AST_GeminiPro25_20251123_180938.json"
with open(log_path, "r", encoding="utf-8") as f:
    logs = json.load(f)
scenes = format_embedding_text(logs.get("scenes"))

print(scenes)
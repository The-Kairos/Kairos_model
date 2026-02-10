import json
from typing import List, Dict

def load_scenes(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    out = []

    for i, s in enumerate(scenes):
        # CHANGED: Now preserves all scene data, not just description
        scene_data = s.copy()
        # Ensure scene_index exists
        if "scene_index" not in scene_data:
            scene_data["scene_index"] = i
        # Ensure description exists (for backward compatibility)
        if "description" not in scene_data:
            scene_data["description"] = scene_data.get("llm_scene_description", "").strip()
        out.append(scene_data)

    return out

# Joy's scene description formatter function:
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

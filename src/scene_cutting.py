# src/scene_cutting.py

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def get_scene_list(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    base_timecode = video_manager.get_base_timecode()

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
    finally:
        video_manager.release()

    # Convert to list of dicts
    return [{"start": s[0].get_frames(), "end": s[1].get_frames(), "frames": []} for s in scene_list]

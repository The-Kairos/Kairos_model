from typing import List, Dict

from src.kairos.model.pyscenedetect import detect_scenes

def get_scene_list(
    input_video_path: str,
    threshold: float = 27,
    min_scene_sec: int = 2,
    frame_skip: int = 3,
    retry_threshold_factor: float = 0.5,
    fallback_interval_sec: int = 20,
) -> List[Dict]:
    return detect_scenes(
        input_video_path=input_video_path,
        threshold=threshold,
        min_scene_sec=min_scene_sec,
        frame_skip=frame_skip,
        retry_threshold_factor=retry_threshold_factor,
        fallback_interval_sec=fallback_interval_sec,
    )


def test():
    test_video = r'Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'
    scenes = get_scene_list(test_video)

    print(f"Found {len(scenes)} scenes.")
    for s in scenes:
        print(
            f"Scene {s['scene_index']:03d}: "
            f"{s['start_timecode']} -> {s['end_timecode']} "
            f"({s['duration_seconds']:.2f} sec)"
        )
# test()

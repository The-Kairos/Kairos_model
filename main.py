from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning import caption_frames

test_video = r'Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'
scenes = get_scene_list(test_video)

print(f"Found {len(scenes)} scenes.")
# for s in scenes:
#     print(
#         f"Scene {s['scene_index']:03d}: "
#         f"{s['start_timecode']} -> {s['end_timecode']} "
#         f"({s['duration_seconds']:.2f} sec)"
#     )

scenes_with_frames = sample_frames(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=4,
)


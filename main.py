from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames

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
    output_dir="./output/frames",
)

captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)

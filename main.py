from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.yolo_inference import run_yolo_on_frames
from src.debug_utils import see_first_scene, see_scenes_cuts
from src.perf_utils import measure_performance

test_video = r"Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4"

# ===== Wrap functions =====
get_scene_list = measure_performance(get_scene_list)
sample_frames = measure_performance(sample_frames)
run_yolo_on_frames = measure_performance(run_yolo_on_frames)

# ===== Total pipeline start =====
import time, psutil, os
process = psutil.Process(os.getpid())
mem_start = process.memory_info().rss / (1024**2)
t_start = time.time()

# 1) Scene detection
scenes = get_scene_list(test_video)
see_scenes_cuts(scenes)

# 2) Frame sampling
scenes_with_frames = sample_frames(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=2,
    output_dir="./output/frames",
)

# 3) YOLO inference
yolo_results = run_yolo_on_frames(
    scenes=scenes_with_frames,
    model_size="yolov8s",
    conf=0.25,
    iou=0.45,
)

# 4) Print first scene info
see_first_scene(yolo_results)

# ===== Total pipeline performance =====
t_end = time.time()
mem_end = process.memory_info().rss / (1024**2)
print(f"\n[PIPELINE PERF] Total time: {t_end - t_start:.2f}s, "
      f"Total memory delta: {mem_end - mem_start:.2f} MB")

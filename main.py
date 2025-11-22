import os
import time
import json
import argparse

import psutil
import torch

from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.captioning.frame_captioning_blip import BLIPCaptioner
from src.detection.yolo_detector import YOLODetector
from src.captioning.llm_fusion_caption import LLMSegmentCaptioner


# ------------------------
# Helpers
# ------------------------
process = psutil.Process(os.getpid())


def mem_mb():
    return process.memory_info().rss / 1024 / 1024


# ------------------------
# CLI args
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Kairos video captioning pipeline")

    parser.add_argument(
        "--video",
        type=str,
        default="Videos/SPAIN.mp4",
        help="Path to input video",
    )

    parser.add_argument(
        "--blip",
        action="store_true",
        help="Enable BLIP captioning",
    )

    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLO object detection",
    )

    return parser.parse_args()


args = parse_args()

VIDEO = args.video
device = "cuda" if torch.cuda.is_available() else "cpu"

# Decide what to run:
# - if no flags: run BOTH
# - if any flag(s): use exactly what was specified
if not args.blip and not args.yolo:
    USE_BLIP = True
    USE_YOLO = True
else:
    USE_BLIP = args.blip
    USE_YOLO = args.yolo

if not USE_BLIP and not USE_YOLO:
    raise ValueError("Both BLIP and YOLO are disabled. Enable at least one of them.")


print("[CONFIG] Video:", VIDEO)
print("[CONFIG] Device:", device)
print(f"[CONFIG] USE_BLIP={USE_BLIP}, USE_YOLO={USE_YOLO}")


# ------------------------
# Load models
# ------------------------
if USE_BLIP:
    print("[INFO] Loading BLIP model...")
    blip_captioner = BLIPCaptioner(device=device)
else:
    blip_captioner = None
    print("[INFO] BLIP disabled")

if USE_YOLO:
    print("[INFO] Loading YOLOv8 model...")
    # change this value to whatever you want (e.g. 0.3, 0.5, 0.7)
    yolo = YOLODetector("yolov8s.pt", conf_threshold=0.5)

else:
    yolo = None
    print("[INFO] YOLO disabled")

print("[INFO] Loading Gemini LLM Segment Captioner...")
llm_captioner = LLMSegmentCaptioner(model="gemini-2.5-flash")


# ------------------------
# Pipeline
# ------------------------
pipeline_start = time.time()
pipeline_mem_start = mem_mb()

# Scene detection
s_t0 = time.time()
scenes = get_scene_list(VIDEO)
s_time = time.time() - s_t0
print(f"[INFO] Detected {len(scenes)} scenes in {s_time:.2f}s")

# Frame sampling
f_t0 = time.time()
# Set output_dir=None if you don't want to write frame images during perf tests
scenes = sample_frames(VIDEO, scenes, num_frames=2, output_dir="output/frames")
f_time = time.time() - f_t0
print(f"[INFO] Sampled frames in {f_time:.2f}s")

results = []
YOLO_total = 0.0
BLIP_total = 0.0
SEG_total = 0.0

for sc in scenes:
    frames = sc["frames"]
    scene_idx = sc["scene_index"]

    # ---- per-scene metrics (before) ----
    scene_t0 = time.time()
    mem_before = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    scene_yolo_time = 0.0
    scene_blip_time = 0.0

    frame_fusions = []

    # ------------------------
    # Per-frame processing
    # ------------------------
    for fr in frames:
        dets = []
        caption = None

        # YOLO (optional)
        if USE_YOLO:
            dets, t_y = yolo.detect(fr)
            scene_yolo_time += t_y
            YOLO_total += t_y

        # BLIP (optional)
        if USE_BLIP:
            caption, t_b = blip_captioner.caption(fr, prompt="a video frame of")
            scene_blip_time += t_b
            BLIP_total += t_b

        frame_fusions.append(
            {
                "yolo_detections": dets,
                "blip_caption": caption,
            }
        )

    # ------------------------
    # Build fusion_texts for the LLM
    # ------------------------
    fusion_texts = []
    for fo in frame_fusions:
        parts = []

        if USE_YOLO:
            if fo["yolo_detections"]:
                objs = ", ".join(
                    [
                        f"{d['label']} ({d['confidence']:.2f})"
                        for d in fo["yolo_detections"]
                    ]
                )
            else:
                objs = "no objects"
            parts.append(f"Objects: {objs}")

        if USE_BLIP:
            parts.append(f"Caption: {fo['blip_caption']}")

        if not parts:
            parts.append("No BLIP or YOLO data available for this frame.")

        fusion_texts.append("\n".join(parts))

    # ------------------------
    # LLM segment caption
    # ------------------------
    t0_llm = time.time()
    seg_cap = llm_captioner.describe_segment(fusion_texts)
    tseg = time.time() - t0_llm
    SEG_total += tseg

    # ---- per-scene metrics (after) ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated()
    else:
        peak_vram = 0

    mem_after = process.memory_info().rss
    scene_time = time.time() - scene_t0

    scene_metrics = {
        "scene_time_s": scene_time,
        "scene_ram_delta_mb": (mem_after - mem_before) / 1024 / 1024,
        "scene_peak_vram_mb": peak_vram / 1024 / 1024,
        "yolo_time_s": scene_yolo_time if USE_YOLO else 0.0,
        "blip_time_s": scene_blip_time if USE_BLIP else 0.0,
        "llm_time_s": tseg,
    }

    scene_res = {
        "scene_index": scene_idx,
        "frames": frame_fusions,
        "segment_caption": {"text": seg_cap},
        "metrics": scene_metrics,
    }
    results.append(scene_res)

# ------------------------
# Final metrics
# ------------------------
pipeline_time = time.time() - pipeline_start
pipeline_mem_end = mem_mb()

print("\n[SUMMARY]")
print(f"Scenes: {len(scenes)}")
print(f"Scene detection: {s_time:.2f}s")
print(f"Sampling: {f_time:.2f}s")
if USE_YOLO:
    print(f"YOLO total: {YOLO_total:.2f}s")
else:
    print("YOLO total: (disabled)")
if USE_BLIP:
    print(f"BLIP total: {BLIP_total:.2f}s")
else:
    print("BLIP total: (disabled)")
print(f"LLM total: {SEG_total:.2f}s")
print(f"Pipeline time: {pipeline_time:.2f}s")
print(f"Memory delta: {pipeline_mem_end - pipeline_mem_start:.2f} MB")

print("\n[SEGMENT CAPTIONS]")
for sc in results:
    print(f"Scene {sc['scene_index']}: {sc['segment_caption']['text']}\n---")

os.makedirs("output/llm_results", exist_ok=True)
with open("output/llm_results/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Results saved to output/llm_results/results.json")

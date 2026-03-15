"""Semantic scene splitting using ViT embeddings (standalone); also exports shared pipeline settings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from pyscene_utils import (
    DEFAULT_VIDEO,
    ensure_hf_offline_if_unreachable,
    finalize_scene_times,
    hf_local_only,
    sample_video_frames,
    save_scene_boundary_frames,
    to_relative,
)
from semantic_utils import bounds_to_times, compute_scene_bounds, normalize_embeddings, pool_tokens

# =========================================================
# Sensitivity settings from main.py
improve_motion_detection = True
pyscene_threshold = 27
pyscene_shortest = 2
frames_per_scene = 3
frame_resolution = 320
blip_start_prompt = "a video frame of"
blip_caption_len = 50
blip_min_length = 15
blip_num_beams = 1
blip_do_sample = True
blip_top_p = 0.85
blip_temperature = 0.65
blip_length_penalty = 1.0
blip_no_repeat_ngram_size = 3
blip_repetition_penalty = 1.1
yolo_action_fps = 4
yolo_conf_thres = 0.8
yolo_iou_thres = 0.5
ast_target_sr = 16000
asr_model_size = "small"
asr_use_vad = True
asr_target_sr = 16000
llm_scene_history = 5
llm_chunk_len = 50000
llm_summary_len = 50000
llm_cooldown_sec = 0
rag_top_k_context = 10
# =========================================================
improve_motion_detection = False
prioritize_speed = False
process_static_videos = False

if improve_motion_detection:
    pyscene_threshold = 15
    pyscene_shortest = 0.5
    frames_per_scene = 5
    yolo_action_fps = 8
if prioritize_speed:
    pyscene_threshold = 40
    frames_per_scene = 1
    llm_chunk_len = 500000
    llm_summary_len = 500000
if process_static_videos:
    pyscene_threshold = 3
    frames_per_scene = 1
    yolo_action_fps = 0.5
# =========================================================

VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_SCORE_THRESHOLD = 0.65
DEFAULT_BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _batched(items: List, batch_size: int):
    """Yield successive *batch_size* chunks from *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _encode_vit(images: List, model: object, processor: object, device: str, batch_size: int) -> List:
    """Compute normalised ViT embeddings for a list of PIL images."""
    import torch

    embeddings: List[torch.Tensor] = []
    for batch in _batched(images, batch_size):
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = pool_tokens(outputs.last_hidden_state)
        pooled = normalize_embeddings(pooled)
        embeddings.extend(pooled.cpu())
    return embeddings


def run(
    video_path: str | Path = DEFAULT_VIDEO,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_id: str = VIT_MODEL_ID,
) -> dict:
    """Run ViT-based semantic scene splitting on a video and return results."""
    import torch
    from transformers import ViTImageProcessor, ViTModel

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ensure_hf_offline_if_unreachable()
    local_only = hf_local_only()

    start = time.perf_counter()
    frames, timestamps, info = sample_video_frames(
        video_path=video_path,
        sample_fps=sample_fps,
        new_size=frame_resolution,
    )

    if not frames:
        raise RuntimeError("No frames sampled from video.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained(
        model_id, local_files_only=local_only
    )
    model = ViTModel.from_pretrained(model_id, local_files_only=local_only).to(device)
    model.eval()

    embeddings = _encode_vit(frames, model, processor, device, batch_size)
    bounds = compute_scene_bounds(embeddings, score_threshold)
    times = bounds_to_times(bounds, timestamps, info["duration"])

    scenes = [
        {"start_seconds": start_s, "end_seconds": end_s}
        for start_s, end_s in times
    ]
    scenes = finalize_scene_times(scenes)
    scenes = save_scene_boundary_frames(
        video_path=video_path,
        scenes=scenes,
        test_name=Path(__file__).stem,
        new_size=frame_resolution,
    )

    elapsed = time.perf_counter() - start
    result = {
        "video": to_relative(video_path),
        "scenes": scenes,
        "elapsed_seconds": elapsed,
        "sample_fps": sample_fps,
        "score_threshold": score_threshold,
        "threshold": score_threshold,
        "embedding_model": model_id,
    }
    return result


def main() -> None:
    """CLI entry-point for ViT semantic scene splitting."""
    parser = argparse.ArgumentParser(description="Semantic scene splitting using ViT embeddings.")
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to the input video.",
    )
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-id", default=VIT_MODEL_ID)
    args = parser.parse_args()

    result = run(
        video_path=args.video,
        sample_fps=args.sample_fps,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        model_id=args.model_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

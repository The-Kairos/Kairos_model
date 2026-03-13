from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

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
from vit_scene import frame_resolution

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _batched(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _load_blip_vision(
    model_id: str,
    device: str,
    local_only: bool,
) -> Tuple[object, object, bool]:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_id, local_files_only=local_only)
    full_model = BlipForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local_only
    ).to(device)
    full_model.eval()
    return full_model.vision_model, processor, True


def _encode_blip(images, vision_model, processor, device, batch_size: int):
    import torch

    embeddings: List[torch.Tensor] = []
    for batch in _batched(images, batch_size):
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = vision_model(pixel_values=pixel_values)
        pooled = pool_tokens(outputs.last_hidden_state)
        pooled = normalize_embeddings(pooled)
        embeddings.extend(pooled.cpu())
    return embeddings


def run(
    video_path: str | Path = DEFAULT_VIDEO,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_id: str = BLIP_MODEL_ID,
) -> dict:
    import torch

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
    vision_model, processor, decoder_removed = _load_blip_vision(
        model_id, device, local_only
    )

    embeddings = _encode_blip(frames, vision_model, processor, device, batch_size)
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
        "decoder_removed": decoder_removed,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic scene splitting using BLIP vision embeddings.")
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to the input video.",
    )
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-id", default=BLIP_MODEL_ID)
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

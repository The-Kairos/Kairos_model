"""Hybrid scene splitting: PySceneDetect initial cuts merged with CLIP embeddings."""

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
    load_frames_at_times,
    save_scene_boundary_frames,
    to_relative,
)
from semantic_utils import cosine_similarity, normalize_embeddings
from vit_scene import frame_resolution, pyscene_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
HYBRID_PYSCENE_THRESHOLD = max(5.0, pyscene_threshold * 0.5)
HYBRID_MIN_SCENE_SEC = 0.5
HYBRID_FRAME_SKIP = 1
DEFAULT_MERGE_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 16


def _batched(items: List, batch_size: int):
    """Yield successive *batch_size* chunks from *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _encode_clip(images: List, model: object, processor: object, device: str, batch_size: int) -> List:
    """Compute normalised CLIP image embeddings for a list of PIL images."""
    import torch

    embeddings: List[torch.Tensor] = []
    for batch in _batched(images, batch_size):
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values)
        feats = normalize_embeddings(feats)
        embeddings.extend(feats.cpu())
    return embeddings


def _merge_scenes(scenes: List[dict], embeddings: List, threshold: float) -> List[dict]:
    """Merge consecutive scenes whose embeddings exceed *threshold* similarity."""
    if not scenes:
        return []

    merged: List[dict] = []
    current = dict(scenes[0])
    current_emb = embeddings[0]

    for scene, emb in zip(scenes[1:], embeddings[1:]):
        sim = cosine_similarity(current_emb, emb)
        if sim >= threshold:
            current["end_seconds"] = scene["end_seconds"]
            combined = (current_emb + emb) / 2.0
            current_emb = normalize_embeddings(combined.unsqueeze(0)).squeeze(0)
        else:
            merged.append(current)
            current = dict(scene)
            current_emb = emb

    merged.append(current)
    return merged


def run(
    video_path: str | Path = DEFAULT_VIDEO,
    pyscene_threshold: float = HYBRID_PYSCENE_THRESHOLD,
    min_scene_sec: float = HYBRID_MIN_SCENE_SEC,
    frame_skip: int = HYBRID_FRAME_SKIP,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_id: str = CLIP_MODEL_ID,
) -> dict:
    """Run hybrid PySceneDetect + CLIP merge and return scene results."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    from src.scene_cutting import get_scene_list

    ensure_hf_offline_if_unreachable()
    local_only = hf_local_only()

    start = time.perf_counter()
    base_scenes = get_scene_list(
        input_video_path=str(video_path),
        threshold=pyscene_threshold,
        min_scene_sec=min_scene_sec,
        frame_skip=frame_skip,
    )

    scenes = [
        {"start_seconds": s["start_seconds"], "end_seconds": s["end_seconds"]}
        for s in base_scenes
    ]

    if not scenes:
        raise RuntimeError("No scenes detected by PySceneDetect.")

    mid_times = [(s["start_seconds"] + s["end_seconds"]) / 2 for s in scenes]
    frames = load_frames_at_times(
        video_path=video_path,
        times=mid_times,
        new_size=frame_resolution,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(
        model_id, local_files_only=local_only
    )
    model = CLIPModel.from_pretrained(
        model_id, local_files_only=local_only
    ).to(device)
    model.eval()

    embeddings = _encode_clip(frames, model, processor, device, batch_size)
    merged = _merge_scenes(scenes, embeddings, merge_threshold)

    merged = finalize_scene_times(merged)
    merged = save_scene_boundary_frames(
        video_path=video_path,
        scenes=merged,
        test_name=Path(__file__).stem,
        new_size=frame_resolution,
    )

    elapsed = time.perf_counter() - start
    result = {
        "video": to_relative(video_path),
        "scenes": merged,
        "elapsed_seconds": elapsed,
        "initial_scene_count": len(scenes),
        "pyscene_threshold": pyscene_threshold,
        "min_scene_sec": min_scene_sec,
        "merge_threshold": merge_threshold,
        "threshold": merge_threshold,
        "embedding_model": model_id,
    }
    return result


def main() -> None:
    """CLI entry-point for hybrid PySceneDetect + CLIP merge."""
    parser = argparse.ArgumentParser(
        description="Hybrid PySceneDetect + CLIP semantic merge."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to the input video.",
    )
    parser.add_argument("--pyscene-threshold", type=float, default=HYBRID_PYSCENE_THRESHOLD)
    parser.add_argument("--min-scene-sec", type=float, default=HYBRID_MIN_SCENE_SEC)
    parser.add_argument("--frame-skip", type=int, default=HYBRID_FRAME_SKIP)
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-id", default=CLIP_MODEL_ID)
    args = parser.parse_args()

    result = run(
        video_path=args.video,
        pyscene_threshold=args.pyscene_threshold,
        min_scene_sec=args.min_scene_sec,
        frame_skip=args.frame_skip,
        merge_threshold=args.merge_threshold,
        batch_size=args.batch_size,
        model_id=args.model_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

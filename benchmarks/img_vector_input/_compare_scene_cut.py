"""Compare BLIP and CLIP+CLIPCap pipelines for scene-cut detection and captioning."""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    GPT2Tokenizer,
)

from blip_to_blip import decode_caption as blip_decode_caption
from blip_to_blip import encode_image as blip_encode_image
from clip_to_clipcap import ClipCapModel
from clip_to_clipcap import decode_caption as clip_decode_caption
from clip_to_clipcap import encode_image as clip_encode_image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_image(path: Path) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def frame_number(path: Path) -> int:
    """Extract the numeric frame index from a filename like 'frame_01.jpg'."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return int(parts[-1])
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Could not parse frame number from {path.name}")


def list_frames(frames_dir: Path) -> tuple[list[Path], list[int]]:
    """Return sorted frame paths and their indices, validating continuity."""
    frames = sorted(frames_dir.glob("frame_*.jpg"), key=frame_number)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    numbers = [frame_number(p) for p in frames]
    if numbers[0] != 0:
        raise ValueError(
            f"Expected to start at frame_00.jpg, got frame_{numbers[0]:02d}.jpg"
        )
    for prev, curr in zip(numbers, numbers[1:]):
        if curr != prev + 1:
            raise ValueError(f"Missing frame between {prev} and {curr}")
    return frames, numbers


def pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Mean-pool and L2-normalize an embedding tensor to a single vector."""
    if embedding.dim() == 3:
        embedding = embedding.mean(dim=1)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.float()
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    return embedding.squeeze(0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1-D tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_scene_bounds(
    pooled_vectors: list[torch.Tensor],
    threshold: float,
) -> list[tuple[int, int]]:
    """Detect scene boundaries by thresholding cosine similarity between consecutive frames."""
    bounds: list[tuple[int, int]] = []
    start_idx = 0
    prev_vec = pooled_vectors[0]

    for idx in range(1, len(pooled_vectors)):
        sim = cosine_similarity(prev_vec, pooled_vectors[idx])
        if sim < threshold:
            bounds.append((start_idx, idx - 1))
            start_idx = idx
        prev_vec = pooled_vectors[idx]

    bounds.append((start_idx, len(pooled_vectors) - 1))
    return bounds


def render_scene_labels(bounds: list[tuple[int, int]], frame_numbers: list[int]) -> list[str]:
    """Convert scene boundary indices into human-readable frame-range labels."""
    labels = []
    for start_idx, end_idx in bounds:
        start = frame_numbers[start_idx]
        end = frame_numbers[end_idx]
        labels.append(f"frame {start}- frame {end}")
    return labels


def run_blip(
    frames: list[Path],
    frame_numbers: list[int],
    device: str,
    threshold: float,
):
    """Run scene-cut detection and captioning using the BLIP pipeline."""
    blip_id = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(blip_id).to(device)
    processor = BlipProcessor.from_pretrained(blip_id)
    model.eval()

    start = time.perf_counter()
    raw_embeddings: list[torch.Tensor] = []
    pooled_embeddings: list[torch.Tensor] = []

    for path in frames:
        image = load_image(path)
        embedding = blip_encode_image(model, processor, image, device)
        embedding = embedding.detach().cpu()
        raw_embeddings.append(embedding)
        pooled_embeddings.append(pool_embedding(embedding))

    bounds = compute_scene_bounds(pooled_embeddings, threshold)
    scenes = render_scene_labels(bounds, frame_numbers)

    captions: list[str] = []
    for start_idx, end_idx in bounds:
        mid_idx = (start_idx + end_idx) // 2
        mid_embedding = raw_embeddings[mid_idx].to(device)
        caption = blip_decode_caption(
            model,
            processor,
            mid_embedding,
            device,
            prompt="a photo of",
        )
        captions.append(caption)

    elapsed = time.perf_counter() - start
    return scenes, captions, elapsed


def run_clip(
    frames: list[Path],
    frame_numbers: list[int],
    device: str,
    threshold: float,
):
    """Run scene-cut detection and captioning using the CLIP+CLIPCap pipeline."""
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    clip_model.eval()

    repo_id = "saad1926q/clipcap-image-captioning"
    ckpt_name = "coco_prefix_best_200k.pt"
    prefix_length = 10
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ClipCapModel(prefix_length=prefix_length).to(device)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    start = time.perf_counter()
    raw_embeddings: list[torch.Tensor] = []
    pooled_embeddings: list[torch.Tensor] = []

    for path in frames:
        image = load_image(path)
        embedding = clip_encode_image(clip_model, clip_processor, image, device)
        embedding = embedding.detach().cpu()
        raw_embeddings.append(embedding)
        pooled_embeddings.append(pool_embedding(embedding))

    bounds = compute_scene_bounds(pooled_embeddings, threshold)
    scenes = render_scene_labels(bounds, frame_numbers)

    captions: list[str] = []
    for start_idx, end_idx in bounds:
        mid_idx = (start_idx + end_idx) // 2
        mid_embedding = raw_embeddings[mid_idx].to(device)
        caption = clip_decode_caption(model, tokenizer, mid_embedding, max_length=30)
        captions.append(caption)

    elapsed = time.perf_counter() - start
    return scenes, captions, elapsed


def main() -> None:
    """Load frames, run both pipelines, and print a JSON comparison of results."""
    frames_dir = BASE_DIR / "video_fps"
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1]).expanduser()
        if raw_path.is_absolute():
            frames_dir = raw_path.resolve()
        else:
            frames_dir = (Path.cwd() / raw_path).resolve()
    threshold = 0.9

    frames, frame_numbers = list_frames(frames_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blip_scenes, blip_captions, blip_time = run_blip(
        frames, frame_numbers, device, threshold
    )
    clip_scenes, clip_captions, clip_time = run_clip(
        frames, frame_numbers, device, threshold
    )

    result = {
        "BLIP": {
            "scenes": blip_scenes,
            "captions": blip_captions,
            "time": blip_time,
        },
        "CLIP": {
            "scenes": clip_scenes,
            "captions": clip_captions,
            "time": clip_time,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

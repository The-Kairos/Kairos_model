"""BLIP frame captioning with lazy model loading."""

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from kairos.utils import print_prefixed

# Lazy-loaded model singletons
_blip_model = None
_blip_processor = None


def _get_blip_model():
    """Load BLIP model and processor on first use, then cache."""
    global _blip_model, _blip_processor
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
    return _blip_model, _blip_processor


def blip_frame(
    image,
    model=None,
    processor=None,
    prompt: Optional[str] = None,
    max_length: int = 50,
    min_length: int = 20,
    num_beams: int = 1,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.8,
    length_penalty: float = 0.8,
    no_repeat_ngram_size: int = 2,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate a BLIP caption for a single frame."""
    if model is None or processor is None:
        model, processor = _get_blip_model()

    # Normalize image to RGB PIL.Image
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
    else:
        raise TypeError("image must be a PIL.Image.Image or a numpy.ndarray")

    device = next(model.parameters()).device

    if prompt is not None:
        inputs = processor(pil_image, prompt, return_tensors="pt")
    else:
        inputs = processor(pil_image, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def caption_frames(
    scenes: list[dict],
    model=None,
    processor=None,
    prompt: Optional[str] = None,
    max_length: int = 50,
    min_length: int = 20,
    num_beams: int = 1,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.8,
    length_penalty: float = 0.8,
    no_repeat_ngram_size: int = 2,
    repetition_penalty: float = 1.2,
    debug: bool = False,
) -> list[dict]:
    """Run BLIP on each frame in each scene and attach captions."""
    if model is None or processor is None:
        model, processor = _get_blip_model()

    enriched_scenes: list[dict] = []

    for scene in scenes:
        if debug:
            print_prefixed("(BLIP)", f"Scene {scene.get('scene_index', '??')}")
        frames = scene.get("frames", [])
        captions: list[str] = []

        for frame in frames:
            caption = blip_frame(
                image=frame,
                model=model,
                processor=processor,
                prompt=prompt,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
            captions.append(caption)
            if debug:
                print_prefixed("(BLIP)", f"{caption}", indent=2)

        new_scene = dict(scene)
        new_scene["frame_captions"] = captions
        enriched_scenes.append(new_scene)

    return enriched_scenes

"""BLIP frame captioning with lazy model loading."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from kairos.core.utils import print_prefixed

# Lazy-loaded model singletons
_blip_model: Any = None
_blip_processor: Any = None

# Default BLIP generation parameters — override any via **kwargs
BLIP_DEFAULTS: dict[str, Any] = {
    "max_length": 50,
    "min_length": 20,
    "num_beams": 1,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.2,
}


def _get_blip_model() -> tuple[Any, Any]:
    """Load the BLIP model and processor on first use, then cache.

    The model is loaded onto CUDA if available, otherwise CPU.  Subsequent
    calls return the cached singletons without reloading.

    Returns:
        A tuple of ``(model, processor)`` where *model* is a
        ``BlipForConditionalGeneration`` instance and *processor* is a
        ``BlipProcessor`` instance.
    """
    global _blip_model, _blip_processor
    if _blip_model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
    return _blip_model, _blip_processor


def blip_frame(
    image: Image.Image | np.ndarray,
    model: Any = None,
    processor: Any = None,
    prompt: str | None = None,
    **generate_kwargs: Any,
) -> str:
    """Generate a BLIP caption for a single frame.

    Any keyword arguments override ``BLIP_DEFAULTS`` and are forwarded
    directly to ``model.generate()``.

    Args:
        image: The input image, either a ``PIL.Image.Image`` or a NumPy
            array in BGR (OpenCV) or RGB format.
        model: A pre-loaded ``BlipForConditionalGeneration`` model.  If
            ``None``, the lazily-loaded singleton is used.
        processor: A pre-loaded ``BlipProcessor``.  If ``None``, the
            lazily-loaded singleton is used.
        prompt: An optional text prompt to condition the caption on.
        **generate_kwargs: Additional keyword arguments forwarded to
            ``model.generate()``, overriding values in ``BLIP_DEFAULTS``.

    Returns:
        The generated caption string, stripped of special tokens and
        leading/trailing whitespace.

    Raises:
        TypeError: If *image* is not a ``PIL.Image.Image`` or
            ``numpy.ndarray``.
    """
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

    kwargs = {**BLIP_DEFAULTS, **generate_kwargs}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def caption_frames(
    scenes: list[dict],
    model: Any = None,
    processor: Any = None,
    debug: bool = False,
    **blip_kwargs: Any,
) -> list[dict]:
    """Run BLIP on each frame in every scene and attach captions.

    Any keyword arguments are forwarded to ``blip_frame`` (and on to
    ``model.generate()``), overriding ``BLIP_DEFAULTS``.

    Args:
        scenes: A list of scene dictionaries.  Each scene should contain a
            ``"frames"`` key mapping to a list of images (PIL or NumPy).
        model: A pre-loaded ``BlipForConditionalGeneration`` model.  If
            ``None``, the lazily-loaded singleton is used.
        processor: A pre-loaded ``BlipProcessor``.  If ``None``, the
            lazily-loaded singleton is used.
        debug: If ``True``, print caption progress to stdout using
            :func:`kairos.core.utils.print_prefixed`.
        **blip_kwargs: Additional keyword arguments forwarded to
            ``blip_frame``.

    Returns:
        A new list of scene dictionaries, each augmented with a
        ``"frame_captions"`` key containing a list of caption strings
        (one per frame).
    """
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
                **blip_kwargs,
            )
            captions.append(caption)
            if debug:
                print_prefixed("(BLIP)", f"{caption}", indent=2)

        new_scene = dict(scene)
        new_scene["frame_captions"] = captions
        enriched_scenes.append(new_scene)

    return enriched_scenes

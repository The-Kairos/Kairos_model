from typing import List, Dict
from src.debug_utils import print_prefixed
import cv2
import numpy as np
import torch
from typing import Optional
from PIL import Image
from src.path_utils import is_low_mem

# ======================================================================
# Load BLIP model and processor
from transformers import BlipProcessor, BlipForConditionalGeneration
_blip_models = {}
_blip_processors = {}


def resolve_blip_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_blip_model(device: Optional[str] = None):
    """
    Lazy load BLIP model only when needed.
    """
    resolved_device = resolve_blip_device(device)
    if resolved_device not in _blip_models:
        _blip_models[resolved_device] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(resolved_device)
        _blip_processors[resolved_device] = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
    return _blip_models[resolved_device], _blip_processors[resolved_device]


def unload_blip(device: Optional[str] = None):
    """Explicitly unload BLIP from memory/GPU only if LOW_MEM_MODE is True."""
    if not is_low_mem():
        return

    targets = [resolve_blip_device(device)] if device else list(_blip_models.keys())
    for target in targets:
        model = _blip_models.pop(target, None)
        processor = _blip_processors.pop(target, None)
        if model is None and processor is None:
            continue
        if model is not None:
            del model
        if processor is not None:
            del processor
        import gc
        gc.collect()
        if torch.cuda.is_available() and target.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[BLIP] Model unloaded from memory ({target}).")
# ======================================================================
# # Load BLIP2 model and processor
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-flan-t5-xl",
#     torch_dtype=torch.float32,    # CPU-friendly
# ).to(device)

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# ======================================================================


def blip_frame(
    image,
    model: Optional[BlipForConditionalGeneration] = None,
    processor: Optional[BlipProcessor] = None,
    device: Optional[str] = None,
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
    """
    Generate a BLIP caption for a single frame.

    Parameters
    ----------
    image :
        Either a NumPy array (OpenCV BGR or RGB) or a PIL.Image.
    model : BlipForConditionalGeneration
        Preloaded BLIP captioning model.
    processor : BlipProcessor
        Matching BLIP processor.
    prompt : str, optional
        Optional conditioning text, e.g. "a cartoon frame of".
        If None, uses unconditional captioning.
    max_length : int
        Maximum length of the generated caption (tokens).
    min_length : int
        Minimum length of the generated caption (tokens).
    num_beams : int
        Beam search width (higher = better but slower).
    do_sample : bool
        Whether to sample (True) or keep decoding deterministic (False).
    top_p : float
        Nucleus sampling probability mass (used when sampling).
    temperature : float
        Sampling temperature (used when sampling).
    length_penalty : float
        Exponential penalty to the length (values < 1.0 favor shorter).
    no_repeat_ngram_size : int
        Prevent repetition of n-grams of this size.
    repetition_penalty : float
        Penalty for repeated tokens (> 1.0 discourages repetition).

    Returns
    -------
    str
        Generated caption.
    """
    if model is None or processor is None:
        model, processor = _get_blip_model(device=device)

    # --- Normalize image to RGB PIL.Image ---
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Assume OpenCV BGR by default
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
    else:
        raise TypeError("image must be a PIL.Image.Image or a numpy.ndarray")

    # Figure out model device (cpu / cuda / mps)
    device = next(model.parameters()).device

    # --- Prepare inputs for BLIP ---
    if prompt is not None:
        inputs = processor(
            pil_image,
            prompt,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            pil_image,
            return_tensors="pt",
        )

    # Move tensors to same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Generate caption ---
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

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def _normalize_blip_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)
    raise TypeError("image must be a PIL.Image.Image or a numpy.ndarray")


def blip_frames_batch(
    images,
    model: Optional[BlipForConditionalGeneration] = None,
    processor: Optional[BlipProcessor] = None,
    device: Optional[str] = None,
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
) -> List[str]:
    if not images:
        return []

    if model is None or processor is None:
        model, processor = _get_blip_model(device=device)

    pil_images = [_normalize_blip_image(image) for image in images]
    model_device = next(model.parameters()).device

    if prompt is not None:
        prompts = [prompt] * len(pil_images)
        inputs = processor(images=pil_images, text=prompts, return_tensors="pt", padding=True)
    else:
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)

    inputs = {k: v.to(model_device) for k, v in inputs.items()}

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

    captions = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [caption.strip() for caption in captions]


def caption_frames(
    scenes: List[Dict],
    model: Optional[BlipForConditionalGeneration] = None,
    processor: Optional[BlipProcessor] = None,
    device: Optional[str] = None,
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
    batch_size: int = 1,
    debug: bool = False,
) -> List[Dict]:
    """
    For each scene in `scenes`, run BLIP on each frame and attach captions.

    Parameters
    ----------
    scenes : List[Dict]
        Scene dictionaries. Each scene is expected to contain a "frames" key
        with a list of images (numpy arrays or PIL images).
    model : BlipForConditionalGeneration
        Preloaded BLIP captioning model.
    processor : BlipProcessor
        Matching BLIP processor.
    prompt : str, optional
        Optional conditioning text for all captions (e.g. "a cartoon frame of").
    max_length : int
        Max caption length (tokens).
    min_length : int
        Min caption length (tokens).
    num_beams : int
        Beam search width.
    do_sample : bool
        Whether to sample or keep it deterministic.
    top_p : float
        Nucleus sampling probability mass (used when sampling).
    temperature : float
        Sampling temperature (used when sampling).
    length_penalty : float
        Exponential penalty to the length (values < 1.0 favor shorter).
    no_repeat_ngram_size : int
        Prevent repetition of n-grams of this size.
    repetition_penalty : float
        Penalty for repeated tokens (> 1.0 discourages repetition).

    Returns
    -------
    List[Dict]
        New list of scenes; each scene dict has an extra key:
            "frame_captions": List[str]
        aligned 1:1 with the "frames" list.
    """
    import gc
    import psutil

    enriched_scenes: List[Dict] = []

    # Lazy load BLIP once for the entire batch
    resolved_device = resolve_blip_device(device)
    model, processor = _get_blip_model(device=resolved_device)
    actual_batch_size = max(1, int(batch_size))

    for scene in scenes:
        if debug:
            scene_idx = scene.get("scene_index", "??")
            print_prefixed("(BLIP)", f"Scene {scene_idx}")
        frames = scene.get("frames", [])
        captions: List[str] = []
        for start in range(0, len(frames), actual_batch_size):
            batch_frames = frames[start:start + actual_batch_size]
            batch_captions = blip_frames_batch(
                images=batch_frames,
                model=model,
                processor=processor,
                device=resolved_device,
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
            captions.extend(batch_captions)
            if debug:
                for caption in batch_captions:
                    print_prefixed("(BLIP)", f"{caption}", indent=2)

        new_scene = dict(scene)  # shallow copy so we don't mutate original reference
        new_scene["frame_captions"] = captions
        
        # Frame Purging: Free up memory after processing
        if "frames" in scene:
            del scene["frames"]
        if "frames" in new_scene:
            del new_scene["frames"]
        gc.collect()

        if debug:
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            print_prefixed("(BLIP)", f"Memory usage: {mem:.2f} MB", indent=4)

        enriched_scenes.append(new_scene)

    # Explicitly unload after the entire batch is done
    unload_blip(device=resolved_device)

    return enriched_scenes


'''
captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=50,
    min_length=20,
    num_beams=1,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    length_penalty=0.8,
    no_repeat_ngram_size=2,
    repetition_penalty=1.2,
    debug=True,
    prompt="a video frame of"
)

Scene 0
  a video frame of a room with a blue door and a pink flower on the floor
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
Scene 1
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
Scene 2
  a video frame of blue and green stripes
  a video frame of a cartoon character holding a sword
  a video frame of a bottle of beer
  a video frame of a man with a hat on his head
Scene 3
  a video frame of a piece of paper on a table
  a video frame of a person writing on a piece of paper
  a video frame of an airplane flying through the sky
  a video frame of a pencil on a piece of paper
Scene 4
  a video frame of a cartoon character sitting on a chair
  a video frame of a sponge sponge and his pencil
  a video frame of a cartoon character
  a video frame of a cartoon character with blue eyes and a smile
Scene 5
  a video frame of an airplane flying in the sky
  a video frame of a hand holding a pencil
  a video frame of a cartoon character holding a pencil
  a video frame of a cartoon character holding a pencil
Scene 6
  a video frame of a cartoon character sitting at a table
  a video frame of a sponge sponge with a piece of paper
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge with a piece of paper
Scene 7
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
Scene 8
  a video frame of a sponge sponge and his friend
  a video frame of sponge spongenan ' s revenge
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
Scene 9
  a video frame of a black background with a white border
'''

from typing import List, Dict
from src.debug_utils import print_prefixed
import cv2
import numpy as np
import torch
from typing import Optional
from PIL import Image
from src.path_utils import is_low_mem

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# Load BLIP model and processor
from transformers import BlipProcessor, BlipForConditionalGeneration
_blip_model = None
_blip_processor = None

def _get_blip_model():
    """
    Lazy load BLIP model only when needed.
    """
    global _blip_model, _blip_processor
    if _blip_model is None:
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
    return _blip_model, _blip_processor

def unload_blip():
    """Explicitly unload BLIP from memory/GPU only if LOW_MEM_MODE is True."""
    global _blip_model, _blip_processor
    if _blip_model is not None and is_low_mem():
        del _blip_model
        del _blip_processor
        _blip_model = None
        _blip_processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[BLIP] Model unloaded from memory (LowMem mode).")
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
        model, processor = _get_blip_model()

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


def caption_frames(
    scenes: List[Dict],
    model: Optional[BlipForConditionalGeneration] = None,
    processor: Optional[BlipProcessor] = None,
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
    model, processor = _get_blip_model()

    for scene in scenes:
        if debug:
            scene_idx = scene.get("scene_index", "??")
            print_prefixed("(BLIP)", f"Scene {scene_idx}")
        frames = scene.get("frames", [])
        captions: List[str] = []

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
    unload_blip()

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

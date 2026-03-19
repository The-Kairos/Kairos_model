"""
Light VLM: BLIP-2 OPT 2.7B (Salesforce). Image captioning with OPT small.
Uses Blip2Processor and Blip2ForConditionalGeneration from transformers.
"""
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

DEFAULT_PROMPT = "a photo of"


def load_vlm_model(model_id="Salesforce/blip2-opt-2.7b"):
    """Load BLIP-2 OPT 2.7B (small). Returns (model, processor)."""
    print(f"Loading {model_id}...")
    processor = Blip2Processor.from_pretrained(model_id)
    # Avoid meta device error (lm_head missing from checkpoint): load without device_map then move
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,  # avoids meta device when lm_head is newly initialized
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor


def caption_image(model, processor, image, prompt=None):
    """
    Caption image using BLIP-2.
    image: PIL Image
    processor: Blip2Processor
    """
    if prompt is None:
        prompt = DEFAULT_PROMPT
    if hasattr(image, "save"):
        img = image
    else:
        from PIL import Image
        img = Image.open(image).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


if __name__ == "__main__":
    from benchmark_utils import benchmark_inference, load_test_image
    TEST_IMAGE = "test_frame.jpg"
    if not os.path.exists(TEST_IMAGE):
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)
    model, processor = load_vlm_model()
    image = load_test_image(TEST_IMAGE)
    result, metrics = benchmark_inference(caption_image, model, processor, image)
    print("Result:", result)
    print("Metrics:", metrics)

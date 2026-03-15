"""Light VLM: LLaVA v1.6 Mistral 7B. Lighter chat-style VLM."""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import os

def load_vlm_model(model_id="llava-hf/llava-v1.6-mistral-7b-hf"):
    """Load and return the 4-bit quantized LLaVA-Mistral model and processor."""
    print(f"Loading {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, processor

def caption_image(model, processor, image, prompt=None):
    """Generate a scene caption for *image* using LLaVA-Mistral."""
    if prompt is None:
        prompt = "[INST] <image>\nDescribe the scene in detail. Focus only on what is visually observable. Do not assume intentions or unseen events. Mention actions, objects, and interactions. [/INST]"
    # New LlavaNextProcessor API expects text/images kwargs, not (prompt, image)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(output[0], skip_special_tokens=True)

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

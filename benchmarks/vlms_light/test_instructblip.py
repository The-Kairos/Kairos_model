"""Light VLM: InstructBLIP (Salesforce). Instruction-following image captioning.

Uses the flan-t5-xl variant for a lighter footprint than vicuna-7b.
"""

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import os

def load_vlm_model(model_id="Salesforce/instructblip-flan-t5-xl"):
    """Load and return the InstructBLIP flan-t5-xl model and processor."""
    print(f"Loading {model_id}...")
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    return model, processor

def caption_image(model, processor, image, prompt=None):
    """Generate an instruction-guided caption for *image* using InstructBLIP."""
    if prompt is None:
        prompt = "Describe the scene in detail. Focus on what is visually observable: actions, objects, and interactions."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
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

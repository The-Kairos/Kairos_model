"""
Light VLM: BLIP-2 (OPT small only).
Image captioning via BLIP-2 OPT 1.3B.
"""
import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_vlm_model(model_id="Salesforce/blip2-opt-1.3b"):
    print(f"Loading {model_id}...")
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    return model, processor


def caption_image(model, processor, image, prompt=None):
    # BLIP-2 base usage: unconditional captioning; prompt is ignored.
    inputs = processor(images=image, return_tensors="pt").to(model.device)
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


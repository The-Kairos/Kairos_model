"""
Light VLM: TinyLLaVA (tiny multimodal LLaVA-style model).
Captioning-style interface, same contract as other light VLMs.
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_vlm_model(model_id="TinyLLaVA/TinyLLaVA-phi-2"):
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, processor


def caption_image(model, processor, image, prompt=None):
    if prompt is None:
        prompt = (
            "Describe the scene in detail. Focus only on what is visually observable: "
            "objects, actions, and interactions."
        )
    # TinyLLaVA variants typically use <image> token plus text prompt.
    text = f"<image>\n{prompt}"
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    return processor.batch_decode(output, skip_special_tokens=True)[0]


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


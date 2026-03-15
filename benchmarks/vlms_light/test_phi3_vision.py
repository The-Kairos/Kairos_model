"""Light VLM: Phi-3.5 Vision (OCR-oriented).

Uses an OCR-focused prompt for document/scene text extraction
alongside brief scene description.
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig


def load_vlm_model(model_id="microsoft/Phi-3.5-vision-instruct"):
    """Load and return the Phi-3.5 Vision model and processor with eager attention."""
    print(f"Loading {model_id}...")
    # Make sure any flash-attn toggles are off at env level as well.
    os.environ.setdefault("FLASH_ATTENTION_2_ENABLED", "0")
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, num_crops=4
    )
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Force standard attention implementation so flash_attn is not required.
    setattr(config, "attn_implementation", "eager")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, processor

def caption_image(model, processor, image, prompt=None):
    """Generate an OCR-oriented caption for *image* using Phi-3.5 Vision."""
    if prompt is None:
        prompt = "Describe what text or text-like content is visible in this image (OCR). Then briefly describe the scene: actions, objects, and interactions."
    # Phi-3.5 processor expects text with <|image_1|> and images list
    text = f"<|image_1|>\n{prompt}"
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)

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

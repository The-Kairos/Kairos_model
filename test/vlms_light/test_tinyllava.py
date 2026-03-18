"""
Light VLM: TinyLLaVA (Phi-2-SigLIP). Small-scale multimodal model with strong performance.
Uses transformers with trust_remote_code.
"""
import os
import tempfile
from PIL import Image

DEFAULT_PROMPT = "Describe the scene in detail. Focus on what is visually observable."


def load_vlm_model(model_id="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"):
    """Load TinyLLaVA model. Returns (model, tokenizer) - tokenizer used as processor for interface."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        model_max_length=config.tokenizer_model_max_length,
        padding_side=config.tokenizer_padding_side,
    )
    return model, tokenizer


def caption_image(model, processor, image, prompt=None):
    """
    Caption image using TinyLLaVA chat().
    image: PIL Image
    processor: tokenizer (TinyLLaVA uses tokenizer, not a processor)
    """
    if prompt is None:
        prompt = DEFAULT_PROMPT
    tokenizer = processor
    # TinyLLaVA chat() expects image as URL or file path
    if hasattr(image, "save"):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        image.save(path, format="JPEG")
        try:
            output_text, _ = model.chat(
                prompt=prompt,
                image=path,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0,
            )
            return output_text.strip() if output_text else ""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    else:
        output_text, _ = model.chat(
            prompt=prompt,
            image=image,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0,
        )
        return output_text.strip() if output_text else ""


if __name__ == "__main__":
    from benchmark_utils import benchmark_inference, load_test_image
    TEST_IMAGE = "test_frame.jpg"
    if not os.path.exists(TEST_IMAGE):
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)
    model, tokenizer = load_vlm_model()
    image = load_test_image(TEST_IMAGE)
    result, metrics = benchmark_inference(caption_image, model, tokenizer, image)
    print("Result:", result)
    print("Metrics:", metrics)

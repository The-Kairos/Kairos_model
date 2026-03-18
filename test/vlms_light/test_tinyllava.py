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
    import builtins
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Patch exec so when transformers loads TinyLLaVA modeling code, we fix tie_weights to accept **kwargs.
    # TinyLlavaForConditionalGeneration.tie_weights(self) doesn't accept recompute_mapping=... from newer transformers.
    _orig_exec = builtins.exec

    def _patched_exec(code, globals_dict=None, locals_dict=None, *args):
        _orig_exec(code, globals_dict, locals_dict if locals_dict is not None else globals_dict, *args)
        g = globals_dict or {}
        for name, obj in list(g.items()):
            if "TinyLlava" in name and hasattr(obj, "tie_weights"):
                _orig = getattr(obj, "tie_weights")

                def _fixed_tie(self, *a, **kw):
                    kw.pop("recompute_mapping", None)
                    return _orig(self)  # original only takes self

                setattr(obj, "tie_weights", _fixed_tie)

    builtins.exec = _patched_exec
    try:
        print(f"Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    finally:
        builtins.exec = _orig_exec
    # TinyLLaVA custom model may lack _supports_sdpa; set to avoid AttributeError during generation
    if not hasattr(model, "_supports_sdpa"):
        model._supports_sdpa = False
    model = model.eval()
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

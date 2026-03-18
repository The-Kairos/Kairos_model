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
    import sys
    import importlib.util
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Import hook to patch TinyLlavaForConditionalGeneration.tie_weights before it's called.
    # The custom model's tie_weights() doesn't accept recompute_mapping=... from newer transformers.
    _original_loader = None

    class TinyLlavaPatcher:
        def find_spec(self, name, path, target=None):
            if "modeling_tinyllava" in name or "modeling_tinyllava_phi" in name:
                return importlib.util.find_spec(name, path)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            if hasattr(module, "TinyLlavaForConditionalGeneration"):
                cls = module.TinyLlavaForConditionalGeneration
                if hasattr(cls, "tie_weights"):
                    _orig = cls.tie_weights
                    def _patched_tie(self, *args, **kwargs):
                        kwargs.pop("recompute_mapping", None)
                        return _orig(self, *args, **kwargs)
                    cls.tie_weights = _patched_tie

    # Install hook - use importlib's machinery
    import importlib.machinery
    _patcher = TinyLlavaPatcher()
    if not hasattr(_patcher, "find_spec"):
        _patcher = None

    if _patcher:
        from importlib import _bootstrap_external
        _orig_exec = _bootstrap_external._call_with_frames_removed
        def _patched_exec(code, globals, *args, **kwargs):
            result = _orig_exec(code, globals, *args, **kwargs)
            # Patch TinyLlavaForConditionalGeneration.tie_weights (any module that defines it)
            for name, obj in list(globals.items()):
                if "TinyLlava" in name and hasattr(obj, "tie_weights"):
                    _orig = getattr(obj, "tie_weights")
                    def _fixed_tie(self, *a, **kw):
                        kw.pop("recompute_mapping", None)
                        return _orig(self, *a, **kw)
                    setattr(obj, "tie_weights", _fixed_tie)
            return result
        try:
            _bootstrap_external._call_with_frames_removed = _patched_exec
        except Exception:
            pass

    print(f"Loading {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    finally:
        try:
            _bootstrap_external._call_with_frames_removed = _orig_exec
        except Exception:
            pass
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

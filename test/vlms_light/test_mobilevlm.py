"""
Light VLM: MobileVLM (Meituan). Fast vision-language model for mobile devices.
Uses the MobileVLM repo's load_pretrained_model.
Requires: pip install git+https://github.com/Meituan-AutoML/MobileVLM.git
Or: git clone https://github.com/Meituan-AutoML/MobileVLM && export PYTHONPATH=/path/to/MobileVLM:$PYTHONPATH
"""
import os
from PIL import Image

DEFAULT_PROMPT = "Describe the scene in detail. Focus on what is visually observable."


def load_vlm_model(model_id="mtgv/MobileVLM_V2-1.7B"):
    """Load MobileVLM model. Returns (model, processor_dict) with tokenizer, image_processor."""
    import sys
    from pathlib import Path
    try:
        from mobilevlm.model.mobilevlm import load_pretrained_model
    except ImportError:
        # Try adding project-level MobileVLM clone to path (from install_mobilevlm.py)
        project_root = Path(__file__).resolve().parent.parent.parent
        mobilevlm_dir = project_root / "MobileVLM"
        if mobilevlm_dir.exists() and (mobilevlm_dir / "mobilevlm").exists():
            if str(mobilevlm_dir) not in sys.path:
                sys.path.insert(0, str(mobilevlm_dir))
            try:
                from mobilevlm.model.mobilevlm import load_pretrained_model
            except ImportError as e:
                raise ImportError(
                    f"MobileVLM cloned but import failed: {e}\n"
                    "Install with: python test/vlms_light/install_mobilevlm.py\n"
                    "Or: pip install git+https://github.com/Meituan-AutoML/MobileVLM.git"
                ) from e
        else:
            raise ImportError(
                "MobileVLM requires the MobileVLM repo. Install with:\n"
                "  python test/vlms_light/install_mobilevlm.py\n"
                "Or: pip install git+https://github.com/Meituan-AutoML/MobileVLM.git"
            ) from None
    print(f"Loading {model_id}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device_map="cuda:0",  # single device to avoid multi-GPU tensor mismatch
    )
    processor = {"tokenizer": tokenizer, "image_processor": image_processor}
    return model, processor


def caption_image(model, processor, image, prompt=None):
    """
    Caption image using MobileVLM.
    image: PIL Image
    processor: dict with 'tokenizer' and 'image_processor'
    """
    from mobilevlm.mm_utils import process_images, tokenizer_image_token
    from mobilevlm.constants import IMAGE_TOKEN_INDEX
    if prompt is None:
        prompt = DEFAULT_PROMPT
    tokenizer = processor["tokenizer"]
    image_processor = processor["image_processor"]
    question = f" \n{prompt}"  # leading space = image token placeholder (MobileVLM/LLaVA style)
    if hasattr(image, "save"):
        img = image
    else:
        img = Image.open(image).convert("RGB")
    image_tensor = process_images([img], image_processor, model.config).to(model.device)
    input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
        use_cache=True,
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.strip()


if __name__ == "__main__":
    from benchmark_utils import benchmark_inference, load_test_image
    TEST_IMAGE = "test_frame.jpg"
    if not os.path.exists(TEST_IMAGE):
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)
    model, processor = load_vlm_model()
    image = load_test_image(TEST_IMAGE)
    result, metrics = benchmark_inference(caption_image, model, processor, image)
    print("Result:", result)
    print("Metrics:", metrics)

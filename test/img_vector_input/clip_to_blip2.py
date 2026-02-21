import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# Add project root and src to sys.path so local imports work if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Usage:
    #   python test/clip_to_blip2.py [image_path] [blip2_model_id]
    image_path = PROJECT_ROOT / "test" / "woman_driving.jpg"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    blip2_id = "Salesforce/blip2-opt-2.7b"
    if len(sys.argv) > 2:
        blip2_id = sys.argv[2]

    prompt = "a photo of"

    image = load_image(image_path)
    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"BLIP-2 model: {blip2_id}")

    print("Loading BLIP-2...")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_id, torch_dtype=dtype
    ).to(device)
    blip2_processor = Blip2Processor.from_pretrained(blip2_id)

    print("BLIP-2 baseline with raw image...")
    inputs = blip2_processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = blip2_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
        )

    caption = blip2_processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    print("BLIP-2 (image) =>", caption)

    print("BLIP-2-style with internal vision encoder (manual pipeline)...")
    try:
        pixel_values = inputs["pixel_values"]

        with torch.no_grad():
            lang_prefix, _, _ = blip2_model.get_image_features(
                pixel_values=pixel_values, return_dict=True
            )

        tokenizer = blip2_processor.tokenizer
        text_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = text_inputs.input_ids.to(device)
        text_embeds = blip2_model.language_model.get_input_embeddings()(input_ids)

        lang_prefix = lang_prefix.to(text_embeds.device, text_embeds.dtype)
        inputs_embeds = torch.cat([lang_prefix, text_embeds], dim=1)
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        with torch.no_grad():
            out_ids = blip2_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=30,
                num_beams=3,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        clip_caption = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        print("BLIP-2 (vision -> Q-Former -> LLM) =>", clip_caption)
    except Exception as exc:
        print("BLIP-2 (CLIP -> Q-Former -> LLM) failed:", repr(exc))


if __name__ == "__main__":
    main()

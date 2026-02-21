import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor

# Add project root and src to sys.path so local imports work if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def make_test_image(size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 40, 120, 180], fill=(255, 60, 60))
    draw.ellipse([140, 60, 210, 130], fill=(60, 120, 255))
    draw.text((10, 10), "test", fill=(10, 10, 10))
    return img


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    image_path = PROJECT_ROOT / "test" / "woman_driving.jpg"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    image = load_image(image_path)
    print(f"Image: {image_path}")

    print("Loading CLIP...")
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)

    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])

    clip_hidden = vision_outputs.last_hidden_state.detach()
    clip_attn = torch.ones(clip_hidden.shape[:2], dtype=torch.long, device=clip_hidden.device)
    print(f"CLIP last_hidden_state shape: {tuple(clip_hidden.shape)}")

    print("Loading BLIP...")
    blip_id = "Salesforce/blip-image-captioning-base"
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_id).to(device)
    blip_processor = BlipProcessor.from_pretrained(blip_id)

    prompt = "a photo of"

    print("BLIP baseline with raw image...")
    blip_inputs = blip_processor(images=image, text=prompt, return_tensors="pt")
    blip_inputs = {k: v.to(device) for k, v in blip_inputs.items()}

    with torch.no_grad():
        baseline_ids = blip_model.generate(
            **blip_inputs,
            max_length=30,
            num_beams=3,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
        )

    baseline_caption = blip_processor.decode(baseline_ids[0], skip_special_tokens=True)
    print("BLIP (image) =>", baseline_caption)

    print("BLIP with CLIP embeddings via text_decoder (experimental)...")
    try:
        text_inputs = blip_processor(text=prompt, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            clip_ids = blip_model.text_decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=clip_hidden,
                encoder_attention_mask=clip_attn,
                max_length=30,
                num_beams=3,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )

        clip_caption = blip_processor.decode(clip_ids[0], skip_special_tokens=True)
        print("BLIP (CLIP embeds) =>", clip_caption)
    except Exception as exc:
        print("BLIP (CLIP embeds) failed:", repr(exc))


if __name__ == "__main__":
    main()

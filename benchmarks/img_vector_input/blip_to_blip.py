import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Add project root and src to sys.path so local imports work if needed
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def encode_image(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    device: str,
) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
    return vision_outputs.last_hidden_state.detach()


def decode_caption(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    encoder_hidden_states: torch.Tensor,
    device: str,
    prompt: str = "a photo of",
) -> str:
    text_inputs = processor(text=prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    encoder_attention_mask = torch.ones(
        encoder_hidden_states.shape[:2], dtype=torch.long, device=device
    )

    with torch.no_grad():
        out_ids = model.text_decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_length=30,
            num_beams=3,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
        )

    return processor.decode(out_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = BASE_DIR / "woman_driving.jpg"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    image = load_image(image_path)

    blip_id = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(blip_id).to(device)
    processor = BlipProcessor.from_pretrained(blip_id)
    model.eval()

    prompt = "a photo of"

    start = time.perf_counter()
    embedding = encode_image(model, processor, image, device)
    embedding_time = time.perf_counter() - start

    start = time.perf_counter()
    caption = decode_caption(model, processor, embedding, device, prompt=prompt)
    caption_time = time.perf_counter() - start

    result = {
        "vector": embedding.detach().cpu().float().tolist(),
        "caption": caption,
        "embedding_time": embedding_time,
        "caption_time": caption_time,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

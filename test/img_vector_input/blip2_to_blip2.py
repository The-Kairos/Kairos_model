import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

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
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    image: Image.Image,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.no_grad():
        lang_prefix, _, _ = model.get_image_features(
            pixel_values=pixel_values, return_dict=True
        )
    return lang_prefix.detach()


def decode_caption(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    lang_prefix: torch.Tensor,
    device: str,
    prompt: str = "a photo of",
) -> str:
    tokenizer = processor.tokenizer
    text_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    text_embeds = model.language_model.get_input_embeddings()(input_ids)

    lang_prefix = lang_prefix.to(text_embeds.device, text_embeds.dtype)
    inputs_embeds = torch.cat([lang_prefix, text_embeds], dim=1)
    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=30,
            num_beams=3,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    image_path = BASE_DIR / "woman_driving.jpg"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    blip2_id = "Salesforce/blip2-opt-2.7b"
    if len(sys.argv) > 2:
        blip2_id = sys.argv[2]

    image = load_image(image_path)

    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_id, torch_dtype=dtype
    ).to(device)
    blip2_processor = Blip2Processor.from_pretrained(blip2_id)
    blip2_model.eval()

    prompt = "a photo of"

    start = time.perf_counter()
    embedding = encode_image(blip2_model, blip2_processor, image, device, dtype)
    embedding_time = time.perf_counter() - start

    start = time.perf_counter()
    caption = decode_caption(blip2_model, blip2_processor, embedding, device, prompt)
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

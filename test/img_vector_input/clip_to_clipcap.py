import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer

# Add project root and src to sys.path so local imports work if needed
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


class MLP(torch.nn.Module):
    def __init__(self, prefix_size: int, intermediate_size: int, out_size: int) -> None:
        super().__init__()
        self.proj1 = torch.nn.Linear(prefix_size, intermediate_size, bias=True)
        self.proj2 = torch.nn.Linear(intermediate_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.tanh(self.proj1(x))
        return self.proj2(z)


class ClipCapModel(torch.nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        from transformers import GPT2LMHeadModel

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.mapping = MLP(
            prefix_size=prefix_size,
            intermediate_size=(self.gpt_embedding_size * prefix_length) // 2,
            out_size=self.gpt_embedding_size * prefix_length,
        )


def generate_greedy(
    model: ClipCapModel,
    tokenizer: GPT2Tokenizer,
    prefix_embed: torch.Tensor,
    max_length: int = 30,
) -> str:
    model.eval()
    tokens = None
    generated = prefix_embed
    eos = tokenizer.eos_token_id or 50256

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            next_embed = model.gpt.transformer.wte(next_token)
            generated = torch.cat((generated, next_embed), dim=1)
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)
            if next_token.item() == eos:
                break

    if tokens is None:
        return ""
    return tokenizer.decode(tokens.squeeze().tolist(), skip_special_tokens=True).strip()


def encode_image(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image: Image.Image,
    device: str,
) -> torch.Tensor:
    pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(pixel_values=pixel_values).float()
    return clip_features.detach()


def decode_caption(
    model: ClipCapModel,
    tokenizer: GPT2Tokenizer,
    clip_features: torch.Tensor,
    max_length: int = 30,
) -> str:
    prefix_embed = model.mapping(clip_features).view(
        -1, model.prefix_length, model.gpt_embedding_size
    )
    return generate_greedy(model, tokenizer, prefix_embed, max_length=max_length)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = BASE_DIR / "woman_driving.jpg"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    image = load_image(image_path)

    repo_id = "saad1926q/clipcap-image-captioning"
    ckpt_name = "coco_prefix_best_200k.pt"
    prefix_length = 10

    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = ClipCapModel(prefix_length=prefix_length).to(device)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    clip_model.eval()
    model.eval()

    start = time.perf_counter()
    embedding = encode_image(clip_model, clip_processor, image, device)
    embedding_time = time.perf_counter() - start

    start = time.perf_counter()
    caption = decode_caption(model, tokenizer, embedding, max_length=30)
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

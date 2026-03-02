"""
Light VLM: SigLIP (retrieval-first). No direct caption; retrieve best-matching template description.
"""
import torch
from transformers import AutoProcessor, AutoModel
import os

# Template scene descriptions for retrieval-based "caption"
SCENE_TEMPLATES = [
    "A person is speaking or presenting.",
    "People are having a conversation.",
    "Cooking or food preparation in a kitchen.",
    "Outdoor scene with nature or buildings.",
    "Sports or physical activity.",
    "Animals or pets in the scene.",
    "Vehicle or traffic.",
    "Indoor room with furniture.",
    "Screen or monitor with text or graphics.",
    "Crowd or group of people.",
    "Close-up of a face or person.",
    "Hands working on something.",
    "Document or paper with text.",
    "Abstract or unclear scene.",
]

def load_vlm_model(model_id="google/siglip-base-patch16-224"):
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    return model, processor

def caption_image(model, processor, image, prompt=None):
    """Encode image, score against SCENE_TEMPLATES, return top-3 as single caption."""
    inputs = processor(
        text=SCENE_TEMPLATES,
        images=image,
        padding="max_length",
        return_tensors="pt",
        truncation=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image: [1, num_templates]
    logits = outputs.logits_per_image[0]
    probs = torch.sigmoid(logits)
    top_k = min(3, len(SCENE_TEMPLATES))
    top_indices = torch.topk(probs, top_k).indices.cpu().tolist()
    captions = [SCENE_TEMPLATES[i] for i in top_indices]
    return " | ".join(captions)

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

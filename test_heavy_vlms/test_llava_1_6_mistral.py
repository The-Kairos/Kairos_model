# test_heavy_vlms/test_llava_1_6_mistral.py
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

def load_vlm_model(model_id="llava-hf/llava-v1.6-mistral-7b-hf"):  # ← CHANGE
    print(f"Loading {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"  # ← ADD for better quality
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto",
        torch_dtype=torch.float16  # ← ADD for consistency
    )
    return model, processor

def caption_frames(model, processor, frames_list, prompt=None):
    if not frames_list:
        return ""

    if prompt is None:
        # Standard LLaVA 1.6 Prompt Template for multi-image
        # We need <image> token for EACH image
        image_tokens = "<image>" * len(frames_list)
        prompt = f"USER: {image_tokens}\nDescribe the scene in detail based on these frames. Focus only on what is visually observable. Mention actions, objects, and interactions. ASSISTANT:"
    
    # LlavaNextProcessor handles list of images automatically
    inputs = processor(text=prompt, images=frames_list, return_tensors="pt").to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    model, processor = load_vlm_model()
    # Create 2 dummy frames
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)) 
        for _ in range(2)
    ]
    
    print(caption_frames(model, processor, dummy_frames))
    
    
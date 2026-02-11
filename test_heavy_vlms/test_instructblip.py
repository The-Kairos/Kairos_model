import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import platform
import warnings
import logging

# Suppress noisy warnings from Accelerate and Transformers
warnings.filterwarnings("ignore", message=".*language_model.*not in the hf_device_map.*")
warnings.filterwarnings("ignore", message=".*The following device_map keys do not match.*")
warnings.filterwarnings("ignore", message=".*weights are not tied.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_vlm_model(model_id="Salesforce/instructblip-vicuna-7b"):
    print(f"Loading {model_id}...")
    
    # Quantization for 7B model on consumer/cloud GPUs
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=True)
    
    # Reverting to 'auto' because manual mapping missed 'language_projection'.
    # We will use warnings filtering to keep the terminal clean instead.
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, processor

def caption_frames(model, processor, frames_list, prompt=None):
    """
    Generate a caption for a list of frames. 
    InstructBLIP is primarily single-image. We will concatenate frames horizontally
    to create a 'panorama' of the scene, which is a common strategy for 
    image-based VLMs handling multi-frame context without native video support.
    """
    if not frames_list:
        return ""
    
    if prompt is None:
        prompt = "Describe the video scene shown in these frames. Mention key actions, objects, and the setting."

    # Concatenate frames horizontally
    total_width = sum(img.width for img in frames_list)
    max_height = max(img.height for img in frames_list)
    
    concat_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in frames_list:
        concat_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Prepare inputs
    inputs = processor(images=concat_img, text=prompt, return_tensors="pt").to("cuda")
    
    # Generate
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=256,
        min_length=10,
        num_beams=5
    )
    
    # Decode only the NEW tokens
    caption = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()
    return caption

if __name__ == "__main__":
    # Test stub
    import numpy as np
    model, processor = load_vlm_model()
    # Create 3 dummy frames
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) 
        for _ in range(3)
    ]
    print(caption_frames(model, processor, dummy_frames))

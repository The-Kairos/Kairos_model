import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

from transformers.cache_utils import DynamicCache

# Monkeypatches for Phi-3.5-Vision compatibility with transformers 4.57.1
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_length", None)
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, seq_length, layer_idx=None: self.get_seq_length()

# Note: We use 'eager' implementation for Phi-3.5-Vision to avoid attention issues 
# in transformers 4.57.1 when Flash Attention 2 is not available.

def load_vlm_model(model_id="microsoft/Phi-3.5-vision-instruct"):
    print(f"Loading {model_id}...")
    
    # Load model with specific configuration for stability
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype=torch.float16, # Explicit half precision
        _attn_implementation='eager' 
    ).eval()
    
    # Note: Phi-3.5 uses a processor, not just a tokenizer
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
    return model, processor

def caption_frames(model, processor, frames_list, prompt="Describe the video scene created by these frames in detail."):
    if not frames_list:
        return ""
        
    # Construct Phi-3.5 prompt for multi-frame
    # We treat multiple frames as multiple images <|image_1|>, <|image_2|>, etc.
    
    content = ""
    for i in range(len(frames_list)):
        content += f"<|image_{i+1}|>\n"
    content += prompt
    
    messages = [
        {"role": "user", "content": content}
    ]
    
    prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(prompt_text, frames_list, return_tensors="pt").to("cuda")
    
    generation_args = {
        "max_new_tokens": 500,
        "do_sample": False,
        # IMPORTANT: use_cache=False is REQUIRED for stability because the 'eager' 
        # implementation in transformers 4.57 has a KV-cache dimension bug.
        # This makes Phi-3.5 slower than LLaVA.
        "use_cache": False, 
    }
    
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    
    # remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response.strip()

if __name__ == "__main__":
    import numpy as np
    model, processor = load_vlm_model()
    dummy_frames = [
        Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8)) 
        for _ in range(2)
    ]
    print(caption_frames(model, processor, dummy_frames))

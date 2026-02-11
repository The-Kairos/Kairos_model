import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from transformers.cache_utils import DynamicCache

# Monkeypatches for Phi-3.5-Vision compatibility with transformers 4.57.1
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_length", None)
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, seq_length, layer_idx=None: self.get_seq_length()

def test_phi3_cache():
    model_id = "microsoft/Phi-3.5-vision-instruct"
    print(f"Loading {model_id} with SDPA...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        _attn_implementation='eager' 
    ).eval()
    
    # --- TEST: Minimal crops and Cache ENABLED ---    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=1)
    
    dummy_frame = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
    content = "<|image_1|>\nDescribe the image."
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(prompt_text, [dummy_frame], return_tensors="pt").to("cuda")
    
    print("Generating with num_crops=1 and use_cache=True...")
    try:
        output = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False, 
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        print("Success! Minimal crops + cache works.")
    except Exception as e:
        print(f"Failed: {e}")
        
        print("\nRetrying with Eager + cache...")
        # Since we can't easily change the model's implementation without reloading, 
        # this test is limited. But let's see.

if __name__ == "__main__":
    test_phi3_cache()

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def load_vlm_model(model_id="microsoft/Phi-3.5-vision-instruct"):
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2' 
    ).eval()
    
    # Note: Phi-3.5 uses a processor, not just a tokenizer
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)
    return model, processor

def caption_image(model, processor, pil_image, prompt="Describe the scene in detail."):
    # Construct Phi-3.5 prompt
    messages = [
        {"role": "user", "content": "<|image_1|>\n" + prompt}
    ]
    
    prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(prompt_text, [pil_image], return_tensors="pt").to("cuda")
    
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    
    # remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response.strip()

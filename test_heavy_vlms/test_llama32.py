import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image

def load_vlm_model(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    print(f"Loading {model_id}...")
    # Llama 3.2 11B fits in 24GB RAM with 8-bit or 4-bit quantization, or BF16 if tight.
    # We'll use 4-bit to be safe and fast.
    from transformers import BitsAndBytesConfig
    pass
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("    Using 4-bit quantization")
    except ImportError:
        quantization_config = None
        print("    bitsandbytes not found, trying fp16 (might OOM on small GPUs)")

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config if quantization_config else None,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        device_map="auto",
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def caption_image(model, processor, pil_image, prompt="Describe the scene in detail."):
    # Llama 3.2 Vision prompt format
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        pil_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500)
        
    # Decode
    response = processor.decode(output[0])
    
    # Extract persistent response (remove input prompt)
    # Llama 3.2 typically puts response after <|start_header_id|>assistant<|end_header_id|>
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    # Clean up other tokens
    response = response.replace("<|eot_id|>", "").strip()
    
    return response

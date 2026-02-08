import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import os

def load_vlm_model(model_id="llava-hf/llava-v1.6-vicuna-7b-hf"):
    print(f"Loading {model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    return model, processor

def caption_image(model, processor, image, prompt=None):
    if prompt is None:
        # Standard LLaVA 1.6 Prompt Template
        prompt = "USER: <image>\nDescribe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions. ASSISTANT:"
    
    # Use explicit keyword arguments to avoid "Incorrect image source" error
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(output[0], skip_special_tokens=True)

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

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        # Create a dummy image if not exists for dry-run
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)
        print(f"Created dummy image {TEST_IMAGE}")

    test_llava()

import torch
from transformers import AutoModel, AutoTokenizer
import os

def load_vlm_model(model_id="OpenGVLab/InternVL-Chat-V1-5"):
    print(f"Loading {model_id}...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

def caption_image(model, tokenizer, image, question=None):
    if question is None:
        question = "Describe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions."
    
    # InternVL preprocessing - simplified, in real high-res InternVL this might be more complex
    pixel_values = model.extract_feature(image)
    
    response, history = model.chat(tokenizer, pixel_values, question, generation_config={"max_new_tokens": 256})
    return response

if __name__ == "__main__":
    from benchmark_utils import benchmark_inference, load_test_image
    TEST_IMAGE = "test_frame.jpg"
    if not os.path.exists(TEST_IMAGE):
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)

    model, tokenizer = load_vlm_model()
    image = load_test_image(TEST_IMAGE)
    
    result, metrics = benchmark_inference(caption_image, model, tokenizer, image)
    print("Result:", result)
    print("Metrics:", metrics)

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        dummy_img.save(TEST_IMAGE)
        print(f"Created dummy image {TEST_IMAGE}")

    test_internvl()

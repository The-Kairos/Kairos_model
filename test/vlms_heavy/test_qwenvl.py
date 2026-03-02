import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from PIL import Image

def load_vlm_model(model_id="Qwen/Qwen-VL-Chat"):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        trust_remote_code=True, 
        fp16=True
    ).eval()
    return model, tokenizer

def caption_image(model, tokenizer, image, question=None):
    if question is None:
        question = "Describe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions."
    
    # Qwen-VL needs a file path or a special format. 
    # To avoid saving files constantly in a library-like call, we use the list_format
    # But Qwen-VL-Chat's chat method usually expects a path in the list_format.
    # We'll save to a temp file for compatibility.
    temp_path = "temp_qwen_frame.jpg"
    if isinstance(image, Image.Image):
        image.save(temp_path)
    else:
        # assume numpy array
        Image.fromarray(image).save(temp_path)
        
    query = tokenizer.from_list_format([
        {'image': temp_path},
        {'text': question},
    ])
    
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

if __name__ == "__main__":
    from benchmark_utils import benchmark_inference, load_test_image
    TEST_IMAGE = "test_frame.jpg"
    if not os.path.exists(TEST_IMAGE):
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

    test_qwenvl()

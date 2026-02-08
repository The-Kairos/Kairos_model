import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_vlm_model(model_id="OpenGVLab/InternVL-Chat-V1-5"):
    print(f"Loading {model_id}...")
    # Using AutoModelForCausalLM to ensure generate() attribute is available
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def caption_image(model, tokenizer, image, question=None):
    if question is None:
        question = "Describe the scene in detail. Focus only on what is visually observable. Mention actions, objects, and interactions."
    
    # Standard InterVL preprocessing for Chat-V1-5
    transform = build_transform(input_size=448)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
    
    generation_config = dict(max_new_tokens=256, do_sample=False)
    
    response, history = model.chat(tokenizer, pixel_values, question, generation_config)
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def load_vlm_model(model_id="OpenGVLab/InternVL-Chat-V1-5"):
    print(f"Loading {model_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Using AutoModelForCausalLM with trust_remote_code=True
    # InternVL-Chat-V1-5 specifically needs the CausalLM class for .generate() or .chat()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Verification: Check if generate exists
    if not hasattr(model, 'generate') and not hasattr(model, 'chat'):
        print(f"      [Warning] Model {model_id} missing 'generate' and 'chat'.")
    return model, tokenizer

def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set([(i, j) for i in range(1, max_num + 1) for j in range(1, max_num + 1) if i * j <= max_num and i * j >= min_num])
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_aspect_ratio[0])) * image_size, (i // (target_aspect_ratio[0])) * image_size, ((i % (target_aspect_ratio[0])) + 1) * image_size, ((i // (target_aspect_ratio[0])) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = image_file.convert('RGB') if not isinstance(image_file, str) else Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)

def caption_image(model, tokenizer, pil_image, prompt="Describe the scene in detail."):
    import torch
    pixel_values = load_image(pil_image, input_size=448, max_num=6).to(torch.bfloat16).cuda()
    generation_config = dict(num_beams=1, max_new_tokens=512, do_sample=False)
    try:
        # Fixed: Using .chat() which is robust for InternVL
        response, history = model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response.strip()
    except Exception as e:
        print(f"      [InternVL Error] .chat() failed: {e}. Trying .generate()...")
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_new_tokens=100)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def run_inference(model, tokenizer, scenes, video_path):
    vlm_scenes = []
    for scene in scenes:
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        mid_frame_idx = int(((scene["start_seconds"] + scene["end_seconds"]) / 2) * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = video.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = caption_image(model, tokenizer, pil_img)
            new_scene = dict(scene); new_scene["frame_captions"] = [caption]; vlm_scenes.append(new_scene)
        else:
            new_scene = dict(scene); new_scene["frame_captions"] = ["None"]; vlm_scenes.append(new_scene)
        video.release()
    return vlm_scenes

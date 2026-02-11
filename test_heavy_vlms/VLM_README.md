# Heavy VLM Benchmark

This directory contains scripts to benchmark "Heavy" Vision-Language Models (VLMs) for Video Understanding.
We focus on models < 13B parameters that can run on consumer/cloud GPUs (L4/A10/A100) while providing rich scene descriptions.

## Selected Models

| Model | ID | Type | Params | Notes |
|-------|----|------|--------|-------|
| **LLaVA-1.6** | `llava-hf/llava-v1.6-vicuna-7b-hf` | Multi-Image | 7B | Strong general baseline. Uses multi-image input. |
| **Phi3.5-V** | `microsoft/Phi-3.5-vision-instruct` | Multi-Image | 4.2B | Lightweight, high capability. Uses `_attn_implementation='eager'`. |
| **InstructBLIP** | `Salesforce/instructblip-vicuna-7b` | Image-based | 7B | Instruction-tuned. Frames are concatenated horizontally. |

## Methodology

### 1. Isolated Execution
Each VLM run typically consumes 12-20GB of VRAM. To avoid OOM errors, we run each VLM on each video in a **separate OS process**.
- `run_benchmark.py`: Orchestrator (loops videos, loops VLMs).
- `run_single_vlm.py`: Worker (loads 1 model, processes 1 video, exits).

### 2. Multi-Frame Sampling (Context)
To improve scene understanding and reduce hallucinations compared to single-frame captioning:
- We sample **2-5 frames** evenly distributed across each scene.
- **LLaVA-1.6 / Phi-3**: Pass frames as a list of images (multi-image prompt).
- **InstructBLIP**: Concatenate frames horizontally into a single large image.

#### 🧠 VLM Implementation & Context Limits

##### **A. LLaVA-1.6-Vicuna (7B)**
Uses a **List of Images** strategy.
- **The Code:**
  ```python
  # From test_llava_1_6.py
  image_tokens = "<image>" * len(frames_list)
  prompt = f"USER: {image_tokens}\nDescribe the scene... ASSISTANT:"
  inputs = processor(text=prompt, images=frames_list, return_tensors="pt")
  ```
- **Context Limitation**: LLaVA-1.6-7B has a **4,096 token limit**.
- **Why it's restrictive**: It uses "AnyRes" which splits a high-res image into multiple patches. One image can consume **~576 to 2,000+ tokens**. Passing 5 frames easily overflows the 4k limit, causing repetition or hallucinations. **2 frames** is the sweet spot for 7B stability.

##### **B. Phi-3.5-Vision (4B)**
Uses **Native Multi-Image** indexing.
- **The Code:**
  ```python
  # From test_phi3v.py
  content = ""
  for i in range(len(frames_list)):
      content += f"<|image_{i+1}|>\n"
  content += prompt
  ```
- **Context Limitation**: Massive **128,000 token limit**.
- **Why it's powerful**: Designed for long-context reasoning. You can passing dozens of frames without hitting the token cap. The primary bottleneck is VRAM and the KV-cache stability (managed by disabling cache in this pipeline).

##### **C. InstructBLIP (7B)**
Uses the **Panorama Concatenation** strategy.
- **The Code:**
  ```python
  # From test_instructblip.py
  total_width = sum(img.width for img in frames_list)
  concat_img = Image.new('RGB', (total_width, max_height))
  # ... pastes all frames side-by-side into ONE image ...
  inputs = processor(images=concat_img, text=prompt, return_tensors="pt")
  ```
- **Context Limitation**: **512 to 2,048 tokens**.
- **Why we concatenate**: The Q-Former was only trained to project **one** set of visual features. By merging frames into a single wide "panorama," we trick the model into seeing a temporal sequence as one visual space.

##### 🚀 Summary Table
| Model | Context Window | Multi-Frame Logic | Best Strategy |
| :--- | :--- | :--- | :--- |
| **LLaVA 1.6** | 4k (Small) | Token List | 2-3 frames |
| **Phi-3.5** | 128k (Huge) | Native Indexing | 5-10+ frames |
| **InstructBLIP** | ~2k (Small) | Canvas Merge | 3-5 frames |

## Generation Parameters & Hallucination Prevention

To ensure accurate, focused descriptions and minimize hallucinations, we use the following settings for all 7B models:

| Model | Temperature | Top-P / Beams | Max Tokens | Context Strategy |
|-------|-------------|---------------|------------|------------------|
| **LLaVA-1.6** | 0.0 (Greedy) | N/A | 256 | **2 frames** per scene (prevents context overflow). |
| **Phi3.5-V** | 0.0 (Greedy) | N/A | 500 | Native multi-image handling. |
| **InstructBLIP** | N/A | 5 Beams | 256 | Deterministic beam search + frame concatenation. |

> [!TIP]
> **Greedy Decoding**: By setting `temperature=0.0` and `do_sample=False`, we force the model to pick the most likely token, leading to more factual and deterministic descriptions.
> **Context Window**: For LLaVA, using too many frames (e.g. 5-10) pushes the model past its optimal context window, leading to "repetition" or "hallucinations". We limit this to 2 frames for maximum stability.

## Easy Automation Script

A master bash script `run_pipeline.sh` is provided to run the pipeline in stages. Each command starts a fresh Python process to ensure memory is cleared.

```bash
# 1. Process base data (ASR/AST/YOLO) for all videos
./run_pipeline.sh base

# 2. Run a specific VLM for all videos
./run_pipeline.sh vlm llava
./run_pipeline.sh vlm instructblip
./run_pipeline.sh vlm phi3v

# 3. Run everything (base + all VLMs)
./run_pipeline.sh all
```

## Manual Execution Guides

If you want to run the pipeline in stages (e.g., process all videos first, then run VLMs manually), use these commands:

### 1. Run Base Processing for a Video
Use `process_base.py` to generate scenes, ASR, AST, and YOLO data.
```bash
# Usage: python process_base.py <video_path> <output_json_path>
python process_base.py "../Videos/my_video.mp4" "results/base/my_video/base_data.json"
```

### 2. Run a Specific VLM on a Video
Once base data exists, use `run_single_vlm.py` to run one model.
```bash
# Usage: python run_single_vlm.py <vlm_name> <video_path> <base_data_path> <output_dir>
python run_single_vlm.py llava "../Videos/my_video.mp4" "results/base/my_video/base_data.json" "results/llava/my_video"
```
*Valid VLM names*: `llava`, `instructblip`, `phi3v`.

### 3. Loop Base Processing (All Videos)
You can use a simple bash loop to process every video in the folder:
```bash
for vid in ../Videos/*.mp4; do
  name=$(basename "$vid" .mp4)
  python process_base.py "$vid" "results/base/$name/base_data.json"
done
```

## Python File Summary

| File | Purpose |
|------|---------|
| **`process_base.py`** | Worker: Processes 1 video's base data (ASR, AST, YOLO). Caches to JSON. |
| **`run_single_vlm.py`** | Worker: Loads 1 VLM, reads base data, runs inference, saves results. |
| **`run_benchmark.py`** | Orchestrator: Automatically loops all videos and all models using the workers above. |
| **`test_<vlm>.py`** | Implementation: Model-specific loading and prompting logic. |

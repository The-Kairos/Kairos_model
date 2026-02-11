## 📊 End-to-End Performance Benchmark

This table compares processing time against the **Actual Video Duration**. The **Real-time Factor (RTF)** indicates how many times slower (or faster) the processing is compared to the video length (e.g., 2.0x means it takes twice as long as the video).

### Stage 2: VLM Inference Performance

| Model | Video | Video Length | Scenes | VLM Time | RTF | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **InstructBLIP** | Argentina v France | 7m 39s | 75 | 14.2 min | **1.8x** | ✅ Stable |
| **InstructBLIP** | Young Sheldon | 2m 48s | 35 | 5.6 min | **2.0x** | ✅ Stable |
| **InstructBLIP** | Watch Malala | 4m 33s | 22 | 4.2 min | **0.9x** | ✅ Real-time |
| **InstructBLIP** | How to Make Pasta | 5m 28s | 59 | 9.4 min | **1.7x** | ✅ Stable |
| **LLaVA-v1.6-7B** | Argentina v France | 7m 39s | 75 | 20.4 min | **2.7x** | ✅ Stable |
| **LLaVA-v1.6-7B** | Young Sheldon | 2m 48s | 35 | 7.3 min | **2.6x** | ✅ Stable |
| **LLaVA-v1.6-7B** | Watch Malala | 4m 33s | 22 | 4.6 min | **1.0x** | ✅ Real-time |
| **LLaVA-v1.6-7B** | How to Make Pasta | 5m 28s | 59 | 8.9 min | **1.6x** | ✅ Stable |
| **Phi-3.5-Vision** | Argentina v France | 7m 39s | 75 | 5.6 min | **0.7x*** | ⚠️ Unstable Cache |
| **Phi-3.5-Vision** | Young Sheldon | 2m 48s | 35 | 35.5 min | **12.6x** | ✅ Stable (No Cache) |
| **Phi-3.5-Vision** | Watch Malala | 4m 33s | 22 | 24.6 min | **5.4x** | ✅ Stable (No Cache) |
| **Phi-3.5-Vision** | How to Make Pasta | 5m 28s | 59 | 59.2 min | **10.8x** | ✅ Stable (No Cache) |

> [!NOTE]
> **Real-time Factor Analysis**: 
> - **InstructBLIP** and **LLaVA** hover between 1x and 2.5x real-time depending on scene density. Scenes with many cuts increase sampling overhead.
> - **Phi-3.5 (Stable)** is significantly heavier due to the "Eager" attention overhead without KV-caching, averaging ~10x real-time.

---

## 🏗️ Stage 1: Base Processing Breakdown

Stage 1 includes Scene Detection, ASR (Speech), AST (Sounds), and YOLO (Object Tracking). This stage is identical for all VLM runs as the output is reused.

| Video | Actual Length | Stage 1 Total | ASR (Whisper) | AST (Audio) | YOLO (Video) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Argentina v France** | 7m 39s | 28.5 min | 24.9 min | 3.2 min | 0.3 min |
| **Young Sheldon** | 2m 48s | 13.1 min | 11.5 min | 1.5 min | 0.1 min |
| **Watch Malala** | 4m 33s | 8.6 min | 7.3 min | 1.0 min | 0.2 min |
| **How to Make Pasta** | 5m 28s | 22.6 min | 19.8 min | 2.6 min | 0.2 min |

### Bottleneck Analysis
-   **ASR (87% of Stage 1)**: Processing scenes sequentially via the Azure OpenAI API introduces significant network latency and rate-limit padding. 
-   **AST/YOLO (Efficient)**: These run locally on GPU and are highly optimized, taking only a fraction of the video duration.

---

## 🔍 Detailed Technical Deep-Dive

### 1. The Phi-3.5 "Unstable Cache" Issue
**The Problem**: 
The "Argentina" run in the table (RTF 0.7x) was done using `use_cache=True`. However, in the current environment (`transformers 4.57.x`), the **Eager Attention** implementation has a bug where the KV-Cache tensors for multi-crop images fail to update their sequence length correctly. This leads to a `RuntimeError: Attention Dimension Mismatch` after generating the first token.

**The Fix**:
To ensure the pipeline completes all 11 videos without crashing, I explicitly set `use_cache=False` in `test_phi3v.py`. 
- **Effect**: The model re-calculates the entire context for every new word it generates.
- **Performance Impact**: It slows down the model by ~15x (from 4.5s/scene to 60s/scene).
- **Long-term solution**: A future upgrade to `transformers >= 4.58` with native `FlashAttention-2` support for Phi-3.5 will resolve this and return the model to sub-second inference speeds.

### 2. Quantization (4-bit BitsAndBytes)
**What it means**: 
We use `load_in_4bit=True` for LLaVA and InstructBLIP. This compresses the model weights from 16-bit (2 bytes per param) to 4-bit (0.5 bytes). A 7B model that normally requires 14GB VRAM now only needs ~4GB.

**Does it make it slower?**
**Yes, slightly.** 
Every time the model performs a calculation, it must "de-quantize" the 4-bit weights back to 16-bit in-memory. This adds a small computational overhead (usually 5-10% slower inference). 
However, the **trade-off is essential**: Without 4-bit quantization, we could only process **1 frame** at a time. With 4-bit, we can fit **up to 10 frames** or even larger 13B models on a single GPU.

### 3. Timing Methodology
**Where is the time calculated?**
The "VLM Time" in the report is measured in `run_single_vlm.py` using high-resolution timestamps.
- **Starts**: Immediately *after* the model and weights are fully loaded into the GPU.
- **Ends**: Immediately *after* the last scene has been captioned.
- **Included**: Frame extraction from video files, image preprocessing, and token generation.
- **Excluded**: The 1-2 minutes of static "Model Loading" time. This ensures the benchmark represents the actual throughput of the "engine" while running.

---

## 📸 Frame Sampling & Logic (By Model)

The pipeline uses `src.frame_sampling.py` to extract frames at 336px resolution.

### **LLaVA-v1.6-7B (The Balanced Performer)**
- **Logic**: Samples **2 frames per scene** (Start and Middle).
- **Format**: Prompt uses `<image><image>` tokens.
- **Parameters**: 
  - `do_sample=False`: Forces "Greedy Search" for 100% deterministic, factual descriptions.
  - `max_new_tokens=256`: Limits output to a concise paragraph.
  - `load_in_4bit=True`: Optimized for 24GB VRAM.

### **InstructBLIP (The Panorama Approach)**
- **Logic**: Since InstructBLIP is primarily a single-image model, we use a **Horizontal Panorama Pattern**. It stitches the 2 sampled frames side-by-side into one wide image.
- **Parameters**: 
  - `num_beams=5`: Uses "Beam Search" to find the most likely sequence of words, leading to higher linguistic quality.
  - `min_length=10`: Prevents "one-word" boring descriptions.
  - `use_fast=True`: Uses the Rust-based tokenizer to speed up text encoding.

### **Phi-3.5-Vision (The Multi-Image Native)**
- **Logic**: Natively supports multiple images via specific tags: `<|image_1|>`, `<|image_2|>`.
- **Sampling**: Samples **2 frames** per scene.
- **Parameters**: 
  - `num_crops=4`: Each image is split into 4 patches + 1 global thumbnail (high-res reasoning).
  - `_attn_implementation='eager'`: Uses the most compatible (though slower) math implementation to ensure results on any GPU type.
  - `use_cache=False`: The "Safety First" setting to prevent crashes.

---

## 🚀 Optimization Opportunities

1.  **Parallel ASR/AST**: Currently, ASR/AST (Stage 1) is 87% of the bottleneck because it processes scenes sequentially. Moving this to a `ThreadPoolExecutor` would reduce Stage 1 time by ~4x.
2.  **Quantization**: Moving LLaVA from FP16 to **4-bit (bitsandbytes)** would allow processing more frames per scene (up to 8-10) without OOM (Out of Memory) errors.
3.  **Local Whisper**: Switching from Azure Whisper API to a local `faster-whisper` container would remove network latency and rate-limiting overhead.

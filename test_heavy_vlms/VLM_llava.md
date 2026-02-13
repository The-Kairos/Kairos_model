# LLaVA Version Landscape & Comprehensive VLM Analysis

## 📚 LLaVA Version History & Features

### Complete LLaVA Family Tree

| Version              | Release  | Size       | Base LLM                   | Vision Encoder | Key Features                       | HuggingFace Model ID                   |
| -------------------- | -------- | ---------- | -------------------------- | -------------- | ---------------------------------- | -------------------------------------- |
| **LLaVA-1.0**        | Apr 2023 | 7B/13B     | Vicuna-v1.1                | CLIP ViT-L/14  | First multimodal GPT-4V competitor | `liuhaotian/llava-v1-0.1-7b`           |
| **LLaVA-1.5**        | Oct 2023 | 7B/13B     | Vicuna-v1.5                | CLIP ViT-L/14  | Academic benchmark leader          | `llava-hf/llava-1.5-7b-hf`             |
| **LLaVA-NeXT (1.6)** | Jan 2024 | 7B/13B/34B | Vicuna/Mistral/Nous-Hermes | CLIP ViT-L/14  | **Multi-image, AnyRes, OCR**       | `llava-hf/llava-v1.6-vicuna-7b-hf` ⭐  |
| **LLaVA-NeXT-Video** | Apr 2024 | 7B/34B     | Vicuna/Mistral             | CLIP ViT-L/14  | Native video support (32 frames)   | `llava-hf/LLaVA-NeXT-Video-7B-hf`      |
| **LLaVA-OneVision**  | Aug 2024 | 7B/72B     | Qwen-2                     | SigLIP         | Single/Multi-image/Video unified   | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| **LLaVA-CoT**        | Nov 2024 | 7B         | Llama-3                    | CLIP ViT-L/14  | Chain-of-thought reasoning         | `lmms-lab/llava-cot-llama3-8b`         |

### 🏆 Which Version Are You Using?

**Your Code**: `llava-hf/llava-v1.6-vicuna-7b-hf` ✅

**This is LLaVA-NeXT (v1.6)** - The **best choice** for your use case!

---

## 🔍 LLaVA v1.6 (NeXT) - Deep Dive

### Core Capabilities

| Feature               | LLaVA 1.5      | **LLaVA 1.6 (You)**    | LLaVA-OneVision    |
| --------------------- | -------------- | ---------------------- | ------------------ |
| **Multi-Image**       | ❌ Single only | ✅ **Up to 4 images**  | ✅ Up to 16 images |
| **AnyRes**            | ❌ Fixed 336px | ✅ **Dynamic grid**    | ✅ Enhanced        |
| **OCR Quality**       | ⭐⭐           | ⭐⭐⭐ **Exceptional** | ⭐⭐⭐             |
| **Person Detection**  | ⭐⭐           | ⭐⭐⭐ **Reliable**    | ⭐⭐⭐             |
| **Spatial Reasoning** | ⭐⭐           | ⭐⭐⭐ **High**        | ⭐⭐⭐             |
| **Video Support**     | ❌             | ⚠️ Frame-by-frame      | ✅ Native          |
| **Context Length**    | 2048           | **4096**               | 8192               |
| **Speed (RTF)**       | ~2.5x          | **1.6-2.7x**           | ~3.5x              |

### ✅ Why LLaVA-1.6 is Perfect for You

1. **Multi-Frame Support**: Can process 2-4 frames per scene natively
2. **AnyRes Technology**: Automatically splits high-res images into grids for better detail
3. **Excellent OCR**: Can read text in videos (scoreboards, signs, UI elements)
4. **Person Recognition**: Reliably detects and describes people, clothing, actions
5. **Proven Stability**: 100% success rate in your benchmarks (1.6-2.7x RTF)

---

## 🚨 Transformers Version Mystery Solved

### Why Requirements Say 4.45 But Code Loads 4.57

```bash
# Your requirements.txt
transformers>=4.45.0  # This is a MINIMUM version constraint
```

**What `>=4.45.0` means**: "Install 4.45.0 **or any newer version**"

When you run `pip install`, it installs the **latest compatible version** available:

- You specified: `>=4.45.0` (minimum)
- PyPI latest: `4.57.1` (at time of install)
- Result: **Installed 4.57.1** ✅

**Check your actual version**:

```bash
pip show transformers | grep Version
```

### ⚠️ Critical Issue for Phi-3.5

Your requirements allow version drift, which is **dangerous** for Phi-3.5:

- ✅ Works: `transformers==4.57.1` (frozen version)
- ❌ Breaks: `transformers==5.0.0+` (dimension mismatch errors)

**Recommended Fix**:

```bash
# Pin exact versions for stability
transformers==4.57.1  # NOT >=4.45.0
torch==2.1.0
bitsandbytes==0.41.0
```

---

## 🏃 Alternative VLMs Under 10B (Better/Faster Options)

### Top Recommendations for Your Constraints

| Model                    | Size | Speed vs LLaVA-1.6 | Memory      | Multi-Frame      | Why Consider                 |
| ------------------------ | ---- | ------------------ | ----------- | ---------------- | ---------------------------- |
| **LLaVA-1.6-Mistral-7B** | 7B   | 🟢 **~20% faster** | 4GB (4-bit) | ✅ 2-4 frames    | Better instruction following |
| **MobileVLM-V2**         | 3B   | 🟢 **~2x faster**  | 2GB (4-bit) | ⚠️ 2 frames      | Designed for speed           |
| **TinyLLaVA**            | 3.1B | 🟢 **~2x faster**  | 2GB (4-bit) | ⚠️ 2 frames      | Phi-2 based, efficient       |
| **InternVL-Chat-V1.5**   | 8B   | 🔴 ~10% slower     | 5GB (4-bit) | ✅ 4-8 frames    | Better multi-image           |
| **Qwen-VL-Chat**         | 7B   | 🟡 Similar         | 4GB (4-bit) | ⚠️ 2 frames      | Strong Chinese + English     |
| **LLaVA-OneVision-7B**   | 7B   | 🔴 ~30% slower     | 4GB (4-bit) | ✅ **16 frames** | Best multi-frame             |

### 🎯 Best Alternatives for Your Use Case

#### **Option 1: LLaVA-1.6-Mistral-7B** (Recommended Upgrade)

```python
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # Drop-in replacement
```

**Pros**:

- ✅ ~20% faster than Vicuna version (1.3-2.2x RTF estimated)
- ✅ Better instruction following (Mistral-7B-Instruct base)
- ✅ Same architecture (AnyRes, multi-image, OCR)
- ✅ **Zero code changes needed**

**Cons**:

- Slightly worse conversational tone
- Same memory footprint

---

#### **Option 2: LLaVA-OneVision-7B** (Best Multi-Frame)

```python
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

model_id = "lmms-lab/llava-onevision-qwen2-7b-ov"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)
```

**Pros**:

- ✅ **Up to 16 frames per scene** (vs your current 2)
- ✅ Native video understanding
- ✅ Unified single/multi-image/video architecture
- ✅ Qwen-2 base (strong reasoning)

**Cons**:

- 🔴 ~30% slower than LLaVA-1.6
- 🔴 Newer model (less battle-tested)
- Requires minor code changes

---

#### **Option 3: MobileVLM-V2-3B** (Speed Champion)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mtgv/MobileVLM_V2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
```

**Pros**:

- ✅ **~2x faster** than LLaVA-1.6 (0.8-1.5x RTF estimated)
- ✅ Only 2GB VRAM (4-bit)
- ✅ Can run 3-4 instances simultaneously

**Cons**:

- 🔴 Lower quality captions (3B model)
- 🔴 Weaker OCR and person recognition
- Only 2 frames per scene

---

## 📊 Benchmark Comparison (Estimated)

Projected performance on **Argentina v France** (75 scenes, 7m 39s):

| Model                             | VLM Time | RTF   | Memory | Quality    | Multi-Frame |
| --------------------------------- | -------- | ----- | ------ | ---------- | ----------- |
| **LLaVA-1.6-Vicuna-7B** (Current) | 20.4 min | 2.7x  | 4GB    | ⭐⭐⭐⭐⭐ | 2 frames    |
| **LLaVA-1.6-Mistral-7B**          | ~16 min  | ~2.1x | 4GB    | ⭐⭐⭐⭐⭐ | 2 frames    |
| **LLaVA-OneVision-7B**            | ~26 min  | ~3.4x | 4GB    | ⭐⭐⭐⭐⭐ | 16 frames   |
| **MobileVLM-V2-3B**               | ~10 min  | ~1.3x | 2GB    | ⭐⭐⭐     | 2 frames    |
| **InstructBLIP** (Current)        | 14.2 min | 1.8x  | 4GB    | ⭐⭐⭐⭐   | Panorama    |

---

## 🎯 Final Recommendations

### For **Your Current Setup** (Keep):

✅ **LLaVA-1.6-Vicuna-7B** is excellent - stick with it!

- Proven stability (2.7x RTF)
- Best balance of quality/speed/memory
- Multi-frame support (2 frames)

### For **Slight Speed Boost** (Easy Upgrade):

🟢 **Switch to LLaVA-1.6-Mistral-7B**

```python
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # Just change this line
```

Expected: 20.4 min → ~16 min (20% faster)

### For **Maximum Multi-Frame Context** (Best Quality):

🔵 **Add LLaVA-OneVision-7B** as 4th VLM option

- Can process **16 frames per scene** (8x more context)
- Best for complex action sequences
- Trade 30% speed for much richer descriptions

### For **Maximum Speed** (Quality Trade-off):

🟡 **Add MobileVLM-V2-3B** as lightweight option

- 2x faster than LLaVA-1.6
- Good for rapid prototyping
- Lower quality acceptable for some use cases

---

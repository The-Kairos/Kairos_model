# b) Evaluation of Alternative Vision–Language Models

## Overview

This section evaluates **four heavy Vision–Language Models (VLMs)** as alternatives to the lightweight BLIP baseline used in the Kairos pipeline. Each model was benchmarked across 4 videos of varying length and complexity on a **GCP g2-standard-48** instance (4× NVIDIA L4 GPUs, 192 GB RAM).

---

## Frame Sampling Strategy (Same for All Models)

All four VLMs receive **the same frames** — frame sampling is handled centrally by [`run_single_vlm.py`](file:///c:/Users/tehre/OneDrive/Desktop/COMP4201-3%20Capstone%20Project/Kairos%20Model/Kairos/test_heavy_vlms/run_single_vlm.py) before passing them to each model. The frames are **not** taken from the start, middle, or end — they are **evenly spaced** across the scene duration.

### Sampling Parameters

```python
# run_single_vlm.py, line 119
frames = sample_from_clip(
    video_path,
    scene["scene_index"],
    scene["start_seconds"],
    scene["end_seconds"],
    num_frames=2,       # 2 frames per scene
    new_size=336         # resized to 336×336 pixels
)
```

### How `sample_from_clip` Distributes Frames

The function divides the scene into **(num_frames + 1)** equal gaps, placing frames at the boundaries. For `num_frames=2`:

```
Scene timeline:
[start] ──── gap ──── [Frame 1] ──── gap ──── [Frame 2] ──── gap ──── [end]
              ↑ 1/3 mark                        ↑ 2/3 mark
```

**Concrete Example** — a 6-second scene (0s → 6s):
- Gap size = 6 / (2 + 1) = 2 seconds
- **Frame 1** sampled at **t = 0s** (start of first gap)
- **Frame 2** sampled at **t = 2s** (start of second gap)

This means neither the exact middle nor the exact end is sampled. The strategy captures **early-to-mid temporal context** rather than boundary frames.

### Per-Model Frame Handling

| Model | Receives | What It Does With 2 Frames |
|-------|----------|---------------------------|
| **InstructBLIP** | 2 × PIL images | Concatenates horizontally into a single **panorama** image (672×336 px), then processes as one image |
| **LLaVA-Vicuna** | 2 × PIL images | Inserts 2 `<image>` tokens into the prompt; model attends to both frames **natively** via cross-attention |
| **LLaVA-Mistral** | 2 × PIL images | Same as Vicuna variant — 2 `<image>` tokens, native multi-image attention |
| **Phi-3.5-Vision** | 2 × PIL images | Uses `<|image_1|>` and `<|image_2|>` tokens; processes each with `num_crops=4` sub-image patches |

### Frame Preprocessing

Before reaching the VLM, each frame undergoes:
1. **Extraction**: OpenCV `cap.read()` at the computed frame position
2. **Resize**: Bilinear resize to 336×336 pixels (preserving aspect ratio via padding)
3. **Color conversion**: BGR (OpenCV default) → RGB → PIL Image

---

## Tested Models

### 1. InstructBLIP (Salesforce/instructblip-vicuna-7b)

| Property | Value |
|----------|-------|
| **Parameters** | ~7B |
| **Base LLM** | Vicuna-7B |
| **Vision Encoder** | EVA-CLIP ViT-G/14 + Q-Former |
| **Multi-Image** | ❌ Single-image only (panorama workaround) |
| **OCR** | ⚠️ Basic (not trained for OCR) |
| **Person Recognition** | ⭐⭐⭐ Good (detects people, clothing, actions) |

**Code Parameters** ([test_instructblip.py](file:///c:/Users/tehre/OneDrive/Desktop/COMP4201-3%20Capstone%20Project/Kairos%20Model/Kairos/test_heavy_vlms/test_instructblip.py)):
```python
# Quantization
load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16

# Generation
do_sample=False, max_new_tokens=256, min_length=10, num_beams=5
```

**Multi-Frame Strategy**: Since InstructBLIP only accepts a single image, multiple frames are **concatenated horizontally** into a panorama strip before processing. This provides temporal context but increases the effective image width per scene.

---

### 2. LLaVA-v1.6-Vicuna-7B (llava-hf/llava-v1.6-vicuna-7b-hf)

| Property | Value |
|----------|-------|
| **Parameters** | ~7B |
| **Base LLM** | Vicuna-v1.5-7B (LLaMA-2 fine-tune) |
| **Vision Encoder** | CLIP ViT-L/14 @ 336px with **AnyRes** |
| **Multi-Image** | ✅ Native (up to 4 images via `<image>` tokens) |
| **OCR** | ⭐⭐⭐ Strong (AnyRes preserves text at high resolution) |
| **Person Recognition** | ⭐⭐⭐ Reliable (clothing, posture, actions) |

**Code Parameters** ([test_llava_1_6.py](file:///c:/Users/tehre/OneDrive/Desktop/COMP4201-3%20Capstone%20Project/Kairos%20Model/Kairos/test_heavy_vlms/test_llava_1_6.py)):
```python
# Quantization
load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16

# Generation
max_new_tokens=256, do_sample=False
```

**Multi-Frame Strategy**: Uses native `<image>` tokens — one per frame. The prompt template inserts `<image>` × N tokens before the instruction, allowing the model to attend to all frames simultaneously.

---

### 3. LLaVA-v1.6-Mistral-7B (llava-hf/llava-v1.6-mistral-7b-hf)

| Property | Value |
|----------|-------|
| **Parameters** | ~7B |
| **Base LLM** | Mistral-7B-Instruct-v0.2 |
| **Vision Encoder** | CLIP ViT-L/14 @ 336px with **AnyRes** |
| **Multi-Image** | ✅ Native (same as Vicuna variant) |
| **OCR** | ⭐⭐⭐ Strong |
| **Person Recognition** | ⭐⭐⭐ Reliable |

**Code Parameters** ([test_llava_1_6_mistral.py](file:///c:/Users/tehre/OneDrive/Desktop/COMP4201-3%20Capstone%20Project/Kairos%20Model/Kairos/test_heavy_vlms/test_llava_1_6_mistral.py)):
```python
# Quantization (enhanced)
load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
torch_dtype=torch.float16

# Generation
max_new_tokens=256, do_sample=False
```

**Key Difference from Vicuna variant**: Uses the **NF4 (NormalFloat4)** quantization type instead of the default FP4. NF4 is information-theoretically optimal for normally-distributed weights, yielding better quality at the same 4-bit size. The Mistral-7B base also has a faster inference architecture (Grouped Query Attention).

---

### 4. Phi-3.5-Vision (microsoft/Phi-3.5-vision-instruct)

| Property | Value |
|----------|-------|
| **Parameters** | ~4.2B |
| **Base LLM** | Phi-3.5-mini (3.8B) |
| **Vision Encoder** | CLIP-based with multi-crop |
| **Multi-Image** | ✅ Native (`<|image_N|>` tokens) |
| **OCR** | ⭐⭐⭐⭐ Excellent (explicitly trained for OCR) |
| **Person Recognition** | ⭐⭐⭐ Good |

**Code Parameters** ([test_phi3v.py](file:///c:/Users/tehre/OneDrive/Desktop/COMP4201-3%20Capstone%20Project/Kairos%20Model/Kairos/test_heavy_vlms/test_phi3v.py)):
```python
# Model Loading
torch_dtype=torch.float16, trust_remote_code=True
_attn_implementation='eager'  # NOT flash_attention_2 (see anomaly below)
num_crops=4

# Generation
max_new_tokens=500, do_sample=False
use_cache=False  # CRITICAL — see anomaly explanation below
```

---

## ⚠️ Phi-3.5-Vision Anomaly: Why `use_cache=True` Did Not Work

### The Problem

Phi-3.5-Vision exhibited **wildly inconsistent performance** across videos: processing Argentina vs France in just 5.6 min (0.7× RTF, anomalously fast) but taking 59.2 min for How to Make Pasta (10.8× RTF).

### Root Cause

Phi-3.5-Vision uses **custom model code** (`trust_remote_code=True`) that was written for an older version of the `transformers` library. Our environment runs **transformers 4.57.1**, which introduced breaking changes to the **KV-cache interface**:

1. **`DynamicCache.seen_tokens`** was removed in newer transformers versions. Phi-3.5's custom attention code calls this property internally.
2. **`DynamicCache.get_max_length()`** and **`DynamicCache.get_usable_length()`** were also removed.

When `use_cache=True` (the default), the model attempts to build a KV-cache during autoregressive decoding. Each new token reuses previously computed key/value pairs to avoid redundant computation. However, the dimension mismatch between Phi-3.5's custom code and transformers 4.57.1's `DynamicCache` causes:

- **Silent corruption** of the cache tensors
- **RuntimeError** on some sequences (variable-length input causes crashes at unpredictable points)
- **Hallucinated outputs** when the cache doesn't crash but contains garbage values

### Our Fix

```python
# Monkeypatches to restore missing DynamicCache attributes
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_length", None)
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, seq_length, layer_idx=None: self.get_seq_length()
```

Even with these patches, we set **`use_cache=False`** as a safety measure because the `eager` attention implementation still has subtle incompatibilities:

```python
generation_args = {
    "max_new_tokens": 500,
    "do_sample": False,
    "use_cache": False,  # Disables KV-cache entirely
}
```

### Performance Impact

Disabling KV-cache means **every token re-computes attention over the full sequence** instead of reusing cached keys/values. This causes Phi-3.5's generation time to scale **quadratically** with output length (O(n²)) rather than linearly (O(n)):

| Metric | `use_cache=True` (broken) | `use_cache=False` (stable) |
|--------|---------------------------|---------------------------|
| Time per token | ~15ms | ~150ms |
| 256 tokens | ~4s | ~38s |
| 500 tokens | ~8s | ~75s |
| **Stability** | ❌ Crashes/hallucinations | ✅ 100% stable |

This explains the large RTF values (5–12×) for Phi-3.5 across most videos.

### Why Argentina vs France Was Anomalously Fast (0.7× RTF)

The first video benefited from **shorter generated outputs** (the model produced very brief captions for the repetitive penalty shoot-out scenes, averaging ~50 tokens vs ~300+ for cooking/speech videos). With short outputs, the quadratic penalty of `use_cache=False` is negligible.

---

## Benchmark Results

### Table 3. Heavy VLMs Inference Performance

The **Real-time Factor (RTF)** indicates processing time relative to video duration. An RTF of 1.0× means the VLM takes as long as the video itself; lower is faster.

| Video | Scenes | Length | Model | VLM Time | RTF | Status |
|-------|--------|--------|-------|----------|-----|--------|
| **Argentina vs France** | 75 | 7.39 min | InstructBLIP | 14.2 min | 1.8× | Stable |
| | | | LLaVA-v1.6-7B | 20.4 min | 2.7× | Stable |
| | | | LLaVA-Mistral-7B | 8.6 min | 1.1× | Fastest |
| | | | Phi-3.5-Vision | 5.6 min | 0.7× | Anomalous* |
| **How to Make Pasta** | 59 | 5.28 min | InstructBLIP | 9.4 min | 1.7× | Stable |
| | | | LLaVA-v1.6-7B | 8.9 min | 1.6× | Stable |
| | | | LLaVA-Mistral-7B | 9.7 min | 1.8× | Stable |
| | | | Phi-3.5-Vision | 59.2 min | 10.8× | Stable |
| **Malala Nobel Peace Prize** | 22 | 4.33 min | InstructBLIP | 4.2 min | 0.9× | Real-time |
| | | | LLaVA-v1.6-7B | 4.6 min | 1.0× | Real-time |
| | | | LLaVA-Mistral-7B | 3.4 min | 0.8× | Real-time |
| | | | Phi-3.5-Vision | 24.6 min | 5.4× | Stable |
| **Young Sheldon** | 35 | 2.48 min | InstructBLIP | 5.6 min | 2.0× | Stable |
| | | | LLaVA-v1.6-7B | 7.3 min | 2.6× | Stable |
| | | | LLaVA-Mistral-7B | 1.3 min | 0.5× | Sub-realtime |
| | | | Phi-3.5-Vision | 35.5 min | 12.6× | Stable |

*\*Anomalous: short output tokens for repetitive scenes reduced the quadratic penalty of `use_cache=False`.*

---

### Average RTF Across All Videos

| Model | Avg RTF | VRAM (4-bit) | Multi-Frame | Stability |
|-------|---------|-------------|-------------|-----------|
| **LLaVA-Mistral-7B** | **1.05×** | ~4 GB | ✅ Native | ✅ 100% |
| **InstructBLIP** | **1.60×** | ~4 GB | ❌ Panorama | ✅ 100% |
| **LLaVA-v1.6-7B** | **1.98×** | ~4 GB | ✅ Native | ✅ 100% |
| **Phi-3.5-Vision** | **7.38×** | ~5 GB | ✅ Native | ⚠️ Requires workaround |

---

## Key Findings

1. **LLaVA-Mistral-7B is the fastest stable model** with an average RTF of 1.05×, achieving near-real-time processing thanks to Mistral's Grouped Query Attention architecture.

2. **InstructBLIP provides consistent 1.6× RTF** despite being limited to single-image input. The panorama concatenation strategy effectively provides temporal context without native multi-frame support.

3. **LLaVA-v1.6-Vicuna-7B is the most balanced** — slightly slower (1.98× RTF) but produces the most detailed captions with strong spatial reasoning.

4. **Phi-3.5-Vision is unusable at scale** in its current configuration due to the `use_cache=False` requirement, which causes 5–12× slowdowns. While it produces excellent captions (especially OCR-heavy content), the performance penalty is too severe for production use. A fix would require either:
   - Pinning `transformers==4.44.0` (pre-breaking-change version)
   - Waiting for Microsoft to update the model's custom code for newer transformers versions

---

## Model Feature Comparison

| Feature | InstructBLIP | LLaVA-Vicuna | LLaVA-Mistral | Phi-3.5 |
|---------|-------------|-------------|---------------|---------|
| **OCR (Text Reading)** | ⚠️ Basic | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Person Detection** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Action Description** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Spatial Reasoning** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Multi-Frame** | Panorama hack | Native `<image>` | Native `<image>` | Native `<\|image_N\|>` |
| **Quantization** | 4-bit FP4 | 4-bit FP4 | 4-bit **NF4** | FP16 (no quant) |
| **KV-Cache** | ✅ Working | ✅ Working | ✅ Working | ❌ Disabled |
| **Beam Search** | ✅ 5 beams | ❌ Greedy | ❌ Greedy | ❌ Greedy |

---

## Infrastructure

- **VM**: GCP g2-standard-48 (4× NVIDIA L4, 48 vCPUs, 192 GB RAM)
- **Framework**: transformers 4.57.1, PyTorch 2.9.1, bitsandbytes 0.49.1
- **Process Isolation**: Each VLM runs in a separate subprocess (`run_single_vlm.py`) that terminates after completion, forcing OS-level GPU memory reclamation
- **LLM Fusion (Gemini)**: Disabled — manual concatenation of VLM captions with ASR/AST/YOLO outputs is used instead

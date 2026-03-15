# Kairos Configuration Guide

> Complete reference for `PipelineConfig` — every tunable parameter, presets, custom configs, and environment variables.

---

## Table of Contents

- [Overview](#overview)
- [Configuration Fields by Category](#configuration-fields-by-category)
  - [Scene Detection](#scene-detection)
  - [Frame Sampling](#frame-sampling)
  - [BLIP Captioning](#blip-captioning)
  - [YOLO Detection](#yolo-detection)
  - [Audio](#audio)
  - [LLM](#llm)
  - [RAG](#rag)
  - [Parallelism](#parallelism)
  - [Paths](#paths)
- [Presets](#presets)
  - [Presets Comparison Table](#presets-comparison-table)
- [Custom Configurations](#custom-configurations)
- [Environment Variables](#environment-variables)

---

## Overview

All tunable parameters for the Kairos video-processing pipeline are centralised in a single `PipelineConfig` dataclass. Configuration values are validated eagerly in `__post_init__`, raising `KairosConfigError` for any out-of-range values.

```python
from kairos import PipelineConfig

# Use defaults
cfg = PipelineConfig.default()

# Use a preset
cfg = PipelineConfig.fast()

# Override specific fields
cfg = PipelineConfig(pyscene_threshold=20.0, frames_per_scene=5)
```

---

## Configuration Fields by Category

### Scene Detection

Controls how the video is segmented into scenes using PySceneDetect's `ContentDetector`.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `pyscene_threshold` | `float` | `27.0` | `> 0` | Sensitivity for the content-change detector. Lower values detect more scene cuts; higher values detect fewer (only dramatic transitions). |
| `pyscene_shortest` | `float` | `2.0` | `>= 0` | Minimum scene length in seconds. Scenes shorter than this are merged with their neighbours. Set to `0` to allow arbitrarily short scenes. |

### Frame Sampling

Controls how many frames are sampled from each scene for visual analysis.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `frames_per_scene` | `int` | `3` | `>= 1` | Number of evenly-spaced frames to sample from each scene for BLIP captioning. More frames give richer descriptions but increase processing time. |
| `frame_resolution` | `int` | `320` | `>= 1` | Target resolution in pixels for the longest side of sampled frames. Frames are resized preserving aspect ratio. Smaller values speed up captioning and YOLO inference. |

### BLIP Captioning

Controls the BLIP image-captioning model's text generation behaviour. All BLIP fields are collected into a single dict via the `blip_params` property and forwarded as `**kwargs` to `model.generate()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `blip_start_prompt` | `str` | `"a video frame of"` | Starting text prompt conditioning the BLIP decoder. Changing this can steer the style of generated captions. |
| `blip_caption_len` | `int` | `50` | Maximum caption length in tokens. Must be `>= 1`. |
| `blip_min_length` | `int` | `15` | Minimum caption length in tokens. Must be `>= 1`. Prevents trivially short captions. |
| `blip_num_beams` | `int` | `1` | Number of beams for beam search. `1` = greedy/sampling; higher values increase quality but reduce speed. |
| `blip_do_sample` | `bool` | `True` | Whether to use sampling during generation. When `False` and `num_beams=1`, decoding is purely greedy. |
| `blip_top_p` | `float` | `0.85` | Nucleus sampling probability threshold. Only tokens with cumulative probability ≤ `top_p` are considered. |
| `blip_temperature` | `float` | `0.65` | Sampling temperature. Lower values produce more deterministic output; higher values increase creativity. |
| `blip_length_penalty` | `float` | `1.0` | Length penalty for beam search. Values `> 1.0` favour longer sequences; values `< 1.0` favour shorter ones. |
| `blip_no_repeat_ngram_size` | `int` | `3` | N-gram size to prevent repetition. No n-gram of this size will appear more than once in the output. |
| `blip_repetition_penalty` | `float` | `1.1` | Repetition penalty factor. Values `> 1.0` discourage the model from repeating tokens. |

### YOLO Detection

Controls YOLOv8 object detection and action sampling.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `yolo_model_path` | `str` | `"models/yolov8s.pt"` | — | Path to the YOLOv8 model weights file. Can be any Ultralytics-compatible weight file (e.g. `yolov8n.pt`, `yolov8m.pt`). |
| `yolo_action_fps` | `float` | `4.0` | `> 0` | Frames per second at which to sample frames for YOLO tracking. Higher values capture finer-grained motion but increase compute. |
| `yolo_conf_thres` | `float` | `0.8` | `[0, 1]` | YOLO confidence threshold. Only detections with confidence ≥ this value are kept. |
| `yolo_iou_thres` | `float` | `0.5` | `[0, 1]` | IoU threshold for Non-Maximum Suppression (NMS). Controls how much bounding-box overlap is tolerated before suppression. |

### Audio

Controls audio extraction, speech recognition (ASR), and sound classification (AST).

| Field | Type | Default | Description |
|---|---|---|---|
| `ast_target_sr` | `int` | `16000` | Target sample rate in Hz for the MIT AST audio classification model. Most AST models expect 16 kHz. |
| `asr_model_size` | `str` | `"medium"` | Whisper model size for speech recognition. Options: `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`. Larger models are more accurate but slower. |
| `asr_use_vad` | `bool` | `True` | Whether to use Voice Activity Detection (Silero VAD) to pre-filter speech segments before transcription. Improves accuracy by reducing noise. |
| `asr_target_sr` | `int` | `16000` | Target sample rate in Hz for Whisper ASR audio input. |

### LLM

Controls LLM-based scene description, narrative summarisation, and synopsis generation.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `llm_scene_history` | `int` | `5` | `>= 0` | Number of previous scene descriptions included as context when generating the current scene's description. Larger values give the LLM more temporal context. |
| `llm_chunk_len` | `int` | `20000` | `> 0` | Maximum character length per narrative chunk during the map-reduce summarization stage. |
| `llm_summary_len` | `int` | `50000` | `> 0` | Maximum character length for the full narrative summary. Triggers additional reduce/consistency passes when exceeded. |
| `llm_cooldown_sec` | `float` | `0.0` | `>= 0` | Cooldown in seconds between consecutive LLM API calls. Useful for respecting rate limits. |

### RAG

Controls Retrieval-Augmented Generation for question answering.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `rag_top_k_context` | `int` | `10` | `>= 1` | Number of top-matching context passages retrieved from the embedding index when answering a question. |

### Parallelism

Controls concurrent execution.

| Field | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `llm_max_workers` | `int` | `4` | `>= 1` | Maximum number of parallel LLM workers used during scene description generation. Higher values speed up processing but increase API load. |

### Paths

Controls filesystem locations for data, prompts, and logs.

| Field | Type | Default | Description |
|---|---|---|---|
| `data_dir` | `str` | `"data"` | Root data directory for processed videos, checkpoints, and video files. |
| `prompts_dir` | `str` | `""` (auto-resolved) | Directory containing prompt templates. When empty (default), resolved automatically via `importlib.resources` to the `kairos/prompts/` package directory. |
| `logs_dir` | `str` | `"logs"` | Directory where pipeline run logs are saved. |

---

## Presets

`PipelineConfig` ships with four preset class methods that return pre-configured instances tailored for common use cases.

### `PipelineConfig.default()`

Returns the baseline configuration with all default values. Suitable for general-purpose video analysis.

### `PipelineConfig.fast()`

Optimised for speed over accuracy. Uses a higher scene-detection threshold, fewer sampled frames, much larger LLM chunks (to minimise round-trips), and more parallel workers.

### `PipelineConfig.motion_sensitive()`

Tuned for high-motion content (sports, action sequences). Uses a lower scene-detection threshold to catch rapid cuts, more frames per scene, shorter minimum scene length, and a higher YOLO action sampling rate.

### `PipelineConfig.static_video()`

Tuned for slow-moving or static content (lectures, surveillance, interviews). Uses a very low scene-detection threshold, fewer frames per scene, and a reduced YOLO action sampling rate.

### Presets Comparison Table

| Parameter | `default()` | `fast()` | `motion_sensitive()` | `static_video()` |
|---|---|---|---|---|
| `pyscene_threshold` | `27.0` | **`40`** | **`15`** | **`3`** |
| `pyscene_shortest` | `2.0` | `2.0` | **`0.5`** | `2.0` |
| `frames_per_scene` | `3` | **`1`** | **`5`** | **`1`** |
| `frame_resolution` | `320` | `320` | `320` | `320` |
| `blip_start_prompt` | `"a video frame of"` | `"a video frame of"` | `"a video frame of"` | `"a video frame of"` |
| `blip_caption_len` | `50` | `50` | `50` | `50` |
| `blip_min_length` | `15` | `15` | `15` | `15` |
| `blip_num_beams` | `1` | `1` | `1` | `1` |
| `blip_do_sample` | `True` | `True` | `True` | `True` |
| `blip_top_p` | `0.85` | `0.85` | `0.85` | `0.85` |
| `blip_temperature` | `0.65` | `0.65` | `0.65` | `0.65` |
| `blip_length_penalty` | `1.0` | `1.0` | `1.0` | `1.0` |
| `blip_no_repeat_ngram_size` | `3` | `3` | `3` | `3` |
| `blip_repetition_penalty` | `1.1` | `1.1` | `1.1` | `1.1` |
| `yolo_model_path` | `"models/yolov8s.pt"` | `"models/yolov8s.pt"` | `"models/yolov8s.pt"` | `"models/yolov8s.pt"` |
| `yolo_action_fps` | `4.0` | `4.0` | **`8`** | **`0.5`** |
| `yolo_conf_thres` | `0.8` | `0.8` | `0.8` | `0.8` |
| `yolo_iou_thres` | `0.5` | `0.5` | `0.5` | `0.5` |
| `ast_target_sr` | `16000` | `16000` | `16000` | `16000` |
| `asr_model_size` | `"medium"` | `"medium"` | `"medium"` | `"medium"` |
| `asr_use_vad` | `True` | `True` | `True` | `True` |
| `asr_target_sr` | `16000` | `16000` | `16000` | `16000` |
| `llm_scene_history` | `5` | `5` | `5` | `5` |
| `llm_chunk_len` | `20000` | **`500000`** | `20000` | `20000` |
| `llm_summary_len` | `50000` | **`500000`** | `50000` | `50000` |
| `llm_cooldown_sec` | `0.0` | `0.0` | `0.0` | `0.0` |
| `rag_top_k_context` | `10` | `10` | `10` | `10` |
| `llm_max_workers` | `4` | **`8`** | `4` | `4` |
| `data_dir` | `"data"` | `"data"` | `"data"` | `"data"` |
| `logs_dir` | `"logs"` | `"logs"` | `"logs"` | `"logs"` |

> **Bold** values indicate differences from the default preset.

---

## Custom Configurations

### Creating from scratch

```python
from kairos import PipelineConfig

cfg = PipelineConfig(
    pyscene_threshold=20.0,
    pyscene_shortest=1.0,
    frames_per_scene=5,
    frame_resolution=640,
    yolo_action_fps=8.0,
    yolo_conf_thres=0.6,
    asr_model_size="large",
    llm_scene_history=10,
    llm_max_workers=8,
)
```

### Modifying a preset

Since `PipelineConfig` is a `@dataclass`, use `dataclasses.replace` to create a modified copy:

```python
from dataclasses import replace
from kairos import PipelineConfig

base = PipelineConfig.fast()
cfg = replace(base, frames_per_scene=3, yolo_conf_thres=0.6)
```

### Serialisation

```python
# To dict (e.g. for logging or checkpoint)
params = cfg.to_dict()

# Reconstruct from dict
cfg2 = PipelineConfig(**params)
```

### Validation

`PipelineConfig.__post_init__` runs automatically on construction and validates:

| Constraint | Error |
|---|---|
| `pyscene_threshold > 0` | `KairosConfigError` |
| `pyscene_shortest >= 0` | `KairosConfigError` |
| `frames_per_scene >= 1` | `KairosConfigError` |
| `frame_resolution >= 1` | `KairosConfigError` |
| `blip_caption_len >= 1` | `KairosConfigError` |
| `blip_min_length >= 1` | `KairosConfigError` |
| `yolo_conf_thres in [0, 1]` | `KairosConfigError` |
| `llm_max_workers >= 1` | `KairosConfigError` |
| `rag_top_k_context >= 1` | `KairosConfigError` |
| `llm_cooldown_sec >= 0` | `KairosConfigError` |

---

## Environment Variables

Kairos reads several environment variables to configure LLM backends and API connections. These are used by `build_llm_client()` in `kairos.llm.client`.

### Backend Selection

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `"openai"` | LLM backend to use. One of `"openai"`, `"gemini"`, or `"claude"`. Overridden by the `llm` argument to `build_llm_client()`. |

### OpenAI / Azure OpenAI

| Variable | Default | Description |
|---|---|---|
| `OPENAI_KEY` | — | API key for the OpenAI (or Azure OpenAI) endpoint. |
| `OPENAI_ENDPOINT` | — | Base URL for the OpenAI API (e.g. Azure endpoint). |
| `OPENAI_MODEL` | `"gpt-4o"` | Model name or deployment ID. Falls back to `OPENAI_DEPLOYMENT`. |
| `OPENAI_DEPLOYMENT` | `"gpt-4o"` | Azure deployment name (used when `OPENAI_MODEL` is unset). |

### Gemini (Vertex AI)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_PROJECT` | `"prj-udst-prod-oussama-1"` | GCP project ID for Vertex AI. |
| `GEMINI_LOCATION` | `"us-central1"` | GCP region for the Vertex AI endpoint. |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model identifier. |
| `GEMINI_EMBEDDING_MODEL` | `"gemini-embedding-001"` | Model used for RAG embedding generation. |
| `GEMINI_RAG_MODEL` | `"gemini-2.5-pro"` | Model used for RAG answer generation. |


> **Authentication:** Gemini uses Google Cloud Vertex AI and authenticates via [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) — typically a GCP service account attached to the VM. It does **not** use API keys. On non-GCP machines, set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON key file.
### Claude (Vertex AI)

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_LOCATION` | `"us-east5"` | GCP region for the Claude Vertex AI endpoint. |
| `CLAUDE_PROJECT` | Falls back to `GEMINI_PROJECT` | GCP project ID for the Claude endpoint. |
| `CLAUDE_MODEL` | `"claude-sonnet-4-6"` | Claude model identifier. |


> **Authentication:** Claude is accessed through Google Cloud Vertex AI (via the `anthropic[vertex]` SDK) and uses the same [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) as Gemini. It does **not** use Anthropic API keys.
### Whisper API (Azure OpenAI)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_API_KEY` | — | API key for the Azure OpenAI Whisper endpoint. |
| `WHISPER_API_ENDPOINT` | — | Full endpoint URL for the Azure OpenAI Whisper deployment. |
| `WHISPER_API_VERSION` | `"2024-12-01-preview"` | API version string. |
| `WHISPER_API_DEPLOYMENT` | — | Azure deployment name for the Whisper model. |

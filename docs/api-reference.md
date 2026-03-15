# Kairos API Reference

> Auto-generated reference for all public classes, methods, and functions in the Kairos video-understanding pipeline.

---

## Table of Contents

- [kairos (top-level)](#kairos-top-level)
- [kairos.config](#kairosconfig)
- [kairos.core.exceptions](#kairoscoreexceptions)
- [kairos.core.models](#kairoscoremodels)
- [kairos.core.scene](#kairoscorescene)
- [kairos.core.pipeline](#kairoscorepipeline)
- [kairos.core.checkpoint](#kairoscorecheckpoint)
- [kairos.core.logging](#kairoscorelogging)
- [kairos.core.timing](#kairoscoretiming)
- [kairos.core.redo](#kairoscoreredo)
- [kairos.core.utils](#kairoscoreutils)
- [kairos.llm.client](#kairosllmclient)
- [kairos.llm.rag](#kairosllmrag)
- [kairos.llm.scene_description](#kairosllmscene_description)
- [kairos.llm.synopsis](#kairosllmsynopsis)
- [kairos.audio.classifier](#kairosaudioclassifier)
- [kairos.audio.extraction](#kairosaudioextraction)
- [kairos.audio.language](#kairosaudiolanguage)
- [kairos.audio.prescan](#kairosaudioprescan)
- [kairos.audio.rms](#kairosaudioRMS)
- [kairos.audio.spectral](#kairosaudiospectral)
- [kairos.audio.text_filter](#kairosaudiotext_filter)
- [kairos.audio.transcription](#kairosaudiotranscription)
- [kairos.audio.vad](#kairosaudiovad)
- [kairos.audio.whisper_api](#kairosaudiowhisper_api)
- [kairos.video.scene_detection](#kairosvideoscene_detection)
- [kairos.video.frame_sampling](#kairosvideoframe_sampling)
- [kairos.video.frame_captioning](#kairosvideoframe_captioning)
- [kairos.video.object_detection](#kairosvideoobject_detection)
- [kairos.video.tracking](#kairosvideotracking)
- [kairos.video.track_summary](#kairosvideotrack_summary)
- [kairos.video.spatial](#kairosvideospatial)
- [kairos.video.debug_draw](#kairosvideodebug_draw)
- [kairos.video.yolo_inference](#kairosvideoyolo_inference)

---

## `kairos` (top-level)

Re-exports of the most commonly used symbols.

| Symbol | Type | Description |
|---|---|---|
| `__version__` | `str` | Package version string (currently `"0.1.0"`). |
| `PipelineConfig` | class | All tunable parameters for the Kairos pipeline (re-exported from `kairos.config`). |
| `KairosError` | class | Base exception for all Kairos errors (re-exported from `kairos.core.exceptions`). |
| `ModelRegistry` | class | Process-wide singleton for lazy-loading ML models (re-exported from `kairos.core.models`). |
| `Scene` | class | Typed dataclass representing a single detected scene (re-exported from `kairos.core.scene`). |

---

## `kairos.config`

Pipeline configuration as a dataclass with presets.

### `class PipelineConfig`

All tunable parameters for the Kairos video processing pipeline. See [Configuration Guide](configuration.md) for full details.

#### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `pyscene_threshold` | `float` | `27.0` | Sensitivity for PySceneDetect content detector. |
| `pyscene_shortest` | `float` | `2.0` | Minimum scene length in seconds. |
| `frames_per_scene` | `int` | `3` | Number of frames to sample per scene. |
| `frame_resolution` | `int` | `320` | Target resolution (longest side) for sampled frames. |
| `blip_start_prompt` | `str` | `"a video frame of"` | Starting text prompt for BLIP captioning. |
| `blip_caption_len` | `int` | `50` | Maximum caption length in tokens. |
| `blip_min_length` | `int` | `15` | Minimum caption length in tokens. |
| `blip_num_beams` | `int` | `1` | Number of beams for beam search. |
| `blip_do_sample` | `bool` | `True` | Whether to use sampling during generation. |
| `blip_top_p` | `float` | `0.85` | Nucleus sampling probability threshold. |
| `blip_temperature` | `float` | `0.65` | Sampling temperature. |
| `blip_length_penalty` | `float` | `1.0` | Length penalty for beam search. |
| `blip_no_repeat_ngram_size` | `int` | `3` | N-gram size to prevent repetition. |
| `blip_repetition_penalty` | `float` | `1.1` | Repetition penalty factor. |
| `yolo_model_path` | `str` | `"models/yolov8s.pt"` | Path to the YOLOv8 model weights. |
| `yolo_action_fps` | `float` | `4.0` | Frames per second for YOLO action sampling. |
| `yolo_conf_thres` | `float` | `0.8` | YOLO confidence threshold. |
| `yolo_iou_thres` | `float` | `0.5` | YOLO IoU threshold for NMS. |
| `ast_target_sr` | `int` | `16000` | Target sample rate for AST audio classification. |
| `asr_model_size` | `str` | `"medium"` | Whisper model size for speech recognition. |
| `asr_use_vad` | `bool` | `True` | Whether to use Voice Activity Detection. |
| `asr_target_sr` | `int` | `16000` | Target sample rate for ASR. |
| `llm_scene_history` | `int` | `5` | Number of previous scenes to include as context. |
| `llm_chunk_len` | `int` | `20000` | Maximum character length per narrative chunk. |
| `llm_summary_len` | `int` | `50000` | Maximum character length for the full summary. |
| `llm_cooldown_sec` | `float` | `0.0` | Cooldown in seconds between LLM API calls. |
| `rag_top_k_context` | `int` | `10` | Number of top contexts to retrieve for RAG. |
| `llm_max_workers` | `int` | `4` | Maximum number of parallel LLM workers. |
| `data_dir` | `str` | `"data"` | Root data directory. |
| `prompts_dir` | `str` | `""` | Directory containing prompt templates (resolved via `importlib.resources` if empty). |
| `logs_dir` | `str` | `"logs"` | Directory for pipeline logs. |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `__post_init__` | `(self) -> None` | Validate configuration values eagerly; raises `KairosConfigError` on invalid values. |
| `default` | `@classmethod (cls) -> PipelineConfig` | Create a configuration with all default values. |
| `fast` | `@classmethod (cls) -> PipelineConfig` | Create a configuration optimised for speed over accuracy. |
| `motion_sensitive` | `@classmethod (cls) -> PipelineConfig` | Create a configuration tuned for high-motion content. |
| `static_video` | `@classmethod (cls) -> PipelineConfig` | Create a configuration tuned for slow-moving or static content. |
| `blip_params` | `@property (self) -> dict[str, Any]` | Collect all BLIP generation fields into a dict for `**kwargs` forwarding. |
| `to_dict` | `(self) -> dict[str, Any]` | Serialize all fields to a plain dictionary. |

---

## `kairos.core.exceptions`

Kairos exception hierarchy. All custom exceptions inherit from `KairosError`.

| Exception | Base | Description |
|---|---|---|
| `KairosError` | `Exception` | Base exception for all Kairos errors. |
| `KairosConfigError` | `KairosError` | Invalid or inconsistent pipeline configuration. |
| `KairosModelError` | `KairosError` | Failure loading or running an ML model (BLIP, YOLO, AST, Whisper). |
| `KairosLLMError` | `KairosError` | Failure calling an external LLM API (OpenAI, Gemini, Claude). |
| `KairosIOError` | `KairosError` | File / network I/O error (missing video, bad checkpoint, etc.). |
| `KairosRAGError` | `KairosError` | Failure during RAG embedding, retrieval, or answer generation. |

---

## `kairos.core.models`

Thread-safe model registry with lazy loading and caching.

### `class ModelRegistry`

Process-wide singleton that lazy-loads and caches ML models thread-safely.

| Method | Signature | Description |
|---|---|---|
| `get` | `@classmethod (cls) -> ModelRegistry` | Return the process-wide singleton (create on first call). |
| `blip` | `(self, device: str \| None = None) -> tuple` | Return `(model, processor)` for BLIP image captioning. |
| `yolo` | `(self, model_path: str = "models/yolov8s.pt") -> Any` | Return a loaded YOLO model. |
| `ast` | `(self, device: str \| None = None) -> tuple` | Return `(feature_extractor, model)` for MIT AST audio classification. |
| `whisper` | `(self, model_size: str = "medium", device: str \| None = None) -> Any` | Return a loaded local Whisper model. |
| `silero_vad` | `(self) -> tuple` | Return `(silero_model, get_speech_timestamps_fn)`. |
| `release` | `(self, name: str) -> None` | Remove a cached model and free its memory. |
| `release_all` | `(self) -> None` | Release every cached model. |
| `is_loaded` | `(self, name: str) -> bool` | Check whether a model is currently cached. |
| `loaded_models` | `(self) -> list[str]` | Return a list of currently cached model names. |

---

## `kairos.core.scene`

Typed Scene dataclass replacing raw `dict` scene representations.

### `class Scene`

A single detected scene with all enriched pipeline data.

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `scene_index` | `int` | `0` | Zero-based scene index. |
| `start_seconds` | `float` | `0.0` | Start time in seconds. |
| `end_seconds` | `float` | `0.0` | End time in seconds. |
| `duration_seconds` | `float` | `0.0` | Duration in seconds. |
| `start_timecode` | `str` | `"00:00:00.000"` | Start timecode string. |
| `end_timecode` | `str` | `"00:00:00.000"` | End timecode string. |
| `clip_path` | `str \| None` | `None` | Filesystem path to the extracted clip. |
| `frames` | `list[np.ndarray]` | `[]` | Sampled BGR frames (transient, not serialized). |
| `frame_paths` | `list[str] \| None` | `None` | Paths to saved frame images. |
| `yolo_frames` | `list[np.ndarray]` | `[]` | Frames sampled at YOLO FPS (transient). |
| `yolo_frame_paths` | `list[str] \| None` | `None` | Paths to saved YOLO frame images. |
| `frame_captions` | `list[str]` | `[]` | BLIP-generated captions per frame. |
| `yolo_detections` | `Any` | `[]` | YOLO detection / track summary data. |
| `audio_natural` | `str` | `""` | AST environmental sound classification result. |
| `audio_speech` | `str` | `""` | Whisper speech transcription text. |
| `llm_scene_description` | `str` | `""` | LLM-generated scene description. |
| `extra` | `dict[str, Any]` | `{}` | Extra fields for forward compatibility. |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `to_dict` | `(self, *, include_transient: bool = False) -> dict` | Convert to a plain dict for JSON checkpoint serialization. |
| `from_dict` | `@classmethod (cls, d: dict) -> Scene` | Reconstruct a Scene from a plain dict (e.g. from checkpoint). |
| `deepcopy` | `(self) -> Scene` | Return a deep copy (including frame arrays). |
| `shallow_copy` | `(self, *, share_frames: bool = True) -> Scene` | Return a shallow copy, optionally sharing frame arrays. |

---

## `kairos.core.pipeline`

Pipeline step orchestration.

| Function | Signature | Description |
|---|---|---|
| `run_pipeline` | `(video_path: str, output_dir: str, cfg: PipelineConfig, client: Any, redo_steps: list[str] \| None = None, redo_only: bool = False) -> None` | Run the full Kairos video-processing pipeline for a single video. |

---

## `kairos.core.checkpoint`

Checkpoint reading, writing, and frame cleanup.

| Function | Signature | Description |
|---|---|---|
| `clear_frames` | `(scene_list: list[dict[str, Any]]) -> list[dict[str, Any]]` | Strip heavy per-frame keys from every scene dictionary before serialization. |
| `read_json` | `(json_path: str \| Path) -> dict[str, Any]` | Read a JSON checkpoint file and return its contents as a dictionary. |
| `save_checkpoint` | `(checkpoint: dict[str, Any] \| list[dict[str, Any]], path: str \| Path) -> dict[str, Any]` | Persist a checkpoint to disk after stripping heavy frame data. |
| `have_key` | `(scenes: list[dict[str, Any]], key: str) -> bool` | Check whether every scene dictionary contains a given key. |
| `save_clips` | `(video_path: str, scenes: list[dict[str, Any]], output_dir: str) -> list[dict[str, Any]]` | Extract per-scene video clips from a source video using FFmpeg. |

---

## `kairos.core.logging`

Pipeline step logging: hardware metrics, GPU stats, and timing decorator.

| Function | Signature | Description |
|---|---|---|
| `get_system_context` | `() -> dict[str, Any]` | Return a summary of the current hardware, OS, and GPU environment. |
| `get_gpu_stats` | `() -> list[dict[str, Any]]` | Return per-GPU utilisation statistics (via NVML, CUDA, or empty). |
| `initiate_log` | `(video_path: str, run_description: str, params: dict[str, Any] \| None = None) -> dict[str, Any]` | Create the initial log dictionary at the start of a pipeline run. |
| `complete_log` | `(log: dict[str, Any], steps: dict[str, dict[str, Any]], vid_len: str, scene_num: int, vid_df: dict[str, Any] \| None = None) -> dict[str, Any]` | Finalise a pipeline log with timing totals and per-step details. |
| `save_log` | `(data: dict[str, Any], path: str) -> str` | Save a JSON-serialisable dictionary to a timestamped file. |
| `log_step` | `() -> Callable[[Callable[P, T]], Callable[P, tuple[T, dict[str, Any]]]]` | Decorator factory that wraps a function with resource logging (wall time, RAM, GPU, I/O). |

---

## `kairos.core.timing`

Pipeline stage timing decorator and JSON timing report.

| Function | Signature | Description |
|---|---|---|
| `timed_stage` | `(stage_name: str) -> Callable` | Decorator that logs wall-clock time for a pipeline stage. |
| `get_timing_records` | `() -> list[dict[str, Any]]` | Return a copy of all accumulated timing records. |
| `clear_timing_records` | `() -> None` | Clear all accumulated timing records. |
| `save_timing_report` | `(path: str \| Path) -> str` | Write all timing records to a JSON file and return the path. |

---

## `kairos.core.redo`

Redo logic for selectively re-running pipeline stages.

### Constants

| Constant | Type | Description |
|---|---|---|
| `PIPELINE_ORDER` | `list[str]` | Canonical execution order of every pipeline stage. |
| `REDO_CHOICES` | `list[str]` | Valid stage names accepted by the `--redo` CLI flag. |

### Functions

| Function | Signature | Description |
|---|---|---|
| `resolve_dependents` | `(steps: Iterable[str]) -> set[str]` | Compute the full set of stages affected by redoing the given steps (transitive). |
| `apply_redo` | `(checkpoint: dict[str, object], output_dir: str \| Path \| None, redo_steps: Iterable[str] \| None, redo_only: bool = False) -> tuple[dict[str, object], dict[str, object]]` | Clear checkpoint data for the requested stages so they will re-run. |
| `get_stop_after_step` | `(redo_steps: Iterable[str] \| None) -> str \| None` | Return the latest pipeline stage present in `redo_steps`. |

---

## `kairos.core.utils`

Shared utility functions: printing, timecodes, prompt loading, normalization, retry, and helpers.

| Function | Signature | Description |
|---|---|---|
| `print_section` | `(title: str) -> None` | Print a section header surrounded by separator lines. |
| `print_prefixed` | `(prefix: str, message: str, indent: int = 0) -> None` | Print a message with a bracketed prefix and optional indent. |
| `format_timecode` | `(seconds: float \| None) -> str` | Convert seconds to an `HH:MM:SS.mmm` timecode string. |
| `load_prompt` | `(filename: str) -> str` | Load a prompt template from the `prompts/` package directory. |
| `apply_gpt_normalization` | `(text: str, filename: str = "gpt_normalizations.json") -> str` | Apply word-boundary find-and-replace rules before sending text to GPT. |
| `is_rate_limit_error` | `(exc: Exception) -> bool` | Check if an exception indicates an API rate-limit error. |
| `retry_with_backoff` | `(fn: Callable[[], Any], *, max_retries: int = 3, base_sec: float = 2.0, is_retryable: Callable[[Exception], bool] \| None = None, jitter: bool = True) -> Any` | Call `fn()` with exponential backoff on retryable errors. |
| `flatten` | `(values: list[Any] \| None) -> list[Any]` | Flatten a list whose elements may themselves be lists or tuples (one level). |

---

## `kairos.llm.client`

LLM client abstraction: Protocol + Gemini/OpenAI/Claude implementations.

### `class LLMClient` (Protocol)

Unified interface for LLM generation.

| Method | Signature | Description |
|---|---|---|
| `model` | `@property -> str` | Return the model identifier string. |
| `generate` | `(self, prompt: str, *, system: str \| None = None, max_tokens: int = 2048, temperature: float = 0.3) -> str` | Generate text from a prompt. |

### `class GeminiLLMClient`

Gemini via Vertex AI (google-genai SDK).

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, client: Any, model: str) -> None` | Initialise with a `google.genai.Client` and a model identifier. |
| `model` | `@property -> str` | Return the Gemini model identifier. |
| `generate` | `(self, prompt: str, *, system: str \| None = None, max_tokens: int = 2048, temperature: float = 0.3) -> str` | Generate text using the Gemini model. |

### `class OpenAILLMClient`

OpenAI / Azure OpenAI (openai SDK).

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, client: Any, model: str) -> None` | Initialise with an `openai.OpenAI` client and model name. |
| `model` | `@property -> str` | Return the OpenAI model identifier. |
| `generate` | `(self, prompt: str, *, system: str \| None = None, max_tokens: int = 2048, temperature: float = 0.3) -> str` | Generate text using the OpenAI chat completions API. |

### `class ClaudeLLMClient`

Claude via Vertex AI (anthropic[vertex] SDK).

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(self, client: Any, model: str) -> None` | Initialise with an `AnthropicVertex` client and model identifier. |
| `model` | `@property -> str` | Return the Claude model identifier. |
| `generate` | `(self, prompt: str, *, system: str \| None = None, max_tokens: int = 2048, temperature: float = 0.3) -> str` | Generate text using the Claude messages API. |

### Functions

| Function | Signature | Description |
|---|---|---|
| `build_llm_client` | `(llm: str \| None = None) -> LLMClient` | Build an LLM client from environment variables (`"gemini"`, `"openai"`, or `"claude"`). |
| `get_embedding_client` | `() -> Any` | Return a raw Gemini `genai.Client` suitable for embedding calls. |

---

## `kairos.llm.rag`

RAG: embed scenes/synopsis, retrieve top-k, generate answers.

| Function | Signature | Description |
|---|---|---|
| `build_contexts` | `(checkpoint: dict[str, Any]) -> list[str]` | Build a flat list of embedding-ready context strings from a checkpoint. |
| `embed_contexts` | `(contexts: list[str], client: Any \| None = None, model: str = EMBEDDING_MODEL, batch_size: int = 250) -> list[list[float]]` | Embed a list of context strings in batches via the Gemini embedding endpoint. |
| `embed_question` | `(question: str, client: Any \| None = None, model: str = EMBEDDING_MODEL) -> Any` | Embed a single question string. |
| `get_top_k_similar` | `(question_embedding: Any, embeddings: list[Any], contexts: list[str], k: int = 5, debug: bool = False, cluster_metadata: dict[str, Any] \| None = None, top_c: int = 3, alpha: float = 0.3) -> list[tuple[str, float]]` | Find the k most similar contexts to a question embedding (cosine + cluster boost). |
| `create_answer` | `(question: str, top_matches: list[tuple[str, float]], client: Any \| None = None, model: str = GENERATION_MODEL) -> str` | Generate a natural-language answer from retrieved context. |
| `make_embedding` | `(checkpoint: dict[str, Any], output_path: str, model: str = EMBEDDING_MODEL, embedding_client: Any \| None = None) -> dict[str, Any]` | Build, cluster, and persist RAG embeddings for a checkpoint. |
| `load_rag_embeddings` | `(path: str) -> dict[str, Any]` | Load previously-saved RAG embeddings from a JSON file. |
| `save_rag_embeddings` | `(path: str, contexts: list[str], embeddings: list[list[float]], model: str = EMBEDDING_MODEL, kmeans_clusters: dict[str, Any] \| None = None) -> dict[str, Any]` | Save RAG embeddings and contexts to a JSON file. |
| `compute_kmeans_clusters` | `(embeddings: list[Any], num_clusters: int \| None = None, random_state: int = 42) -> dict[str, Any]` | Run K-Means clustering over embedding vectors. |

---

## `kairos.llm.scene_description`

LLM-powered scene description with two-stage map-reduce.

| Function | Signature | Description |
|---|---|---|
| `describe_scenes` | `(scenes: list[dict[str, Any]], client: LLMClient, hist_size: int = 3, YOLO_key: str = "yolo_detections", FLIP_key: str = "frame_captions", ASR_key: str = "audio_natural", AST_key: str = "audio_speech", SUMMARY_key: str = "llm_scene_description", ..., debug: bool = False) -> list[dict[str, Any]]` | Two-stage (map-reduce) scene description pipeline with parallel LLM calls. |
| `describe_flash_scene` | `(scene_text: str, client: LLMClient, prompt_path: str \| None = None, gpt_temperature: float = 0.3, video_path: str \| None = None) -> str` | Generate an LLM summary for a single scene. |
| `raw_descriptions` | `(scenes: list[dict[str, Any]], YOLO_key: str = "yolo_detections", FLIP_key: str = "frame_captions", ASR_key: str = "audio_natural", AST_key: str = "audio_speech") -> list[str]` | Convert a list of scene dicts into raw formatted text descriptions. |
| `format_single_description` | `(captions: list[str], yolo: list[dict[str, Any]] \| dict[str \| int, list[dict[str, Any]]]) -> str` | Format captions and YOLO detections for a single scene into text. |

---

## `kairos.llm.synopsis`

Video synopsis orchestration: scene summarization and structured synopsis generation.

| Function | Signature | Description |
|---|---|---|
| `summarize_scenes` | `(client: Any, scenes: list[dict[str, Any]], chunk_size: int = CHUNK_SIZE, summary_len: int = FINAL_CHUNK_SIZE, debug: bool = False, output_dir: str \| None = None, max_workers: int \| None = None, reduce_group_size: int = SUMMARY_REDUCE_GROUP_SIZE) -> dict[str, Any]` | Summarise scenes into a narrative via parallel map-reduce. |
| `synthesize_synopsis` | `(client: Any, data: dict[str, Any], debug: bool = False, output_dir: str \| None = None, synopsis_ext: str = "md", highlights_count: str \| int = "4-6", timeline_count: str \| int = "4-6", extra_questions_count: int = 15, consistency_pass_mode: str = "off") -> dict[str, Any]` | Produce a final synopsis and Q&A from the narrative. |
| `call_gpt` | `(client: Any, prompt: str, retries: int = 6, retry_base_sec: float = 2.0) -> str` | Call the LLM with automatic retries on transient errors. |
| `call_gpt_safe` | `(client: Any, prompt: str, fallback_text: str, debug: bool = False, context: str = "call", safe_prompt: str \| None = None, raw_fallback: str \| None = None) -> str` | Call the LLM with graceful fallback on failure. |

---

## `kairos.audio.classifier`

MIT AST (Audio Spectrogram Transformer) parallelized per-scene classification.

| Function | Signature | Description |
|---|---|---|
| `classify_scene_audio` | `(audio_slice: np.ndarray, sr: int, threshold: float = 0.3, device: str = "cpu", fe: ASTFeatureExtractor \| None = None, model: ASTForAudioClassification \| None = None) -> str` | Classify an audio segment using the AST model; returns comma-separated labels. |
| `extract_sounds_optimized` | `(scenes: list[dict], scan_result: dict, target_sr: int = 16000, max_workers: int = 4, use_processes: bool = False, force_cpu: bool = False, debug: bool = False) -> tuple[list[dict], dict]` | Run AST classification per scene with parallel execution and skip logic. |

---

## `kairos.audio.extraction`

Audio extraction from video files using PyAV with ffmpeg fallback.

| Function | Signature | Description |
|---|---|---|
| `load_audio_av` | `(video_path: str, target_sr: int = 16000, debug: bool = False) -> tuple[np.ndarray, int]` | Extract full audio from a video file using PyAV, with ffmpeg fallback. |

---

## `kairos.audio.language`

Language detection using Whisper.

| Function | Signature | Description |
|---|---|---|
| `detect_languages` | `(audio: np.ndarray, sr: int, speech_regions: list[dict], debug: bool = False) -> dict[str, object]` | Detect spoken languages in audio using OpenAI Whisper (returns primary language, counts, multilingual flag). |

---

## `kairos.audio.prescan`

Audio pre-scan with dynamic thresholds.

| Function | Signature | Description |
|---|---|---|
| `scan_audio` | `(video_path: str, scenes: list[dict], target_sr: int = 16000, debug: bool = False) -> dict` | Perform a full 2-stage audio pre-scan with dynamic thresholds (extraction, RMS, VAD, spectral, language). |
| `get_dynamic_thresholds` | `(duration_minutes: float) -> dict[str, float \| int]` | Get dynamically scaled audio analysis thresholds based on video duration. |

---

## `kairos.audio.rms`

RMS energy profiling for audio signals.

| Function | Signature | Description |
|---|---|---|
| `compute_rms_profile` | `(audio: np.ndarray, sr: int, window_sec: float = 1.0) -> dict[str, np.ndarray \| float]` | Compute the RMS energy profile of an audio signal (returns per-window RMS values and dBFS stats). |
| `compute_per_scene_rms` | `(audio: np.ndarray, sr: int, scenes: list[dict]) -> list[float]` | Compute the RMS energy level in dBFS for each scene. |

---

## `kairos.audio.spectral`

Spectral flatness computation for audio signals.

| Function | Signature | Description |
|---|---|---|
| `compute_spectral_flatness_mean` | `(audio: np.ndarray, sr: int, debug: bool = False) -> float` | Compute the mean spectral flatness of an audio signal (0 = tonal, 1 = noise). |

---

## `kairos.audio.text_filter`

Hallucination filtering and text cleaning for Whisper transcription output.

| Function | Signature | Description |
|---|---|---|
| `filter_hallucinations` | `(segments: list[dict], primary_lang: str \| None = None) -> list[dict]` | Filter Whisper hallucinated segments (low confidence, repeated, emoji-heavy). |
| `clean_repetitive_text` | `(text: str) -> str` | Clean repetitive phrases and words from transcription text. |

---

## `kairos.audio.transcription`

Whisper-based speech transcription with parallel chunking and scene mapping.

| Function | Signature | Description |
|---|---|---|
| `extract_speech_singlecall` | `(scenes: list[dict], scan_result: dict, model_size: str = "small", use_vad: bool = True, language: str \| None = None, parallel: bool = False, use_api: bool = True, force_cpu: bool = False, debug: bool = False) -> tuple[list[dict], dict]` | Main entry point: scan audio, transcribe, and map speech to scenes. |
| `transcribe_parallel` | `(audio: np.ndarray, sr: int, model_size: str = "medium", chunk_size_sec: int = 600, overlap_sec: int = 30, lang_info: dict \| None = None, use_vad: bool = True, force_cpu: bool = False, debug: bool = False, use_api: bool = True, client: Any \| None = None) -> dict` | Transcribe audio in parallel chunks and merge the results. |
| `transcribe_full_video` | `(audio: np.ndarray, sr: int, model_size: str = "small", use_vad: bool = True, force_cpu: bool = False, debug: bool = False, silero_model: Any \| None = None, get_speech_ts_fn: Callable \| None = None) -> dict` | Transcribe an entire audio track in a single Whisper call. |
| `map_segments_to_scenes` | `(whisper_segments: list[dict], scenes: list[dict]) -> list[str]` | Map Whisper segments to scene boundaries and aggregate text. |
| `clean_audio` | `(audio: np.ndarray, sr: int, silero_model: Any \| None = None, get_speech_ts: Callable \| None = None) -> np.ndarray` | Denoise an audio waveform with optional VAD-guided enhancement. |

---

## `kairos.audio.vad`

Silero VAD speech detection with lazy model loading.

| Function | Signature | Description |
|---|---|---|
| `detect_speech_regions` | `(audio: np.ndarray, sr: int, thresholds: dict[str, float \| int], silero_model: Any \| None = None, get_speech_ts_fn: Callable \| None = None) -> list[dict[str, float]]` | Detect speech regions in an audio waveform using Silero VAD. |

---

## `kairos.audio.whisper_api`

Whisper API client for Azure OpenAI Whisper transcription.

| Function | Signature | Description |
|---|---|---|
| `transcribe_via_api` | `(audio_np: np.ndarray, sr: int, language: str \| None = None, client: AzureOpenAI \| None = None) -> list[dict]` | Transcribe audio via the Azure OpenAI Whisper API. |

---

## `kairos.video.scene_detection`

Scene detection using PySceneDetect with fallback segmentation.

| Function | Signature | Description |
|---|---|---|
| `get_scene_list` | `(input_video_path: str, threshold: float = 27, min_scene_sec: int = 2, frame_skip: int = 3, retry_threshold_factor: float = 0.5, fallback_interval_sec: int = 20) -> list[dict]` | Detect scenes in a video using PySceneDetect and return structured metadata. |

---

## `kairos.video.frame_sampling`

Frame sampling from video scenes at fixed counts or FPS.

| Function | Signature | Description |
|---|---|---|
| `sample_frames` | `(input_video_path: str, scenes: list[dict], num_frames: int = 4, new_size: int = 320, output_dir: str \| None = None) -> list[dict]` | Loop over scenes and attach sampled frames to each scene dict. |
| `sample_fps` | `(input_video_path: str, scenes: list[dict], fps: float = 4.0, new_size: int = 320, output_dir: str \| None = None, frames_key: str = "frames", frame_paths_key: str = "frame_paths", store_paths: bool = False, store_meta: bool = False) -> list[dict]` | Loop over scenes and attach frames sampled at a fixed FPS. |
| `sample_from_clip` | `(input_video_path: str, scene_index: int, start_seconds: float, end_seconds: float, num_frames: int = 5, new_size: int = 320) -> list[np.ndarray]` | Sample evenly-spaced frames from a scene interval. |
| `sample_from_clip_fps` | `(input_video_path: str, scene_index: int, start_seconds: float, end_seconds: float, fps: float = 4.0, new_size: int = 320, return_meta: bool = False) -> list[np.ndarray] \| tuple[list[np.ndarray], list[int], list[float]]` | Sample frames from a scene interval at a fixed FPS. |
| `resize_frame` | `(frame: np.ndarray, new_size: int = 320) -> np.ndarray` | Resize a frame so the longest side equals `new_size`, preserving aspect ratio. |

---

## `kairos.video.frame_captioning`

BLIP frame captioning with lazy model loading.

| Function | Signature | Description |
|---|---|---|
| `caption_frames` | `(scenes: list[dict], model: Any = None, processor: Any = None, debug: bool = False, **blip_kwargs: Any) -> list[dict]` | Run BLIP on each frame in every scene and attach captions. |
| `blip_frame` | `(image: Image.Image \| np.ndarray, model: Any = None, processor: Any = None, prompt: str \| None = None, **generate_kwargs: Any) -> str` | Generate a BLIP caption for a single frame. |

---

## `kairos.video.object_detection`

YOLOv8 object detection and tracking per scene (orchestrator).

| Function | Signature | Description |
|---|---|---|
| `detect_object_yolo` | `(scenes: list[dict], model_size: str = "models/yolov8s.pt", model: Any = None, conf: float = 0.5, iou: float = 0.45, output_dir: str \| None = None, use_bytetrack: bool = True, tracker: str = "bytetrack.yaml", fallback_iou: float = 0.3, frame_key: str = "frames", summary_key: str = "yolo_detections", debug: bool = False, **track_kwargs: Any) -> list[dict]` | Run YOLO detection and tracking on all scenes. |

---

## `kairos.video.tracking`

Object tracking: IoU-based fallback tracker and track building.

| Function | Signature | Description |
|---|---|---|
| `assign_track_ids_iou` | `(yolo_dict: dict[int, list[dict]], iou_threshold: float = 0.3) -> dict[int, list[dict]]` | Assign track IDs using a simple IoU-based fallback tracker. |
| `build_tracks` | `(yolo_dict: dict[int, list[dict]]) -> dict[int, dict]` | Group detections by track ID into per-track dictionaries. |
| `has_track_ids` | `(yolo_dict: dict[int, list[dict]]) -> bool` | Check whether any detection already has a track ID. |

---

## `kairos.video.track_summary`

Track summary building and formatting for YOLO detections.

| Function | Signature | Description |
|---|---|---|
| `build_track_summaries` | `(frames: list[np.ndarray], yolo_dict: dict[int, list[dict]], **kwargs: object) -> list[dict]` | Build per-scene track summaries with movement and relation labels. |
| `format_track_summaries` | `(summaries: list[dict], style: str = "compact") -> list[str]` | Format a list of track summaries as human-readable strings. |

---

## `kairos.video.spatial`

Spatial analysis: position labels, movement detection, and inter-object relations.

| Function | Signature | Description |
|---|---|---|
| `position_label` | `(x_center: float, y_center: float, frame_w: int, frame_h: int) -> str` | Compute a human-readable position label (e.g. `"top-left"`) for a point in a frame. |
| `movement_label` | `(start_center: tuple[float, float], end_center: tuple[float, float], start_area: float, end_area: float, frame_w: int, frame_h: int) -> str` | Describe the movement of a tracked object between its first and last detection. |
| `compute_relations` | `(tracks: dict[int, dict], yolo_dict: dict[int, list[dict]], frame_w: int, frame_h: int, rel_min_frames: int = 2, proximity_ratio: float = 0.12, moving_with_min_frames: int = 2, moving_with_cos: float = 0.8, moving_with_speed_ratio: tuple[float, float] = (0.5, 2.0), moving_with_min_speed: float = 1.0) -> dict[int, list[str]]` | Compute spatial relations and moving-with relations for tracked objects. |
| `path_metrics` | `(dets: list[dict]) -> tuple[float, float, float]` | Compute path-related metrics: `(path_length, net_displacement, angle_variance)`. |

---

## `kairos.video.debug_draw`

Debug drawing utilities for YOLO detections.

| Function | Signature | Description |
|---|---|---|
| `debug_draw_yolo` | `(frame: np.ndarray, detections: list[dict], save_path: str \| None = None) -> np.ndarray` | Draw YOLO detections on a frame for debugging (bounding boxes, labels, track IDs). |

---

## `kairos.video.yolo_inference`

YOLOv8 inference: single-frame detection and multi-frame tracking.

| Function | Signature | Description |
|---|---|---|
| `run_yolo_on_frame` | `(model: Any, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[dict]` | Run YOLOv8 on a single frame and return detections. |
| `run_yolo_track_on_frames` | `(model: Any, frames: list[np.ndarray], conf: float = 0.25, iou: float = 0.45, tracker: str = "bytetrack.yaml") -> Any \| None` | Run YOLOv8 tracking on a list of frames (returns results or `None` on failure). |
| `parse_yolo_results` | `(results: Any, model: Any) -> dict[int, list[dict]]` | Parse raw YOLO result objects into a structured detection dictionary. |

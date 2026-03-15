"""Pipeline configuration as a dataclass with presets.

Every tunable parameter lives here.  Presets (``fast``, ``motion_sensitive``,
``static_video``) override sensible subsets.  A ``__post_init__`` validator
catches nonsensical values early.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kairos.core.exceptions import KairosConfigError


@dataclass
class PipelineConfig:
    """All tunable parameters for the Kairos video processing pipeline.

    Attributes:
        pyscene_threshold: Sensitivity for PySceneDetect content detector.
        pyscene_shortest: Minimum scene length in seconds.
        frames_per_scene: Number of frames to sample per scene.
        frame_resolution: Target resolution (longest side) for sampled frames.
        blip_start_prompt: Starting text prompt for BLIP captioning.
        blip_caption_len: Maximum caption length in tokens.
        blip_min_length: Minimum caption length in tokens.
        blip_num_beams: Number of beams for beam search.
        blip_do_sample: Whether to use sampling during generation.
        blip_top_p: Nucleus sampling probability threshold.
        blip_temperature: Sampling temperature.
        blip_length_penalty: Length penalty for beam search.
        blip_no_repeat_ngram_size: N-gram size to prevent repetition.
        blip_repetition_penalty: Repetition penalty factor.
        yolo_model_path: Path to the YOLOv8 model weights.
        yolo_action_fps: Frames per second for YOLO action sampling.
        yolo_conf_thres: YOLO confidence threshold.
        yolo_iou_thres: YOLO IoU threshold for NMS.
        ast_target_sr: Target sample rate for AST audio classification.
        asr_model_size: Whisper model size for speech recognition.
        asr_use_vad: Whether to use Voice Activity Detection.
        asr_target_sr: Target sample rate for ASR.
        llm_scene_history: Number of previous scenes to include as context.
        llm_chunk_len: Maximum character length per narrative chunk.
        llm_summary_len: Maximum character length for the full summary.
        llm_cooldown_sec: Cooldown in seconds between LLM API calls.
        rag_top_k_context: Number of top contexts to retrieve for RAG.
        llm_max_workers: Maximum number of parallel LLM workers.
        data_dir: Root data directory.
        prompts_dir: Directory containing prompt templates.
        logs_dir: Directory for pipeline logs.
    """

    # Scene detection
    pyscene_threshold: float = 27.0
    pyscene_shortest: float = 2.0

    # Frame sampling
    frames_per_scene: int = 3
    frame_resolution: int = 320

    # BLIP captioning
    blip_start_prompt: str = "a video frame of"
    blip_caption_len: int = 50
    blip_min_length: int = 15
    blip_num_beams: int = 1
    blip_do_sample: bool = True
    blip_top_p: float = 0.85
    blip_temperature: float = 0.65
    blip_length_penalty: float = 1.0
    blip_no_repeat_ngram_size: int = 3
    blip_repetition_penalty: float = 1.1

    # YOLO object detection
    yolo_model_path: str = "models/yolov8s.pt"
    yolo_action_fps: float = 4.0
    yolo_conf_thres: float = 0.8
    yolo_iou_thres: float = 0.5

    # Audio
    ast_target_sr: int = 16000
    asr_model_size: str = "medium"
    asr_use_vad: bool = True
    asr_target_sr: int = 16000

    # LLM scene description
    llm_scene_history: int = 5
    llm_chunk_len: int = 20000
    llm_summary_len: int = 50000
    llm_cooldown_sec: float = 0.0

    # RAG
    rag_top_k_context: int = 10

    # Parallelism
    llm_max_workers: int = 4

    # Paths (defaults derived at runtime if not set)
    data_dir: str = "data"
    prompts_dir: str = ""  # empty → resolved via importlib.resources
    logs_dir: str = "logs"

    def __post_init__(self) -> None:
        """Validate configuration values eagerly.

        Raises:
            KairosConfigError: If any configuration value is out of its
                acceptable range.
        """
        if self.pyscene_threshold <= 0:
            raise KairosConfigError(
                f"pyscene_threshold must be > 0, got {self.pyscene_threshold}"
            )
        if self.pyscene_shortest < 0:
            raise KairosConfigError(
                f"pyscene_shortest must be >= 0, got {self.pyscene_shortest}"
            )
        if self.frames_per_scene < 1:
            raise KairosConfigError(
                f"frames_per_scene must be >= 1, got {self.frames_per_scene}"
            )
        if self.frame_resolution < 1:
            raise KairosConfigError(
                f"frame_resolution must be >= 1, got {self.frame_resolution}"
            )
        if self.blip_caption_len < 1:
            raise KairosConfigError(
                f"blip_caption_len must be >= 1, got {self.blip_caption_len}"
            )
        if self.blip_min_length < 1:
            raise KairosConfigError(
                f"blip_min_length must be >= 1, got {self.blip_min_length}"
            )
        if self.yolo_conf_thres < 0 or self.yolo_conf_thres > 1:
            raise KairosConfigError(
                f"yolo_conf_thres must be in [0, 1], got {self.yolo_conf_thres}"
            )
        if self.llm_max_workers < 1:
            raise KairosConfigError(
                f"llm_max_workers must be >= 1, got {self.llm_max_workers}"
            )
        if self.rag_top_k_context < 1:
            raise KairosConfigError(
                f"rag_top_k_context must be >= 1, got {self.rag_top_k_context}"
            )
        if self.llm_cooldown_sec < 0:
            raise KairosConfigError(
                f"llm_cooldown_sec must be >= 0, got {self.llm_cooldown_sec}"
            )

        # Resolve prompts_dir via importlib.resources if not explicitly set
        if not self.prompts_dir:
            try:
                ref = importlib.resources.files("kairos") / "prompts"
                self.prompts_dir = str(ref)
            except (TypeError, ModuleNotFoundError):
                self.prompts_dir = str(Path(__file__).resolve().parent / "prompts")

    @classmethod
    def default(cls) -> PipelineConfig:
        """Create a configuration with all default values.

        Returns:
            A new ``PipelineConfig`` instance with default settings.
        """
        return cls()

    @classmethod
    def fast(cls) -> PipelineConfig:
        """Create a configuration optimised for speed over accuracy.

        Uses a higher scene-detection threshold, fewer sampled frames,
        larger LLM chunks, and more parallel workers.

        Returns:
            A new ``PipelineConfig`` instance tuned for fast processing.
        """
        return cls(
            pyscene_threshold=40,
            frames_per_scene=1,
            llm_chunk_len=500000,
            llm_summary_len=500000,
            llm_max_workers=8,
        )

    @classmethod
    def motion_sensitive(cls) -> PipelineConfig:
        """Create a configuration tuned for high-motion content.

        Uses a lower scene-detection threshold, more frames per scene,
        and a higher YOLO action sampling rate.

        Returns:
            A new ``PipelineConfig`` instance tuned for motion-heavy video.
        """
        return cls(
            pyscene_threshold=15,
            pyscene_shortest=0.5,
            frames_per_scene=5,
            yolo_action_fps=8,
        )

    @classmethod
    def static_video(cls) -> PipelineConfig:
        """Create a configuration tuned for slow-moving or static content.

        Uses a very low scene-detection threshold, fewer frames per scene,
        and a reduced YOLO action sampling rate.

        Returns:
            A new ``PipelineConfig`` instance tuned for static video.
        """
        return cls(
            pyscene_threshold=3,
            frames_per_scene=1,
            yolo_action_fps=0.5,
        )

    @property
    def blip_params(self) -> dict[str, Any]:
        """Collect all BLIP generation fields into a dict.

        Used for ``**kwargs`` forwarding to ``model.generate()``.

        Returns:
            A dictionary of BLIP generation parameters.
        """
        return {
            "prompt": self.blip_start_prompt,
            "max_length": self.blip_caption_len,
            "min_length": self.blip_min_length,
            "num_beams": self.blip_num_beams,
            "do_sample": self.blip_do_sample,
            "top_p": self.blip_top_p,
            "temperature": self.blip_temperature,
            "length_penalty": self.blip_length_penalty,
            "no_repeat_ngram_size": self.blip_no_repeat_ngram_size,
            "repetition_penalty": self.blip_repetition_penalty,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain ``dict``.

        Returns:
            A dictionary representation of this configuration.
        """
        return asdict(self)

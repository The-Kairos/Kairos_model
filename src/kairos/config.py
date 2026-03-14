"""Pipeline configuration as a dataclass with presets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All tunable parameters for the Kairos video processing pipeline."""

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

    @classmethod
    def default(cls) -> PipelineConfig:
        return cls()

    @classmethod
    def fast(cls) -> PipelineConfig:
        return cls(
            pyscene_threshold=40,
            frames_per_scene=1,
            llm_chunk_len=500000,
            llm_summary_len=500000,
        )

    @classmethod
    def motion_sensitive(cls) -> PipelineConfig:
        return cls(
            pyscene_threshold=15,
            pyscene_shortest=0.5,
            frames_per_scene=5,
            yolo_action_fps=8,
        )

    @classmethod
    def static_video(cls) -> PipelineConfig:
        return cls(
            pyscene_threshold=3,
            frames_per_scene=1,
            yolo_action_fps=0.5,
        )

    @property
    def blip_params(self) -> dict:
        """Collect all BLIP generation fields into a single dict for **kwargs forwarding."""
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

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

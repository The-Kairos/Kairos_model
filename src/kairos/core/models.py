"""Thread-safe model registry: lazy-load and cache ML models with a single lock per model.

Usage::

    from kairos.core.models import ModelRegistry

    registry = ModelRegistry.get()          # process-wide singleton
    model, processor = registry.blip()      # loads on first call, cached after
    registry.release("blip")                # free memory when done
    registry.release_all()                  # free everything
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from kairos.core.exceptions import KairosModelError


def _default_device() -> str:
    """Return ``'cuda'`` when a GPU is available, else ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelRegistry:
    """Process-wide singleton that lazy-loads and caches ML models thread-safely.

    Each model family (blip, yolo, ast, whisper, silero_vad, scene_detector)
    has its own :class:`threading.Lock` so independent loads can run concurrently
    while the *same* model is only loaded once.
    """

    _instance: ModelRegistry | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, Any] = {}
        self._meta_lock = threading.Lock()

    # ---- Singleton access ------------------------------------------------

    @classmethod
    def get(cls) -> ModelRegistry:
        """Return the process-wide singleton (create on first call)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---- Internal helpers ------------------------------------------------

    def _lock_for(self, name: str) -> threading.Lock:
        with self._meta_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    def _get_or_load(self, name: str, loader):
        """Return the cached value or call *loader()* once under a lock."""
        if name in self._cache:
            return self._cache[name]
        lock = self._lock_for(name)
        with lock:
            if name not in self._cache:
                self._cache[name] = loader()
            return self._cache[name]

    # ---- Public model accessors ------------------------------------------

    def blip(self, device: str | None = None) -> tuple:
        """Return ``(model, processor)`` for BLIP image captioning.

        The model is loaded on the first call and moved to *device*
        (defaults to GPU if available).
        """

        def _load():
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError as exc:
                raise KairosModelError(
                    "transformers is required for BLIP captioning"
                ) from exc

            dev = device or _default_device()
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(dev)
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base", use_fast=True
            )
            return model, processor

        return self._get_or_load("blip", _load)

    def yolo(self, model_path: str = "models/yolov8s.pt") -> Any:
        """Return a loaded YOLO model."""

        def _load():
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise KairosModelError(
                    "ultralytics is required for YOLO detection"
                ) from exc
            return YOLO(model_path)

        return self._get_or_load("yolo", _load)

    def ast(self, device: str | None = None) -> tuple:
        """Return ``(feature_extractor, model)`` for MIT AST audio classification."""

        def _load():
            try:
                from transformers import (
                    AutoFeatureExtractor,
                    AutoModelForAudioClassification,
                )
            except ImportError as exc:
                raise KairosModelError(
                    "transformers is required for AST classification"
                ) from exc

            ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            fe = AutoFeatureExtractor.from_pretrained(ast_name)
            model = AutoModelForAudioClassification.from_pretrained(ast_name)
            dev = device or "cpu"  # AST typically runs on CPU
            model = model.to(dev)
            return fe, model

        return self._get_or_load("ast", _load)

    def whisper(self, model_size: str = "medium", device: str | None = None) -> Any:
        """Return a loaded local Whisper model."""
        cache_key = f"whisper_{model_size}"

        def _load():
            try:
                import whisper
            except ImportError as exc:
                raise KairosModelError(
                    "openai-whisper is required for local transcription"
                ) from exc
            return whisper.load_model(model_size, device=device)

        return self._get_or_load(cache_key, _load)

    def silero_vad(self) -> tuple:
        """Return ``(silero_model, get_speech_timestamps_fn)``."""

        def _load():
            try:
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    onnx=False,
                    trust_repo=True,
                )
                get_speech_ts = utils[0]
                return model, get_speech_ts
            except Exception as exc:
                raise KairosModelError(f"Failed to load Silero VAD: {exc}") from exc

        return self._get_or_load("silero_vad", _load)

    # ---- Lifecycle -------------------------------------------------------

    def release(self, name: str) -> None:
        """Remove a cached model and free its memory."""
        lock = self._lock_for(name)
        with lock:
            obj = self._cache.pop(name, None)
            del obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_all(self) -> None:
        """Release every cached model."""
        with self._meta_lock:
            names = list(self._cache.keys())
        for name in names:
            self.release(name)

    def is_loaded(self, name: str) -> bool:
        """Check whether a model is currently cached."""
        return name in self._cache

    def loaded_models(self) -> list[str]:
        """Return a list of currently cached model names."""
        return list(self._cache.keys())

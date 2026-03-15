"""Embedding math helpers: pooling, normalisation, cosine similarity, and scene boundary detection."""

from __future__ import annotations

from typing import List, Tuple

import torch


def pool_tokens(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pool token embeddings into a single vector per sample."""
    if hidden_states.dim() == 3:
        return hidden_states.mean(dim=1)
    if hidden_states.dim() == 2:
        return hidden_states
    if hidden_states.dim() == 1:
        return hidden_states.unsqueeze(0)
    raise ValueError(f"Unexpected embedding shape: {tuple(hidden_states.shape)}")


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalise embedding vectors along the last dimension."""
    return torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the cosine similarity between two 1-D tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_scene_bounds(
    pooled_vectors: List[torch.Tensor],
    threshold: float,
) -> List[Tuple[int, int]]:
    """Detect scene boundaries by comparing consecutive embedding similarities."""
    if not pooled_vectors:
        return []

    bounds: List[Tuple[int, int]] = []
    start_idx = 0
    prev_vec = pooled_vectors[0]

    for idx in range(1, len(pooled_vectors)):
        sim = cosine_similarity(prev_vec, pooled_vectors[idx])
        if sim < threshold:
            bounds.append((start_idx, idx - 1))
            start_idx = idx
        prev_vec = pooled_vectors[idx]

    bounds.append((start_idx, len(pooled_vectors) - 1))
    return bounds


def bounds_to_times(
    bounds: List[Tuple[int, int]],
    timestamps: List[float],
    duration: float,
) -> List[Tuple[float, float]]:
    """Convert frame-index bounds to (start_seconds, end_seconds) intervals."""
    times: List[Tuple[float, float]] = []
    if not bounds:
        return times

    for idx, (start_idx, _end_idx) in enumerate(bounds):
        start_time = timestamps[start_idx]
        if idx + 1 < len(bounds):
            next_start_idx = bounds[idx + 1][0]
            end_time = timestamps[next_start_idx]
        else:
            end_time = duration
        if end_time < start_time:
            end_time = start_time
        times.append((start_time, end_time))
    return times

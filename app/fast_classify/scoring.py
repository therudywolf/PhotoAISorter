# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Calibrated CLIP scores and confidence from top-1 vs top-2 margin."""

from __future__ import annotations

import numpy as np

from app.constants import UNCATEGORIZED


def softmax_probs(sims: np.ndarray, *, temperature: float) -> np.ndarray:
    """Row-wise softmax over cosine similarities."""
    if sims.size == 0:
        return sims
    t = max(1e-4, float(temperature))
    scaled = sims / t
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    denom = exp.sum(axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-9)
    return exp / denom


def sims_to_tag_scores(
    sims_row: np.ndarray,
    tags: list[str],
    *,
    temperature: float,
) -> dict[str, float]:
    probs = softmax_probs(sims_row.reshape(1, -1), temperature=temperature)[0]
    return {tag: float(probs[i]) for i, tag in enumerate(tags)}


def confidence_from_probs(probs: np.ndarray) -> tuple[float, float]:
    """Return (top probability, margin to second place)."""
    if probs.size == 0:
        return 0.0, 0.0
    order = np.argsort(-probs)
    top_p = float(probs[order[0]])
    if probs.size == 1:
        return top_p, top_p
    second_p = float(probs[order[1]])
    return top_p, top_p - second_p


def needs_review(
    top_prob: float,
    margin: float,
    *,
    min_prob: float,
    min_margin: float,
) -> bool:
    return top_prob < min_prob or margin < min_margin

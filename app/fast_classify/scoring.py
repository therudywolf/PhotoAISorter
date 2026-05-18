# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Calibrated CLIP scores and confidence from top-1 vs top-2 margin."""

from __future__ import annotations

import numpy as np

from app.constants import UNCATEGORIZED


def raw_similarity_margin(sims_row: np.ndarray) -> tuple[float, float]:
    """Top cosine similarity and gap to second place (before softmax)."""
    if sims_row.size == 0:
        return 0.0, 0.0
    order = np.argsort(-sims_row)
    top = float(sims_row[order[0]])
    second = float(sims_row[order[1]]) if sims_row.size > 1 else 0.0
    return top, top - second


def topk_softmax_probs(sims_row: np.ndarray, *, temperature: float, top_k: int = 10) -> np.ndarray:
    """Softmax only among top-k cosine scores — avoids 60-way dilution on large tag lists."""
    if sims_row.size == 0:
        return sims_row
    k = max(1, min(int(top_k), int(sims_row.size)))
    idx = np.argpartition(sims_row, -k)[-k:]
    sub = sims_row[idx].astype(np.float64, copy=False)
    probs = softmax_probs(sub.reshape(1, -1), temperature=temperature)[0]
    out = np.zeros(sims_row.shape[0], dtype=np.float32)
    out[idx] = probs.astype(np.float32)
    return out


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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Quality presets: tune model size, crops, and throughput for CPU vs GPU."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.fast_classify.config import FastClassifySettings

QUALITY_FAST = "fast"
QUALITY_BALANCED = "balanced"
QUALITY_MAX = "max"
QUALITY_ULTRA = "ultra"

QUALITY_VALUES: frozenset[str] = frozenset(
    {QUALITY_FAST, QUALITY_BALANCED, QUALITY_MAX, QUALITY_ULTRA}
)


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cpu_count() -> int:
    try:
        import os

        return max(2, os.cpu_count() or 4)
    except Exception:
        return 4


def finalize_fast_classify_settings(
    settings: FastClassifySettings,
    *,
    explicit_keys: frozenset[str] | None = None,
) -> FastClassifySettings:
    """Apply quality tier defaults; respect keys the user set in gui_settings.json."""
    explicit = explicit_keys or frozenset()
    q = (settings.quality or QUALITY_ULTRA).strip().lower()
    if q not in QUALITY_VALUES:
        q = QUALITY_ULTRA
    cuda = _cuda_ready() and (settings.device or "auto").strip().lower() != "cpu"
    cpus = _cpu_count()

    if q == QUALITY_FAST:
        patch: dict = {
            "quality": q,
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "batch_size": 96 if cuda else 48,
            "image_max_side": 384,
            "multi_crop": False,
            "multi_crop_views": 1,
            "video_frames": 3,
            "prefetch_workers": min(6, cpus),
            "confidence_threshold": 0.28,
            "min_margin": 0.07,
            "softmax_temperature": 0.06,
            "exemplar_boost": 1.18,
            "text_prompt_fusion": 0.55,
            "text_prompt_max_pool": False,
            "crop_score_max_pool": False,
            "min_raw_similarity": 0.17,
            "min_raw_margin": 0.025,
            "top_k_softmax": 8,
        }
    elif q == QUALITY_BALANCED:
        patch = {
            "quality": q,
            "model_name": "ViT-B-16" if cuda else "ViT-B-32",
            "pretrained": "openai",
            "batch_size": 64 if cuda else 32,
            "image_max_side": 448,
            "multi_crop": True,
            "multi_crop_views": 3,
            "video_frames": 5,
            "prefetch_workers": min(8, cpus),
            "confidence_threshold": 0.26,
            "min_margin": 0.08,
            "softmax_temperature": 0.065,
            "exemplar_boost": 1.24,
            "text_prompt_fusion": 0.62,
            "text_prompt_max_pool": False,
            "crop_score_max_pool": True,
            "min_raw_similarity": 0.19,
            "min_raw_margin": 0.03,
            "top_k_softmax": 10,
        }
    elif q == QUALITY_MAX:
        patch = {
            "quality": q,
            "model_name": "ViT-L-14" if cuda else "ViT-B-16",
            "pretrained": "openai",
            "batch_size": 36 if cuda else 24,
            "image_max_side": 512 if cuda else 448,
            "multi_crop": True,
            "multi_crop_views": 5,
            "video_frames": 7,
            "prefetch_workers": min(12, cpus),
            "confidence_threshold": 0.22,
            "min_margin": 0.10,
            "softmax_temperature": 0.065,
            "exemplar_boost": 1.32,
            "text_prompt_fusion": 0.68,
            "text_prompt_max_pool": False,
            "crop_score_max_pool": True,
            "min_raw_similarity": 0.20,
            "min_raw_margin": 0.035,
            "top_k_softmax": 12,
        }
    else:
        patch = {
            "quality": q,
            "model_name": "ViT-L-14" if cuda else "ViT-B-16",
            "pretrained": "openai",
            "batch_size": 28 if cuda else 20,
            "image_max_side": 576 if cuda else 480,
            "multi_crop": True,
            "multi_crop_views": 9,
            "video_frames": 9,
            "prefetch_workers": min(12, cpus),
            "confidence_threshold": 0.20,
            "min_margin": 0.11,
            "softmax_temperature": 0.055,
            "exemplar_boost": 1.38,
            "text_prompt_fusion": 0.75,
            "text_prompt_max_pool": True,
            "crop_score_max_pool": True,
            "min_raw_similarity": 0.23,
            "min_raw_margin": 0.04,
            "top_k_softmax": 15,
        }

    for key in (
        "model_name",
        "pretrained",
        "batch_size",
        "image_max_side",
        "multi_crop",
        "multi_crop_views",
        "video_frames",
        "prefetch_workers",
        "confidence_threshold",
        "min_margin",
        "softmax_temperature",
        "exemplar_boost",
        "text_prompt_fusion",
        "text_prompt_max_pool",
        "crop_score_max_pool",
        "min_raw_similarity",
        "min_raw_margin",
        "top_k_softmax",
    ):
        if key in explicit:
            patch.pop(key, None)

    return replace(settings, **patch)

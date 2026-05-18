# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Settings for local CLIP / hybrid classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.fast_classify.quality import (
    QUALITY_ULTRA,
    finalize_fast_classify_settings,
)


def refs_dir() -> Path:
    from app.paths import refs_dir as _dir

    return _dir()


@dataclass(frozen=True)
class FastClassifySettings:
    quality: str = QUALITY_ULTRA
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    batch_size: int = 28
    image_max_side: int = 576
    confidence_threshold: float = 0.20
    min_margin: float = 0.11
    softmax_temperature: float = 0.055
    vlm_fallback: bool = True
    exemplar_boost: float = 1.38
    text_prompt_fusion: float = 0.75
    text_prompt_max_pool: bool = True
    crop_score_max_pool: bool = True
    device: str = "auto"
    use_fp16: bool = True
    cache_embeddings: bool = True
    multi_crop: bool = True
    multi_crop_views: int = 9
    prefetch_workers: int = 8
    video_frames: int = 9
    torch_compile: bool = True
    min_raw_similarity: float = 0.21
    min_raw_margin: float = 0.035
    top_k_softmax: int = 12
    weights_path: str = ""

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any] | None,
        *,
        explicit_keys: frozenset[str] | None = None,
    ) -> FastClassifySettings:
        if not isinstance(raw, dict):
            return finalize_fast_classify_settings(cls())
        explicit = explicit_keys if explicit_keys is not None else frozenset(raw.keys())
        s = cls(
            quality=str(raw.get("quality", QUALITY_ULTRA) or QUALITY_ULTRA),
            model_name=str(raw.get("model_name", "ViT-L-14") or "ViT-L-14"),
            pretrained=str(raw.get("pretrained", "openai") or "openai"),
            batch_size=max(1, min(512, int(raw.get("batch_size", 28)))),
            image_max_side=max(128, min(768, int(raw.get("image_max_side", 576)))),
            confidence_threshold=max(
                0.05, min(0.95, float(raw.get("confidence_threshold", 0.20)))
            ),
            min_margin=max(0.01, min(0.5, float(raw.get("min_margin", 0.11)))),
            softmax_temperature=max(
                0.01, min(0.2, float(raw.get("softmax_temperature", 0.055)))
            ),
            vlm_fallback=bool(raw.get("vlm_fallback", True)),
            exemplar_boost=max(1.0, min(2.5, float(raw.get("exemplar_boost", 1.38)))),
            text_prompt_fusion=max(
                0.0, min(1.0, float(raw.get("text_prompt_fusion", 0.75)))
            ),
            text_prompt_max_pool=bool(raw.get("text_prompt_max_pool", True)),
            crop_score_max_pool=bool(raw.get("crop_score_max_pool", True)),
            device=str(raw.get("device", "auto") or "auto"),
            use_fp16=bool(raw.get("use_fp16", True)),
            cache_embeddings=bool(raw.get("cache_embeddings", True)),
            multi_crop=bool(raw.get("multi_crop", True)),
            multi_crop_views=max(1, min(12, int(raw.get("multi_crop_views", 9)))),
            prefetch_workers=max(1, min(16, int(raw.get("prefetch_workers", 8)))),
            video_frames=max(1, min(12, int(raw.get("video_frames", 9)))),
            torch_compile=bool(raw.get("torch_compile", True)),
            min_raw_similarity=max(
                0.05, min(0.45, float(raw.get("min_raw_similarity", 0.21)))
            ),
            min_raw_margin=max(0.005, min(0.2, float(raw.get("min_raw_margin", 0.035)))),
            top_k_softmax=max(2, min(40, int(raw.get("top_k_softmax", 12)))),
            weights_path=str(raw.get("weights_path", "") or "").strip(),
        )
        return finalize_fast_classify_settings(s, explicit_keys=explicit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "batch_size": self.batch_size,
            "image_max_side": self.image_max_side,
            "confidence_threshold": self.confidence_threshold,
            "min_margin": self.min_margin,
            "softmax_temperature": self.softmax_temperature,
            "vlm_fallback": self.vlm_fallback,
            "exemplar_boost": self.exemplar_boost,
            "text_prompt_fusion": self.text_prompt_fusion,
            "text_prompt_max_pool": self.text_prompt_max_pool,
            "crop_score_max_pool": self.crop_score_max_pool,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "cache_embeddings": self.cache_embeddings,
            "multi_crop": self.multi_crop,
            "multi_crop_views": self.multi_crop_views,
            "prefetch_workers": self.prefetch_workers,
            "video_frames": self.video_frames,
            "torch_compile": self.torch_compile,
            "min_raw_similarity": self.min_raw_similarity,
            "min_raw_margin": self.min_raw_margin,
            "top_k_softmax": self.top_k_softmax,
            "weights_path": self.weights_path,
        }


def load_fast_classify_settings(gui_settings: dict[str, Any] | None = None) -> FastClassifySettings:
    if isinstance(gui_settings, dict):
        block = gui_settings.get("fast_classify")
        if isinstance(block, dict):
            return FastClassifySettings.from_dict(block)
    from app.settings_store import load_gui_settings

    saved = load_gui_settings()
    block = saved.get("fast_classify") if isinstance(saved, dict) else None
    return FastClassifySettings.from_dict(block if isinstance(block, dict) else None)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Settings for local CLIP / hybrid classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

def refs_dir() -> Path:
    from app.paths import refs_dir as _dir

    return _dir()


@dataclass(frozen=True)
class FastClassifySettings:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    batch_size: int = 64
    image_max_side: int = 448
    confidence_threshold: float = 0.24
    min_margin: float = 0.09
    softmax_temperature: float = 0.07
    vlm_fallback: bool = True
    exemplar_boost: float = 1.28
    device: str = "auto"
    use_fp16: bool = True
    cache_embeddings: bool = True
    prefetch_workers: int = 4
    video_frames: int = 5
    weights_path: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FastClassifySettings:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            model_name=str(raw.get("model_name", "ViT-B-32") or "ViT-B-32"),
            pretrained=str(raw.get("pretrained", "openai") or "openai"),
            batch_size=max(1, min(512, int(raw.get("batch_size", 64)))),
            image_max_side=max(128, min(768, int(raw.get("image_max_side", 384)))),
            confidence_threshold=max(0.05, min(0.95, float(raw.get("confidence_threshold", 0.28)))),
            min_margin=max(0.01, min(0.5, float(raw.get("min_margin", 0.06)))),
            softmax_temperature=max(0.01, min(0.2, float(raw.get("softmax_temperature", 0.05)))),
            vlm_fallback=bool(raw.get("vlm_fallback", True)),
            exemplar_boost=max(1.0, min(2.0, float(raw.get("exemplar_boost", 1.15)))),
            device=str(raw.get("device", "auto") or "auto"),
            use_fp16=bool(raw.get("use_fp16", True)),
            cache_embeddings=bool(raw.get("cache_embeddings", True)),
            prefetch_workers=max(1, min(16, int(raw.get("prefetch_workers", 4)))),
            video_frames=max(1, min(9, int(raw.get("video_frames", 3)))),
            weights_path=str(raw.get("weights_path", "") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "batch_size": self.batch_size,
            "image_max_side": self.image_max_side,
            "confidence_threshold": self.confidence_threshold,
            "min_margin": self.min_margin,
            "softmax_temperature": self.softmax_temperature,
            "vlm_fallback": self.vlm_fallback,
            "exemplar_boost": self.exemplar_boost,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "cache_embeddings": self.cache_embeddings,
            "prefetch_workers": self.prefetch_workers,
            "video_frames": self.video_frames,
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

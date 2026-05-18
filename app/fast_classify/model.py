# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Lazy-loaded OpenCLIP model for batch embedding."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from app.fast_classify.config import FastClassifySettings

_model_lock = threading.Lock()
_shared: dict[str, Any] = {}


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def missing_clip_message() -> str:
    return (
        "Для режима «Быстрая (CLIP)» установите зависимости: "
        "pip install torch open-clip-torch"
    )


class ClipEmbedder:
    def __init__(self, settings: FastClassifySettings, *, on_log: Callable[[str], None] | None = None) -> None:
        if not clip_available():
            raise ImportError(missing_clip_message())
        import open_clip
        import torch

        self._torch = torch
        self._settings = settings
        self._on_log = on_log
        cache_key = f"{settings.model_name}:{settings.pretrained}"
        with _model_lock:
            if cache_key not in _shared:
                if on_log:
                    on_log(f"CLIP: загрузка {settings.model_name} ({settings.pretrained})…")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    settings.model_name,
                    pretrained=settings.pretrained,
                )
                tokenizer = open_clip.get_tokenizer(settings.model_name)
                device = _resolve_device(settings.device, torch)
                model = model.to(device)
                model.eval()
                _shared[cache_key] = (model, preprocess, tokenizer, device)
            self._model, self._preprocess, self._tokenizer, self._device = _shared[cache_key]

    @property
    def device(self) -> str:
        return str(self._device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self._tokenizer(texts)
        if tokens.device.type != self._device.type:
            tokens = tokens.to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image], *, micro_batch: int = 64) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        cap = max(1, min(256, int(micro_batch)))
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), cap):
            batch = images[start : start + cap]
            tensors = torch.stack([self._preprocess(im) for im in batch])
            if tensors.device.type != self._device.type:
                tensors = tensors.to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


def _resolve_device(pref: str, torch) -> Any:
    p = (pref or "auto").strip().lower()
    if p == "cpu":
        return torch.device("cpu")
    if p == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if p == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

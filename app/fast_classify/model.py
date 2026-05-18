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
        device = _resolve_device(settings.device, torch)
        is_cuda = device.type == "cuda"
        use_fp16 = bool(getattr(settings, "use_fp16", True)) and is_cuda
        cache_key = f"{settings.model_name}:{settings.pretrained}:{device}:{use_fp16}"
        with _model_lock:
            if cache_key not in _shared:
                if on_log:
                    on_log(
                        f"CLIP: загрузка {settings.model_name} ({settings.pretrained}) "
                        f"на {device}{' fp16' if use_fp16 else ''}…"
                    )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    settings.model_name,
                    pretrained=settings.pretrained,
                )
                tokenizer = open_clip.get_tokenizer(settings.model_name)
                model = model.to(device)
                if use_fp16:
                    model = model.half()
                if is_cuda:
                    try:
                        model = model.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                model.eval()
                _shared[cache_key] = (model, preprocess, tokenizer, device, use_fp16)
            self._model, self._preprocess, self._tokenizer, self._device, self._fp16 = _shared[cache_key]
        self._is_cuda = is_cuda
        self._embed_dtype = torch.float16 if self._fp16 else torch.float32

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def use_fp16(self) -> bool:
        return bool(self._fp16)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self._tokenizer(texts)
        if tokens.device.type != self._device.type:
            tokens = tokens.to(self._device)
        with torch.inference_mode():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().float().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image], *, micro_batch: int = 64) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        cap = max(1, min(1024, int(micro_batch)))
        if self._is_cuda and cap < 64:
            cap = 64
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), cap):
            batch = images[start : start + cap]
            tensors = torch.stack([self._preprocess(im) for im in batch])
            if self._is_cuda:
                tensors = tensors.pin_memory()
                tensors = tensors.to(self._device, non_blocking=True)
                tensors = tensors.to(memory_format=torch.channels_last)
            else:
                tensors = tensors.to(self._device)
            if self._fp16:
                tensors = tensors.half()
            with torch.inference_mode():
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.detach().float().cpu().numpy().astype(np.float32))
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

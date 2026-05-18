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
from app.fast_classify.weights import (
    clip_cache_dir,
    ensure_clip_weights_file,
    format_clip_load_error,
)

_model_lock = threading.Lock()
_shared: dict[str, Any] = {}


def clip_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("open_clip") is not None
        and importlib.util.find_spec("torch") is not None
    )


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
        cache_dir = str(clip_cache_dir())
        weights_path, pre_cfg = ensure_clip_weights_file(settings, on_log=on_log)
        cache_key = (
            f"{settings.model_name}:{weights_path}:{device}:{use_fp16}:{cache_dir}"
        )
        with _model_lock:
            if cache_key not in _shared:
                if on_log:
                    on_log(
                        f"CLIP: инициализация {settings.model_name} "
                        f"на {device}{' fp16' if use_fp16 else ''}…"
                    )
                load_kwargs: dict[str, Any] = {"cache_dir": cache_dir}
                mean = pre_cfg.get("mean")
                std = pre_cfg.get("std")
                if mean is not None:
                    load_kwargs["image_mean"] = mean
                if std is not None:
                    load_kwargs["image_std"] = std
                try:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        settings.model_name,
                        pretrained=str(weights_path),
                        **load_kwargs,
                    )
                except Exception as e:
                    raise RuntimeError(format_clip_load_error(e)) from e
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
        i = 0
        cur_cap = cap
        while i < len(images):
            batch = images[i : i + cur_cap]
            try:
                chunks.append(self._encode_batch(batch))
                i += len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_cap > 1:
                    new_cap = max(1, cur_cap // 2)
                    if self._on_log:
                        self._on_log(f"CLIP CUDA OOM: уменьшаю batch {cur_cap} -> {new_cap}")
                    cur_cap = new_cap
                    continue
                if self._on_log:
                    self._on_log("CLIP CUDA OOM на batch=1, фоллбэк на CPU для этого батча")
                chunks.append(self._encode_batch_cpu_fallback(batch))
                i += len(batch)
        return np.vstack(chunks)

    def _encode_batch(self, batch: list[Image.Image]) -> np.ndarray:
        import torch

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
        return feats.detach().float().cpu().numpy().astype(np.float32)

    def _encode_batch_cpu_fallback(self, batch: list[Image.Image]) -> np.ndarray:
        """Encode on CPU when CUDA OOM at batch size 1 (shared model moved under global lock)."""
        import torch

        if not batch:
            return np.zeros((0, self._text_dim()), dtype=np.float32)
        if self._device.type == "cpu":
            return self._encode_batch(batch)
        with _model_lock:
            cpu = torch.device("cpu")
            was_fp16 = self._fp16
            orig_device = self._device
            try:
                self._model = self._model.to(cpu)
                if was_fp16:
                    self._model = self._model.float()
                self._device = cpu
                self._is_cuda = False
                self._fp16 = False
                return self._encode_batch(batch)
            finally:
                self._model = self._model.to(orig_device)
                if was_fp16 and orig_device.type == "cuda":
                    self._model = self._model.half()
                self._device = orig_device
                self._is_cuda = orig_device.type == "cuda"
                self._fp16 = was_fp16

    def _text_dim(self) -> int:
        # CLIP visual and text share the embedding dimension; probe via a tiny text encode.
        try:
            v = self.encode_texts(["probe"])
            return int(v.shape[1])
        except Exception:
            return 512


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

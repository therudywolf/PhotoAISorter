# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Lazy-loaded OpenCLIP model for batch embedding."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from app.fast_classify.config import FastClassifySettings
from app.fast_classify.device_info import resolve_clip_device
from app.fast_classify.weights import (
    clip_cache_dir,
    ensure_clip_weights_file,
    format_clip_load_error,
)

_model_lock = threading.Lock()
_shared: dict[str, Any] = {}


def torch_compile_usable(torch: Any) -> bool:
    """
    Whether torch.compile can actually run here.

    torch.compile() returns a wrapper immediately but defers Inductor/Triton
    compilation to the first forward pass. Triton has no official Windows wheel,
    so on Windows that first forward crashes (not the compile() call) — which is
    exactly the "GPU immediately errors" symptom. Gate it off where Triton is
    missing so CUDA still runs in plain eager mode.
    """
    if not hasattr(torch, "compile"):
        return False
    if sys.platform.startswith("win"):
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


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
        device = resolve_clip_device(settings.device, torch, on_log=on_log)
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
                load_kwargs: dict[str, Any] = {
                    "cache_dir": cache_dir,
                    # OpenAI .pt checkpoints are TorchScript; PyTorch 2.6+ defaults weights_only=True.
                    "weights_only": False,
                }
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
                eager_model = model
                active_model = eager_model
                want_compile = is_cuda and bool(getattr(settings, "torch_compile", True))
                if want_compile and torch_compile_usable(torch):
                    try:
                        # "default" (not "reduce-overhead"): reduce-overhead uses CUDA
                        # graphs that break on the varying batch sizes the OOM fallback
                        # produces. The first forward still compiles lazily, so encode_*
                        # falls back to eager_model if that compilation fails.
                        active_model = torch.compile(eager_model, mode="default")
                        if on_log:
                            on_log("CLIP: torch.compile включён (CUDA).")
                    except Exception as e:
                        active_model = eager_model
                        if on_log:
                            on_log(f"CLIP: torch.compile пропущен: {e}")
                elif want_compile and on_log:
                    on_log("CLIP: torch.compile недоступен (нет Triton/Windows) — обычный режим CUDA.")
                _shared[cache_key] = (
                    active_model,
                    eager_model,
                    preprocess,
                    tokenizer,
                    device,
                    use_fp16,
                )
            (
                self._model,
                self._eager_model,
                self._preprocess,
                self._tokenizer,
                self._device,
                self._fp16,
            ) = _shared[cache_key]
        self._compiled = self._model is not self._eager_model
        self._is_cuda = is_cuda
        self._embed_dtype = torch.float16 if self._fp16 else torch.float32

    def _forward(self, method: str, inp: Any) -> Any:
        """
        Run encode_image / encode_text, surviving a deferred torch.compile failure.

        OOM is re-raised for the batch-shrink handler. Any other error from the
        compiled module (e.g. an Inductor/Triton backend failure on the first
        forward) drops this embedder to the eager model permanently and retries,
        so a bad compile degrades to plain CUDA instead of aborting the sort.
        """
        import torch

        try:
            return getattr(self._model, method)(inp)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            if self._model is self._eager_model:
                raise
            if self._on_log:
                self._on_log(
                    f"CLIP: torch.compile дал сбой ({type(e).__name__}); переключаюсь на eager-режим."
                )
            self._model = self._eager_model
            self._compiled = False
            return getattr(self._eager_model, method)(inp)

    def _disable_cuda_permanently(self, reason: str) -> None:
        """
        Move this embedder to CPU for the rest of its life after a fatal CUDA error.

        A non-OOM CUDA failure in an eager forward (cuBLAS/cuDNN status, an fp16 op
        gap, a driver fault) is not recoverable by retrying on the GPU — every later
        batch would raise again, and the per-batch CPU round-trip is expensive. So the
        switch is sticky: the shared module is moved to CPU once, under the global lock.
        """
        import torch

        with _model_lock:
            if self._device.type != "cuda":
                return
            cpu = torch.device("cpu")
            model = self._eager_model.to(cpu)
            if self._fp16:
                model = model.float()
            self._eager_model = model
            self._model = model
            self._device = cpu
            self._is_cuda = False
            self._fp16 = False
            self._compiled = False
        if self._on_log:
            self._on_log(f"CLIP: ошибка CUDA ({reason}); до конца работы использую CPU.")

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def use_fp16(self) -> bool:
        return bool(self._fp16)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        try:
            return self._encode_texts_once(texts)
        except RuntimeError as e:
            # First CLIP forward happens here during FastClassifier init; degrading
            # to CPU keeps the sort alive instead of aborting on a broken GPU.
            if not self._is_cuda:
                raise
            self._disable_cuda_permanently(type(e).__name__)
            return self._encode_texts_once(texts)

    def _encode_texts_once(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self._tokenizer(texts)
        if tokens.device.type != self._device.type:
            tokens = tokens.to(self._device)
        with torch.inference_mode():
            feats = self._forward("encode_text", tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().float().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image], *, micro_batch: int = 64) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        cap = max(1, min(1024, int(micro_batch)))
        if self._is_cuda:
            if "ViT-L" in self._settings.model_name and cap > 48:
                cap = 48
            elif cap < 32:
                cap = 32
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
            except RuntimeError as e:
                # Non-OOM CUDA failure (cuBLAS/cuDNN, fp16 op gap, driver fault):
                # switch to CPU once and re-encode this same batch there.
                if not self._is_cuda:
                    raise
                self._disable_cuda_permanently(type(e).__name__)
                torch.cuda.empty_cache()
                continue
        return np.vstack(chunks)

    def _encode_batch(self, batch: list[Image.Image]) -> np.ndarray:
        import torch

        tensors = torch.stack([self._preprocess(im) for im in batch])
        if self._is_cuda:
            # Lay channels_last out on the CPU tensor before the async H2D copy, and
            # tolerate pinned-host-memory exhaustion (skip pinning rather than crash).
            tensors = tensors.contiguous(memory_format=torch.channels_last)
            try:
                tensors = tensors.pin_memory()
            except RuntimeError:
                pass
            tensors = tensors.to(self._device, non_blocking=True)
        else:
            tensors = tensors.to(self._device)
        if self._fp16:
            tensors = tensors.half()
        with torch.inference_mode():
            feats = self._forward("encode_image", tensors)
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
            orig_model = self._model
            orig_compiled = self._compiled
            # Use the eager module on CPU: a CUDA-compiled graph cannot run on CPU.
            try:
                eager_cpu = self._eager_model.to(cpu)
                if was_fp16:
                    eager_cpu = eager_cpu.float()
                self._model = eager_cpu
                self._eager_model = eager_cpu
                self._device = cpu
                self._is_cuda = False
                self._fp16 = False
                self._compiled = False
                return self._encode_batch(batch)
            finally:
                restored = self._eager_model.to(orig_device)
                if was_fp16 and orig_device.type == "cuda":
                    restored = restored.half()
                self._eager_model = restored
                self._model = orig_model if orig_compiled else restored
                self._device = orig_device
                self._is_cuda = orig_device.type == "cuda"
                self._fp16 = was_fp16
                self._compiled = orig_compiled

    def _text_dim(self) -> int:
        # CLIP visual and text share the embedding dimension; probe via a tiny text encode.
        try:
            v = self.encode_texts(["probe"])
            return int(v.shape[1])
        except Exception:
            return 512



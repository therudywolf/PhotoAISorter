# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""torch.compile gating and eager fallback (the GPU "immediately errors" fix)."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
from app.fast_classify.model import ClipEmbedder, torch_compile_usable


class _FakeTorch:
    """Minimal stand-in exposing the attributes torch_compile_usable reads."""

    def compile(self, *_a: object, **_k: object) -> object:  # pragma: no cover - presence only
        return None


def test_compile_disabled_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert torch_compile_usable(_FakeTorch()) is False


def test_compile_disabled_without_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.setattr("builtins.__import__", _blocking_import("triton"))
    assert torch_compile_usable(_FakeTorch()) is False


def test_compile_enabled_with_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "triton", ModuleType("triton"))
    assert torch_compile_usable(_FakeTorch()) is True


def test_compile_disabled_when_no_compile_attr() -> None:
    class NoCompile:
        pass

    assert torch_compile_usable(NoCompile()) is False


def _blocking_import(blocked: str):
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == blocked:
            raise ImportError(f"{blocked} unavailable")
        return real_import(name, *args, **kwargs)

    return fake_import


class _BadModule:
    """Compiled module whose forward dies the way a Triton backend failure would."""

    def encode_image(self, _inp: object) -> object:
        raise RuntimeError("BackendCompilerFailed: triton not found")

    def encode_text(self, _inp: object) -> object:
        raise RuntimeError("BackendCompilerFailed: triton not found")


class _GoodModule:
    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    def encode_image(self, inp: object) -> object:
        self.image_calls += 1
        return inp

    def encode_text(self, inp: object) -> object:
        self.text_calls += 1
        return inp


def _embedder_with(model: object, eager: object) -> ClipEmbedder:
    emb = ClipEmbedder.__new__(ClipEmbedder)
    emb._model = model
    emb._eager_model = eager
    emb._compiled = model is not eager
    emb._on_log = None
    return emb


def test_forward_falls_back_to_eager_on_compile_error() -> None:
    eager = _GoodModule()
    emb = _embedder_with(_BadModule(), eager)

    sentinel = object()
    out = emb._forward("encode_image", sentinel)

    assert out is sentinel
    assert emb._model is eager  # permanently switched off the broken compiled module
    assert emb._compiled is False
    assert eager.image_calls == 1


def test_forward_reraises_when_already_eager() -> None:
    eager = _BadModule()
    emb = _embedder_with(eager, eager)
    with pytest.raises(RuntimeError):
        emb._forward("encode_text", object())


def test_forward_reraises_cuda_oom() -> None:
    import torch

    class _OomModule:
        def encode_image(self, _inp: object) -> object:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    eager = _GoodModule()
    emb = _embedder_with(_OomModule(), eager)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        emb._forward("encode_image", object())
    # OOM must NOT trigger the eager swap — the batch-shrink handler owns that path.
    assert emb._compiled is True
    assert eager.image_calls == 0


class _Movable:
    """Module whose .to()/.float() are no-ops returning self (for the CPU switch)."""

    def to(self, *_a: object, **_k: object) -> "_Movable":
        return self

    def float(self) -> "_Movable":
        return self


def test_disable_cuda_permanently_switches_state() -> None:
    import torch

    emb = ClipEmbedder.__new__(ClipEmbedder)
    moved = _Movable()
    emb._eager_model = moved
    emb._model = moved
    emb._device = torch.device("cuda")
    emb._is_cuda = True
    emb._fp16 = True
    emb._compiled = True
    emb._on_log = None

    emb._disable_cuda_permanently("CUBLAS_STATUS_NOT_INITIALIZED")

    assert emb._device.type == "cpu"
    assert emb._is_cuda is False
    assert emb._fp16 is False
    assert emb._compiled is False
    # Idempotent: a second call on an already-CPU embedder is a no-op, not an error.
    emb._disable_cuda_permanently("again")
    assert emb._device.type == "cpu"


def test_encode_texts_degrades_to_cpu_on_cuda_runtime_error() -> None:
    emb = ClipEmbedder.__new__(ClipEmbedder)
    emb._is_cuda = True
    state = {"calls": 0, "disabled": False}

    def fake_once(_texts: list[str]) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("cuBLAS error")
        return "cpu-result"

    def fake_disable(_reason: str) -> None:
        state["disabled"] = True
        emb._is_cuda = False

    emb._encode_texts_once = fake_once  # type: ignore[method-assign]
    emb._disable_cuda_permanently = fake_disable  # type: ignore[method-assign]

    assert emb.encode_texts(["hello"]) == "cpu-result"
    assert state == {"calls": 2, "disabled": True}


def test_encode_texts_reraises_runtime_error_on_cpu() -> None:
    emb = ClipEmbedder.__new__(ClipEmbedder)
    emb._is_cuda = False

    def fake_once(_texts: list[str]) -> str:
        raise RuntimeError("genuine cpu bug")

    emb._encode_texts_once = fake_once  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="genuine cpu bug"):
        emb.encode_texts(["hello"])


def test_encode_images_sticky_cpu_on_cuda_runtime_error() -> None:
    import numpy as np
    from app.fast_classify.config import FastClassifySettings
    from PIL import Image

    emb = ClipEmbedder.__new__(ClipEmbedder)
    emb._is_cuda = True
    emb._settings = FastClassifySettings()
    emb._on_log = None
    state = {"calls": 0}

    def fake_encode_batch(batch: list[object]) -> np.ndarray:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("cuda error: device-side assert")
        return np.ones((len(batch), 4), dtype=np.float32)

    def fake_disable(_reason: str) -> None:
        emb._is_cuda = False

    emb._encode_batch = fake_encode_batch  # type: ignore[method-assign]
    emb._disable_cuda_permanently = fake_disable  # type: ignore[method-assign]

    imgs = [Image.new("RGB", (8, 8)) for _ in range(2)]
    out = emb.encode_images(imgs, micro_batch=2)

    assert out.shape == (2, 4)
    assert emb._is_cuda is False  # switched to CPU and retried the same batch

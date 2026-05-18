# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP model load kwargs and device resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.fast_classify.config import FastClassifySettings
from app.fast_classify.device_info import resolve_clip_device
from app.fast_classify.model import ClipEmbedder


def test_create_model_uses_weights_only_false(monkeypatch: object, tmp_path: Path) -> None:
    import torch

    from app.fast_classify import model as model_mod

    captured: dict = {}

    def fake_create(*_a, **kwargs: object) -> tuple:
        captured.update(kwargs)
        m = MagicMock()
        m.eval.return_value = m
        m.to.return_value = m
        m.half.return_value = m
        return m, None, None

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")

    model_mod._shared.clear()
    monkeypatch.setattr("app.fast_classify.model.clip_available", lambda: True)
    monkeypatch.setattr(
        "app.fast_classify.model.ensure_clip_weights_file",
        lambda *_a, **_k: (weights, {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)}),
    )
    monkeypatch.setattr(
        "app.fast_classify.model.resolve_clip_device",
        lambda *_a, **_k: torch.device("cpu"),
    )
    monkeypatch.setattr("open_clip.create_model_and_transforms", fake_create)
    monkeypatch.setattr("open_clip.get_tokenizer", lambda *_a: MagicMock())
    monkeypatch.setattr("app.fast_classify.weights.install_clip_download_patch", lambda: None)

    ClipEmbedder(FastClassifySettings(), on_log=None)
    assert captured.get("weights_only") is False


def test_resolve_clip_device_auto_cuda_when_available() -> None:
    import torch as torch_real

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070"
    torch.device = torch_real.device
    dev = resolve_clip_device("auto", torch)
    assert dev.type == "cuda"


def test_resolve_clip_device_honors_cpu_setting() -> None:
    import torch as torch_real

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.device = torch_real.device
    dev = resolve_clip_device("cpu", torch)
    assert dev.type == "cpu"

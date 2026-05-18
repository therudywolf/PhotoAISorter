# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""FastClassifier with mocked CLIP embedder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.fast_classify.config import FastClassifySettings
from app.fast_classify.pipeline import FastClassifier
from app.fast_classify.registry import clear_classifier_cache, get_classifier
from app.tag_config import ResolvedTagConfig, TagMode


class _MockEmbedder:
    device = "cpu"

    def __init__(self, settings: FastClassifySettings | None = None, *, on_log=None) -> None:
        self._dog = np.array([1.0, 0.0], dtype=np.float32)
        self._cat = np.array([0.0, 1.0], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            low = t.lower()
            if "my dog" in low or "black lab" in low:
                out.append(self._dog)
            else:
                out.append(self._cat)
        return np.stack(out, axis=0)

    def encode_images(self, images: list[Image.Image], *, micro_batch: int = 64) -> np.ndarray:
        feats = []
        for im in images:
            r, _g, _b = im.convert("RGB").getpixel((0, 0))
            if r < 128:
                feats.append(self._dog)
            else:
                feats.append(self._cat)
        return np.stack(feats, axis=0)


def _cfg() -> ResolvedTagConfig:
    return ResolvedTagConfig(
        mode=TagMode.HYBRID,
        categories=("my_dog", "cat", "uncategorized"),
        prompts={"my_dog": "black lab", "cat": "other cat", "uncategorized": "unknown"},
        whitelist=frozenset({"my_dog", "cat", "uncategorized"}),
    )


def test_fast_classifier_batch_vectorized(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("app.fast_classify.pipeline.clip_available", lambda: True)
    monkeypatch.setattr("app.fast_classify.pipeline.ClipEmbedder", _MockEmbedder)
    monkeypatch.setattr("app.fast_classify.pipeline.ensure_refs_layout", lambda **_: None)

    dark_im = Image.new("RGB", (64, 64), (10, 10, 10))
    bright_im = Image.new("RGB", (64, 64), (250, 250, 250))
    dark = tmp_path / "dark.png"
    bright = tmp_path / "bright.png"

    clf = FastClassifier(_cfg(), FastClassifySettings(confidence_threshold=0.1))
    r_dark = clf.classify_image(dark, dark_im)
    r_bright = clf.classify_image(bright, bright_im)
    assert r_dark.category == "my_dog", r_dark.candidates
    assert r_bright.category == "cat", r_bright.candidates

    batch = clf.classify_batch([dark, bright])
    assert len(batch) == 2


def test_classifier_registry_reuses_instance(monkeypatch: object) -> None:
    monkeypatch.setattr("app.fast_classify.pipeline.clip_available", lambda: True)
    monkeypatch.setattr("app.fast_classify.pipeline.ClipEmbedder", _MockEmbedder)
    monkeypatch.setattr("app.fast_classify.pipeline.ensure_refs_layout", lambda **_: None)
    clear_classifier_cache()
    settings = FastClassifySettings()
    cfg = _cfg()
    a = get_classifier(cfg, settings)
    b = get_classifier(cfg, settings)
    assert a is not None and b is not None
    assert a is b
    clear_classifier_cache()


def test_heuristic_short_circuits_clip(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("app.fast_classify.pipeline.clip_available", lambda: True)
    embedder = MagicMock()
    monkeypatch.setattr("app.fast_classify.pipeline.ClipEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr("app.fast_classify.pipeline.ensure_refs_layout", lambda **_: None)

    p = tmp_path / "screen_shot_test.png"
    Image.new("RGB", (400, 800), (30, 30, 30)).save(p)
    wl = frozenset({"screenshot", "uncategorized"})
    cfg = ResolvedTagConfig(
        mode=TagMode.HYBRID,
        categories=tuple(wl),
        prompts={"screenshot": "ui", "uncategorized": "x"},
        whitelist=wl,
    )
    clf = FastClassifier(cfg, FastClassifySettings())
    result = clf.classify_path(p)
    embedder.encode_images.assert_not_called()
    assert result.category == "screenshot"

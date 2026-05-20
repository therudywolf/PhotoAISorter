# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""VLM/CLIP merge helper for hybrid sort."""

from __future__ import annotations

from app.classification_result import ClassificationResult
from app.constants import UNCATEGORIZED
from app.sort_hybrid import _merge_vlm_with_clip


def _res(cat: str, conf: float, *, review: bool = False) -> ClassificationResult:
    return ClassificationResult(cat, [cat], conf, "test", review, "")


def test_merge_keeps_clip_when_vlm_uncategorized() -> None:
    clip = _res("my_dog", 0.72)
    vlm = _res(UNCATEGORIZED, 0.5)
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "my_dog"


def test_merge_keeps_clip_when_vlm_weaker_but_clip_confident() -> None:
    clip = _res("my_dog", 0.8, review=True)
    vlm = _res("cat", 0.6)
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "my_dog"


def test_merge_takes_vlm_when_clearly_better() -> None:
    clip = _res("my_dog", 0.4, review=True)
    vlm = _res("cat", 0.92)
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "cat"


def test_merge_keeps_clip_on_weak_vlm_default_confidence() -> None:
    clip = _res("my_dog", 0.62, review=True)
    vlm = _res("cat", 0.75)
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "my_dog"


def test_vlm_fallback_trigger_skips_borderline_review() -> None:
    from app.sort_hybrid import _clip_needs_vlm_fallback

    borderline = _res("my_dog", 0.58, review=True)
    assert not _clip_needs_vlm_fallback(borderline, confidence_threshold=0.55)
    low = _res("my_dog", 0.35, review=True)
    assert _clip_needs_vlm_fallback(low, confidence_threshold=0.55)

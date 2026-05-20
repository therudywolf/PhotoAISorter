# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Regression tests for filename heuristics in fast_classify."""

from __future__ import annotations

from pathlib import Path

from app.fast_classify.heuristics import heuristic_tag
from PIL import Image

_WL = frozenset({"screenshot", "document", "receipt_barcode", "iam", "my_dog"})


def test_phone_aspect_photo_is_not_a_screenshot() -> None:
    """A vertical phone photo must not be heuristically tagged as a screenshot."""
    im = Image.new("RGB", (1080, 1920), color=(40, 90, 120))
    assert heuristic_tag(Path("IMG_20220512.jpg"), im, whitelist=_WL) is None


def test_bright_large_photo_is_not_a_document() -> None:
    """A bright photo (snow, white wall) must not be tagged as a document."""
    im = Image.new("RGB", (1600, 1200), color=(245, 245, 245))
    assert heuristic_tag(Path("snow_field.jpg"), im, whitelist=_WL) is None


def test_filename_screenshot_still_detected() -> None:
    im = Image.new("RGB", (1080, 1920), color=(30, 30, 30))
    res = heuristic_tag(Path("Screenshot_2024-01-01.png"), im, whitelist=_WL)
    assert res is not None and res[0] == "screenshot"

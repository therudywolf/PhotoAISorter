# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Multi-crop views for CLIP test-time augmentation (accuracy on HDD-sized libraries)."""

from __future__ import annotations

from PIL import Image


def multi_crop_views(im: Image.Image, *, views: int = 5) -> list[Image.Image]:
    """Return 1 (center) or 5 crops: full frame + four 88% corner windows."""
    n = max(1, min(5, int(views)))
    if n == 1:
        return [im]
    w, h = im.size
    if w < 32 or h < 32:
        return [im]
    out: list[Image.Image] = [im]
    cw = max(24, int(w * 0.88))
    ch = max(24, int(h * 0.88))
    boxes = (
        (0, 0, cw, ch),
        (w - cw, 0, w, ch),
        (0, h - ch, cw, h),
        (w - cw, h - ch, w, h),
    )
    for box in boxes[: n - 1]:
        out.append(im.crop(box))
    return out

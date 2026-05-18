# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Multi-crop views for CLIP test-time augmentation (accuracy on HDD-sized libraries)."""

from __future__ import annotations

from PIL import Image


def multi_crop_views(im: Image.Image, *, views: int = 5) -> list[Image.Image]:
    """
    Return 1–12 crops: full frame, corners, center zoom, and edge bands (portraits).
    More views = higher accuracy, slower encode.
    """
    n = max(1, min(12, int(views)))
    if n == 1:
        return [im]
    w, h = im.size
    if w < 32 or h < 32:
        return [im]

    out: list[Image.Image] = [im]

    def _crop(box: tuple[int, int, int, int]) -> None:
        if len(out) >= n:
            return
        x0, y0, x1, y1 = box
        if x1 > x0 and y1 > y0:
            out.append(im.crop((x0, y0, x1, y1)))

    cw = max(24, int(w * 0.88))
    ch = max(24, int(h * 0.88))
    for box in (
        (0, 0, cw, ch),
        (w - cw, 0, w, ch),
        (0, h - ch, cw, h),
        (w - cw, h - ch, w, h),
    ):
        _crop(box)

    cz = max(24, int(min(w, h) * 0.72))
    cx0 = (w - cz) // 2
    cy0 = (h - cz) // 2
    _crop((cx0, cy0, cx0 + cz, cy0 + cz))

    # horizontal bands (faces, pets)
    band_h = max(24, int(h * 0.55))
    y_mid = max(0, (h - band_h) // 2)
    _crop((0, y_mid, w, y_mid + band_h))

    band_w = max(24, int(w * 0.55))
    x_mid = max(0, (w - band_w) // 2)
    _crop((x_mid, 0, x_mid + band_w, h))

    # extra corner zooms at 76% for ultra (9+ views)
    cw2 = max(24, int(w * 0.76))
    ch2 = max(24, int(h * 0.76))
    for box in (
        (0, 0, cw2, ch2),
        (w - cw2, h - ch2, w, h),
    ):
        _crop(box)

    return out[:n]

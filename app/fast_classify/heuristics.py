# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Fast filename / image-stat heuristics before CLIP."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

_SCREENSHOT_NAME_RE = re.compile(
    r"(?i)(screenshot|screen[_-]?shot|screencap|capture d''écran|снимок экрана|скрин)"
)
_DOCUMENT_NAME_RE = re.compile(
    r"(?i)(scan|scanned|document|invoice|receipt|чек|квитанц|договор|contract)"
)
_RECEIPT_NAME_RE = re.compile(r"(?i)(receipt|barcode|qr[_-]?code|чек|квитанц|invoice)")


def _light_background_ratio(im: Image.Image, *, sample: int = 96) -> float:
    thumb = im.convert("RGB").resize((sample, sample), Image.Resampling.BILINEAR)
    pixels = list(thumb.getdata())
    light = sum(1 for r, g, b in pixels if r > 210 and g > 210 and b > 210)
    return light / max(1, len(pixels))


def heuristic_tag(path: Path, im: Image.Image, *, whitelist: frozenset[str]) -> tuple[str, float, str] | None:
    """Return (tag, confidence, reason) when a cheap rule is very confident."""
    name = path.name
    w, h = im.size
    if w < 8 or h < 8:
        return None

    if "screenshot" in whitelist and _SCREENSHOT_NAME_RE.search(name):
        return ("screenshot", 0.92, "filename_screenshot")

    if "receipt_barcode" in whitelist and _RECEIPT_NAME_RE.search(name):
        return ("receipt_barcode", 0.9, "filename_receipt")

    if "document" in whitelist and _DOCUMENT_NAME_RE.search(name):
        return ("document", 0.88, "filename_document")

    aspect = w / float(h) if h else 1.0
    if "screenshot" in whitelist and min(w, h) >= 400:
        for target in (9 / 16, 9 / 19.5, 3 / 4, 16 / 9):
            if abs(aspect - target) < 0.04 or abs(aspect - 1 / target) < 0.04:
                if _light_background_ratio(im) < 0.72:
                    return ("screenshot", 0.8, "aspect_ui_like")

    if "document" in whitelist or "receipt_barcode" in whitelist:
        light = _light_background_ratio(im)
        if light > 0.62 and max(w, h) >= 500:
            tag = (
                "receipt_barcode"
                if "receipt_barcode" in whitelist and _RECEIPT_NAME_RE.search(name)
                else "document"
            )
            if tag in whitelist:
                return (tag, 0.78, "bright_paper_like")

    return None

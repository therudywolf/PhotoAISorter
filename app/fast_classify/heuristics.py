# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Fast filename / image-stat heuristics before CLIP."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

_SCREENSHOT_NAME_RE = re.compile(
    r"(?i)(screenshot|screen[_-]?shot|screencap|capture d['']?écran|снимок экрана|скрин)"
)
_DOCUMENT_NAME_RE = re.compile(
    r"(?i)(scan|scanned|document|invoice|receipt|чек|квитанц|договор|contract)"
)
_RECEIPT_NAME_RE = re.compile(r"(?i)(receipt|barcode|qr[_-]?code|чек|квитанц|invoice)")


def heuristic_tag(path: Path, im: Image.Image, *, whitelist: frozenset[str]) -> tuple[str, float, str] | None:
    """Return (tag, confidence, reason) for cheap, reliable filename-only rules.

    Image-statistic heuristics (aspect ratio, brightness) were removed: a
    vertical phone photo shares its aspect ratio with a screenshot, so they
    routed ordinary photos into screenshot/document. CLIP decides those now.
    """
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

    return None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Image hashing and in-memory resize for vision API."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.constants import JPEG_QUALITY, MAX_IMAGE_SIDE

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pil_to_rgb(im: Image.Image) -> Image.Image:
    # GIF/PNG palette: через RGBA — стабильнее, чем прямой P→RGB (альфа, подложка).
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGBA")
        background = Image.new("RGB", im.size, (255, 255, 255))
        alpha = im.split()[-1]
        background.paste(im, mask=alpha)
        return background
    return im.convert("RGB")


def pil_image_to_jpeg_data_uri(
    im: Image.Image,
    *,
    max_side: int = MAX_IMAGE_SIDE,
    quality: int = JPEG_QUALITY,
) -> str:
    """Resize PIL image to max side, encode JPEG, return data URI."""
    im = _pil_to_rgb(im)
    w, h = im.size
    long_side = max(w, h)
    max_side = max(64, int(max_side))
    if long_side > max_side:
        scale = max_side / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=max(35, min(95, int(quality))), optimize=True)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def video_contact_sheet_data_uri(
    frames: list[Image.Image],
    *,
    max_frames: int = 3,
    tile_max_side: int = 512,
    output_max_side: int = 1024,
    quality: int = 76,
) -> str:
    """Pack chronological video frames into one compact JPEG for single-image vision APIs."""
    selected = [im for im in frames[: max(1, int(max_frames))] if im is not None]
    if not selected:
        raise ValueError("no frames")
    tiles: list[Image.Image] = []
    tile_max_side = max(128, min(768, int(tile_max_side)))
    for im in selected:
        rgb = _pil_to_rgb(im)
        w, h = rgb.size
        long_side = max(w, h)
        if long_side > tile_max_side:
            scale = tile_max_side / float(long_side)
            rgb = rgb.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.LANCZOS,
            )
        tiles.append(rgb.copy())
    gap = 8
    width = sum(im.size[0] for im in tiles) + gap * (len(tiles) - 1)
    height = max(im.size[1] for im in tiles)
    sheet = Image.new("RGB", (width, height), (8, 10, 12))
    x = 0
    for tile in tiles:
        y = (height - tile.size[1]) // 2
        sheet.paste(tile, (x, y))
        x += tile.size[0] + gap
    return pil_image_to_jpeg_data_uri(sheet, max_side=output_max_side, quality=quality)


def load_image_rgb(path: Path, *, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Load still image as RGB, optionally downscale for local ML models."""
    with Image.open(path) as im:
        im = _pil_to_rgb(im)
        w, h = im.size
        long_side = max(w, h)
        cap = max(64, int(max_side))
        if long_side > cap:
            scale = cap / float(long_side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        return im.copy()


def image_to_jpeg_base64_data_uri(path: Path) -> str:
    """
    Load image, convert to RGB if needed, resize so max side <= MAX_IMAGE_SIDE,
    encode as JPEG in memory, return data URI for OpenAI-compatible vision.
    """
    with Image.open(path) as im:
        return pil_image_to_jpeg_data_uri(im)


def _probe_font(size: int) -> ImageFont.ImageFont:
    candidates: list[str | Path] = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
                Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf",
            ]
        )
    elif sys.platform == "darwin":
        candidates.append("/System/Library/Fonts/Supplemental/Arial.ttf")
    candidates.append("arial.ttf")
    for p in candidates:
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()


def tiny_jpeg_data_uri() -> str:
    """Minimal 1x1 JPEG (legacy fallback)."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (245, 245, 245)).save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def vision_test_card_data_uri() -> str:
    """
    Synthetic 512x512 test image: shapes + text so vision models have real content to describe.
    Used for API vision probe (avoids 1x1 pixel and heavy classification prompt).
    """
    w, h = 512, 512
    im = Image.new("RGB", (w, h), (32, 36, 48))
    draw = ImageDraw.Draw(im)
    draw.rectangle([24, 24, 220, 220], fill=(210, 70, 70), outline=(255, 255, 255), width=4)
    draw.ellipse([260, 40, 490, 270], fill=(55, 170, 95), outline=(255, 255, 255), width=4)
    draw.polygon([(256, 300), (120, 480), (392, 480)], fill=(70, 130, 210), outline=(255, 255, 255), width=3)
    font = _probe_font(32)
    draw.text((48, 300), "VISION TEST", fill=(255, 220, 60), font=font)
    draw.text((48, 360), "shapes: red square, green circle, blue triangle", fill=(220, 220, 230), font=font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

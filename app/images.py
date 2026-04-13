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
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        background = Image.new("RGB", im.size, (255, 255, 255))
        alpha = im.split()[-1] if im.mode == "RGBA" else None
        background.paste(im, mask=alpha)
        return background
    return im.convert("RGB")


def pil_image_to_jpeg_data_uri(im: Image.Image) -> str:
    """Resize PIL image to max side, encode JPEG, return data URI."""
    im = _pil_to_rgb(im)
    w, h = im.size
    long_side = max(w, h)
    if long_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


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

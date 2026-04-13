"""Shared thumbnail helpers for duplicate-finder UI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from app.constants import GIF_EXTENSION, VIDEO_EXTENSIONS
from app.video_frames import is_animated_gif, preview_frame_for_thumbnail


def thumb_ctk(
    path: Path, size: tuple[int, int] = (88, 88), *, on_video_log: Callable[[str], None] | None = None
) -> ctk.CTkImage | None:
    log = on_video_log or (lambda _m: None)
    try:
        suf = path.suffix.lower()
        if suf in VIDEO_EXTENSIONS or (suf == GIF_EXTENSION and is_animated_gif(path)):
            pil = preview_frame_for_thumbnail(path, on_log=log)
            if pil is None:
                return None
            im = pil.convert("RGB")
            im.thumbnail(size, Image.Resampling.LANCZOS)
            owned = im.copy()
            return ctk.CTkImage(light_image=owned, dark_image=owned, size=owned.size)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(size, Image.Resampling.LANCZOS)
            owned = im.copy()
            return ctk.CTkImage(light_image=owned, dark_image=owned, size=owned.size)
    except OSError:
        return None

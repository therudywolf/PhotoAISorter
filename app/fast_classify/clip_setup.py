# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Prepare CLIP for hybrid mode: deps check, weight download, human-readable status."""

from __future__ import annotations

from collections.abc import Callable

from app.fast_classify.config import FastClassifySettings, load_fast_classify_settings
from app.fast_classify.model import clip_available, missing_clip_message
from app.fast_classify.weights import clip_cache_dir, ensure_clip_weights_file
from app.tag_mode_ui import CLIP_QUALITY_LABELS


def describe_clip_settings(settings: FastClassifySettings) -> str:
    q = CLIP_QUALITY_LABELS.get(settings.quality, settings.quality)
    crop = (
        f", {settings.multi_crop_views} кропов"
        if settings.multi_crop and settings.multi_crop_views > 1
        else ""
    )
    return (
        f"Профиль «{q}»: {settings.model_name}, {settings.image_max_side}px{crop}, "
        f"батч {settings.batch_size}."
    )


def build_fast_classify_gui_block(
    *,
    quality_key: str,
    device_key: str,
    vlm_fallback: bool,
) -> dict:
    """Minimal gui_settings fast_classify block (profile fields applied on load)."""
    return {
        "quality": quality_key,
        "device": device_key,
        "vlm_fallback": bool(vlm_fallback),
    }


def resolve_gui_fast_classify_settings(
    gui_settings: dict | None,
  *,
  quality_key: str = "ultra",
  device_key: str = "auto",
  vlm_fallback: bool = True,
) -> FastClassifySettings:
    block = build_fast_classify_gui_block(
        quality_key=quality_key,
        device_key=device_key,
        vlm_fallback=vlm_fallback,
    )
    if isinstance(gui_settings, dict) and isinstance(gui_settings.get("fast_classify"), dict):
        merged = {**gui_settings["fast_classify"], **block}
    else:
        merged = block
    return load_fast_classify_settings({"fast_classify": merged})


def ensure_clip_ready(
    settings: FastClassifySettings | None = None,
    *,
    on_log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Verify torch/open_clip and download model weights if missing.
    Returns (ok, status_line_for_GUI).
    """
    if not clip_available():
        msg = missing_clip_message()
        if on_log:
            on_log(msg)
        return False, msg

    s = settings or load_fast_classify_settings()
    if on_log:
        on_log(
            f"CLIP: подготовка {s.model_name} ({s.pretrained}), "
            f"кэш весов: {clip_cache_dir()}"
        )
    try:
        path, _ = ensure_clip_weights_file(s, on_log=on_log)
    except Exception as e:
        err = f"CLIP: не удалось загрузить веса — {e}"
        if on_log:
            on_log(err)
        return False, err

    line = f"{describe_clip_settings(s)} Файл: {path.name}."
    if on_log:
        on_log(f"CLIP: готово. {line}")
    return True, line

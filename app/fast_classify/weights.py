# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP weight download helpers (URL-first, app-local cache)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.fast_classify.config import FastClassifySettings

_patch_installed = False


def clip_cache_dir() -> Path:
    from app.db import default_db_path

    path = default_db_path().parent / "clip_weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_clip_download_patch() -> None:
    """Prefer OpenAI/direct URLs over Hugging Face (HF often fails on Windows GUI runs)."""
    global _patch_installed
    if _patch_installed:
        return
    _patch_installed = True

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import open_clip.factory as factory_mod
    import open_clip.pretrained as pretrained_mod

    original: Callable[..., Any] = pretrained_mod.download_pretrained

    def download_pretrained_robust(
        cfg: dict[str, Any],
        prefer_hf_hub: bool = True,
        cache_dir: str | None = None,
    ) -> str:
        del prefer_hf_hub  # always try direct URL before HF hub
        if not cfg:
            return ""
        target_dir = cache_dir or str(clip_cache_dir())
        errors: list[str] = []
        for use_hf, label in ((False, "прямая ссылка"), (True, "Hugging Face")):
            try:
                path = original(cfg, prefer_hf_hub=use_hf, cache_dir=target_dir)
                if path and os.path.isfile(path):
                    return path
            except Exception as e:
                errors.append(f"{label}: {e}")
        raise RuntimeError(
            "Не удалось загрузить веса CLIP (" + "; ".join(errors) + f"). Каталог: {target_dir}"
        )

    pretrained_mod.download_pretrained = download_pretrained_robust
    factory_mod.download_pretrained = download_pretrained_robust


def resolve_pretrained_arg(settings: FastClassifySettings) -> str:
    """Tag or local file path passed to open_clip.create_model_and_transforms."""
    custom = (getattr(settings, "weights_path", "") or "").strip()
    if custom:
        path = Path(custom).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"Файл весов CLIP не найден: {path}")
    return settings.pretrained


def format_clip_load_error(exc: BaseException) -> str:
    cache = clip_cache_dir()
    return (
        f"Ошибка загрузки CLIP: {exc}\n"
        f"Кэш весов: {cache}\n"
        "Проверьте интернет и доступ на запись. При повторе скачайте ViT-B-32.pt "
        "(тег openai) вручную и укажите путь в настройках fast_classify.weights_path."
    )

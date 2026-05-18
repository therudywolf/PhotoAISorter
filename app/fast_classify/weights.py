# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP weight download (GUI-safe: no HF-first, no tqdm on missing stdout)."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.fast_classify.config import FastClassifySettings

_patch_installed = False
_MIN_OPENAI_VITB32_BYTES = 300_000_000


def clip_cache_dir() -> Path:
    from app.paths import clip_weights_dir, migrate_roaming_clip_data

    migrate_roaming_clip_data()
    path = clip_weights_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_download_env() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")


def install_clip_download_patch() -> None:
    """Fallback: if open_clip still downloads internally, try URL before HF."""
    global _patch_installed
    if _patch_installed:
        return
    _patch_installed = True
    _configure_download_env()

    import open_clip.factory as factory_mod
    import open_clip.pretrained as pretrained_mod

    if not hasattr(pretrained_mod, "_download_pretrained_orig"):
        pretrained_mod._download_pretrained_orig = pretrained_mod.download_pretrained
    original: Callable[..., Any] = pretrained_mod._download_pretrained_orig

    def download_pretrained_robust(
        cfg: dict[str, Any],
        prefer_hf_hub: bool = True,
        cache_dir: str | None = None,
    ) -> str:
        del prefer_hf_hub
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

    if not hasattr(factory_mod, "_load_state_dict_orig"):
        factory_mod._load_state_dict_orig = factory_mod.load_state_dict

    def load_state_dict_robust(
        checkpoint_path: str,
        device: str = "cpu",
        weights_only: bool = True,
    ) -> Any:
        del weights_only
        return factory_mod._load_state_dict_orig(
            checkpoint_path, device=device, weights_only=False
        )

    factory_mod.load_state_dict = load_state_dict_robust


def _pretrained_cfg(settings: FastClassifySettings) -> dict[str, Any]:
    from open_clip.pretrained import get_pretrained_cfg

    model_key = settings.model_name.replace("/", "-")
    cfg = get_pretrained_cfg(model_key, settings.pretrained)
    if not cfg:
        raise RuntimeError(
            f"Неизвестная пара CLIP: {settings.model_name} / {settings.pretrained}"
        )
    return cfg


def _download_url_quiet(
    url: str,
    dest: Path,
    *,
    expected_sha_prefix: str = "",
    on_log: Callable[[str], None] | None = None,
) -> None:
    """Download without tqdm (works when sys.stdout is None in GUI)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    with urllib.request.urlopen(url, timeout=300) as source:
        total = int(source.headers.get("Content-Length") or 0)
        read = 0
        last_log_pct = -1
        with open(tmp, "wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                read += len(chunk)
                if on_log and total > 0:
                    pct = min(100, (100 * read) // total)
                    if pct >= last_log_pct + 5:
                        last_log_pct = pct
                        on_log(f"CLIP: загрузка весов {pct}%…")

    if expected_sha_prefix:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if not digest.startswith(expected_sha_prefix):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                "SHA256 скачанного файла CLIP не совпадает с ожидаемым"
            )

    tmp.replace(dest)


def ensure_clip_weights_file(
    settings: FastClassifySettings,
    *,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Return local .pt path and open_clip preprocess cfg; download via OpenAI URL when needed."""
    _configure_download_env()
    install_clip_download_patch()

    custom = (getattr(settings, "weights_path", "") or "").strip()
    cfg = _pretrained_cfg(settings)
    if custom:
        path = Path(custom).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Файл весов CLIP не найден: {path}")
        if on_log:
            on_log(f"CLIP: веса из файла {path}")
        return path.resolve(), cfg

    cache = clip_cache_dir()
    url = str(cfg.get("url") or "").strip()
    if not url:
        raise RuntimeError(
            f"Для тега {settings.pretrained!r} нет прямой ссылки; "
            "укажите fast_classify.weights_path в настройках."
        )

    filename = os.path.basename(url)
    dest = cache / filename
    if dest.is_file() and dest.stat().st_size >= _MIN_OPENAI_VITB32_BYTES:
        if on_log:
            on_log(f"CLIP: веса в кэше ({dest.name})")
        return dest, cfg

    expected_sha = ""
    if "openaipublic" in url:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2:
            expected_sha = parts[-2]

    if on_log:
        on_log(
            f"CLIP: скачивание {filename} (~{max(1, _MIN_OPENAI_VITB32_BYTES // 1_000_000)} МБ)…"
        )
    _download_url_quiet(url, dest, expected_sha_prefix=expected_sha, on_log=on_log)
    if on_log:
        on_log(f"CLIP: веса сохранены в {dest}")
    return dest, cfg


def format_clip_load_error(exc: BaseException) -> str:
    cache = clip_cache_dir()
    return (
        f"Ошибка загрузки CLIP: {exc}\n"
        f"Кэш весов: {cache}\n"
        "Скачайте ViT-B-32.pt вручную (тег openai) и укажите путь в "
        "fast_classify.weights_path, либо проверьте интернет и права на запись."
    )

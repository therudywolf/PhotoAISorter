# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Reference images per tag for embedding-based matching."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image

from app.constants import STILL_IMAGE_EXTENSIONS
from app.fast_classify.config import refs_dir

DEFAULT_EXEMPLAR_TAGS: tuple[str, ...] = (
    "iam",
    "iam_face",
    "iam_body",
    "my_dog",
    "my_cat",
    "my_dog_closeup",
    "my_cat_closeup",
)

_README = """\
Reference photos for fast (CLIP) sorting
=====================================

Put a few clear photos in each subfolder (5–20 JPEG/PNG per tag):

  refs/iam/          — photos of you (any angle)
  refs/iam_face/     — face close-ups (optional, improves iam_face)
  refs/my_dog/       — your black labrador
  refs/my_cat/       — your ginger fold cat

Only folders that exist are used. forest (custom tag list) stays in context_tags.json.
"""


def ensure_refs_layout(
    on_log: Callable[[str], None] | None = None,
    *,
    extra_tags: Iterable[str] = (),
) -> Path:
    root = refs_dir()
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.is_file():
        readme.write_text(_README, encoding="utf-8")
    all_tags = list(DEFAULT_EXEMPLAR_TAGS) + [t for t in extra_tags if t]
    seen: set[str] = set()
    for tag in all_tags:
        if tag in seen or not tag:
            continue
        seen.add(tag)
        (root / tag).mkdir(parents=True, exist_ok=True)
    if on_log:
        on_log(f"Эталоны: {root}")
    return root


def list_exemplar_paths(tag: str, *, limit: int = 48) -> list[Path]:
    folder = refs_dir() / tag
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in STILL_IMAGE_EXTENSIONS:
            out.append(p)
    return out[:limit]


def load_exemplar_images(
    tag: str,
    *,
    max_side: int,
    loader: Callable[[Path], Image.Image],
    on_log: Callable[[str], None] | None = None,
) -> list[Image.Image]:
    paths = list_exemplar_paths(tag)
    images: list[Image.Image] = []
    failed: list[str] = []
    for p in paths:
        try:
            images.append(loader(p))
        except OSError as e:
            failed.append(f"{p.name}: {e}")
            continue
    if failed and on_log is not None:
        on_log(f"Эталон '{tag}': {len(failed)} файлов не загрузились: {failed[:3]}")
    return images

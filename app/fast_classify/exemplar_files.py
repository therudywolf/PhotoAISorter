# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Copy/remove exemplar image files on disk."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from app.constants import STILL_IMAGE_EXTENSIONS
from app.fast_classify.exemplars import list_exemplar_paths, refs_dir


def exemplar_dir_for_tag(tag: str) -> Path:
    d = refs_dir() / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unique_dest(folder: Path, src: Path) -> Path:
    base = folder / src.name
    if not base.exists():
        return base
    stem, suf = src.stem, src.suffix
    for i in range(2, 10_000):
        candidate = folder / f"{stem}_{i}{suf}"
        if not candidate.exists():
            return candidate
    raise OSError(f"too many copies for {src.name}")


def add_exemplar_files(
    tag: str,
    paths: Iterable[Path | str],
    *,
    limit: int = 48,
) -> int:
    """Copy image files into refs/<tag>/; return number added."""
    folder = exemplar_dir_for_tag(tag)
    current = len(list_exemplar_paths(tag, limit=limit + 1))
    added = 0
    for raw in paths:
        if current + added >= limit:
            break
        src = Path(raw)
        if not src.is_file():
            continue
        if src.suffix.lower() not in STILL_IMAGE_EXTENSIONS:
            continue
        dest = _unique_dest(folder, src)
        shutil.copy2(src, dest)
        added += 1
    return added


def remove_exemplar_file(tag: str, filename: str) -> bool:
    folder = refs_dir() / tag
    path = folder / filename
    if path.is_file() and path.parent.resolve() == folder.resolve():
        path.unlink()
        return True
    return False

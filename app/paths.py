# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Persistent paths: project-local data/ (refs, CLIP weights) vs roaming AppData (DB, settings)."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.db import default_db_path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def persistent_data_dir() -> Path:
    """Install-local cache (not cleared with sort SQLite cache; not in git)."""
    root = project_root() / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def refs_dir() -> Path:
    return persistent_data_dir() / "refs"


def clip_weights_dir() -> Path:
    return persistent_data_dir() / "clip_weights"


def clip_embedding_cache_path() -> Path:
    return persistent_data_dir() / "clip_cache.sqlite3"


def migrate_roaming_clip_data() -> None:
    """One-time copy refs/weights from %APPDATA%/PhotoAISorter if project data/ is empty."""
    old_base = default_db_path().parent
    new_base = persistent_data_dir()
    pairs = (
        ("refs", True),
        ("clip_weights", True),
        ("clip_cache.sqlite3", False),
    )
    for name, is_dir in pairs:
        src = old_base / name
        dst = new_base / name
        if dst.exists():
            continue
        if is_dir and src.is_dir():
            try:
                shutil.copytree(src, dst)
            except OSError:
                pass
        elif not is_dir and src.is_file():
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass

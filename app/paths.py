# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Paths: project tmp/ (fast cache on SSD) vs data/ (refs, weights)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def project_tmp_dir() -> Path:
    """Volatile caches (SQLite WAL, hash index) — put project on SSD for HDD libraries."""
    override = (os.environ.get("PHOTO_AI_SORTER_TMP") or "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        root = project_root() / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def app_state_dir() -> Path:
    """Sort DB, signatures, gui settings, context_tags."""
    d = project_tmp_dir() / "app_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def persistent_data_dir() -> Path:
    """Large persistent assets (refs, CLIP weights)."""
    root = project_root() / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def refs_dir() -> Path:
    return persistent_data_dir() / "refs"


def clip_weights_dir() -> Path:
    return persistent_data_dir() / "clip_weights"


def clip_embedding_cache_path() -> Path:
    return project_tmp_dir() / "clip_cache.sqlite3"


def file_hash_cache_path() -> Path:
    return project_tmp_dir() / "file_hashes.sqlite3"


def _legacy_roaming_dir() -> Path | None:
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "PhotoAISorter"
    legacy = Path.home() / ".local" / "share" / "PhotoAISorter"
    return legacy if legacy.is_dir() else None


def migrate_roaming_clip_data() -> None:
    """Copy refs/weights from old %APPDATA% into data/ when missing."""
    old_base = _legacy_roaming_dir()
    if old_base is None:
        return
    new_base = persistent_data_dir()
    for name, is_dir in (("refs", True), ("clip_weights", True)):
        src, dst = old_base / name, new_base / name
        if dst.exists() or not src.exists():
            continue
        try:
            if is_dir and src.is_dir():
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)
        except OSError:
            pass


def migrate_app_state_to_project_tmp() -> None:
    """One-time: move SQLite/settings from %APPDATA% to project tmp/app_state."""
    old_base = _legacy_roaming_dir()
    if old_base is None or not old_base.is_dir():
        return
    new_base = app_state_dir()
    names = (
        "state.sqlite3",
        "signatures.sqlite3",
        "gui_settings.json",
        "context_tags.json",
        "secrets.json",
        "duplicate_delete_journal.jsonl",
        "clip_cache.sqlite3",
        "file_hashes.sqlite3",
    )
    for name in names:
        src, dst = old_base / name, new_base / name if name != "clip_cache.sqlite3" else project_tmp_dir() / name
        if name == "clip_cache.sqlite3":
            dst = clip_embedding_cache_path()
        if name == "file_hashes.sqlite3":
            dst = file_hash_cache_path()
        if dst.exists() or not src.is_file():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError:
            pass
    # clip_cache at old roaming root
    old_clip = old_base / "clip_cache.sqlite3"
    if old_clip.is_file() and not clip_embedding_cache_path().exists():
        try:
            shutil.copy2(old_clip, clip_embedding_cache_path())
        except OSError:
            pass


def migrate_legacy_project_root() -> None:
    """Move obsolete repo-root files into tmp/app_state and data/clip_weights."""
    root = project_root()
    legacy_presets = root / "local_presets.json"
    dst_tags = app_state_dir() / "context_tags.json"
    if legacy_presets.is_file():
        try:
            backup = root / "local_presets.json.migrated"
            if not dst_tags.is_file():
                dst_tags.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_presets, dst_tags)
            if not backup.is_file():
                legacy_presets.replace(backup)
        except OSError:
            pass
    cache = root / ".cache"
    if not cache.is_dir():
        return
    weights = clip_weights_dir()
    weights.mkdir(parents=True, exist_ok=True)
    for pt in cache.rglob("*.pt"):
        if not pt.is_file():
            continue
        dest = weights / pt.name
        if dest.is_file():
            continue
        try:
            if pt.stat().st_size >= 1_000_000:
                shutil.copy2(pt, dest)
        except OSError:
            pass

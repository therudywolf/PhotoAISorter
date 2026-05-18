# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Reference photo layout for fast CLIP."""

from __future__ import annotations

from pathlib import Path

from app.fast_classify.exemplars import ensure_refs_layout, list_exemplar_paths


def test_ensure_refs_layout_creates_folders(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setattr("app.fast_classify.exemplars.refs_dir", lambda: tmp_path / "refs")
    root = ensure_refs_layout()
    assert root.is_dir()
    assert (root / "iam").is_dir()
    assert (root / "my_dog").is_dir()
    assert (root / "README.txt").is_file()


def test_list_exemplar_paths_filters_images(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setattr("app.fast_classify.exemplars.refs_dir", lambda: tmp_path / "refs")
    folder = tmp_path / "refs" / "iam"
    folder.mkdir(parents=True)
    (folder / "a.jpg").write_bytes(b"x")
    (folder / "note.txt").write_text("x", encoding="utf-8")
    paths = list_exemplar_paths("iam")
    assert len(paths) == 1
    assert paths[0].suffix.lower() == ".jpg"

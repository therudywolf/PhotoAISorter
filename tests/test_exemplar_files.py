# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Exemplar file copy/remove helpers."""

from __future__ import annotations

from pathlib import Path

from app.fast_classify import exemplar_files as ef
from app.fast_classify.exemplars import list_exemplar_paths


def test_add_exemplar_files(monkeypatch: object, tmp_path: Path) -> None:
    from app.fast_classify import exemplars

    monkeypatch.setattr(ef, "refs_dir", lambda: tmp_path)
    monkeypatch.setattr(exemplars, "refs_dir", lambda: tmp_path)
    src = tmp_path / "in.jpg"
    src.write_bytes(b"jpeg")
    n = ef.add_exemplar_files("my_dog", [src])
    assert n == 1
    listed = list_exemplar_paths("my_dog")
    assert len(listed) == 1
    assert listed[0].name.endswith(".jpg")

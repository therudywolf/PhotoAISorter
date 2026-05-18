# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Persistent data directory layout."""

from __future__ import annotations

from app import paths


def test_refs_under_project_data() -> None:
    refs = paths.refs_dir()
    assert refs.name == "refs"
    assert refs.parent.name == "data"
    assert refs.parent.parent == paths.project_root()


def test_tmp_cache_under_project() -> None:
    tmp = paths.project_tmp_dir()
    assert tmp.name == "tmp"
    assert tmp.parent == paths.project_root()
    assert paths.clip_embedding_cache_path().parent == tmp

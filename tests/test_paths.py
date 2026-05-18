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

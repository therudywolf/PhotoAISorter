# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Tag mode UI helpers and hybrid readiness."""

from __future__ import annotations

from app.tag_mode_ui import (
    FLEXIBLE_TAG_MODES,
    PRESET_TAG_MODES,
    build_tag_mode_hint,
    hybrid_start_blockers,
    label_for_mode,
    mode_from_label,
    refs_button_enabled,
)


def test_mode_label_roundtrip() -> None:
    for mode in PRESET_TAG_MODES + FLEXIBLE_TAG_MODES:
        assert mode_from_label(label_for_mode(mode)) == mode


def test_refs_button_only_for_custom_lists() -> None:
    assert refs_button_enabled("hybrid")
    assert refs_button_enabled("custom")
    assert not refs_button_enabled("preset_sfw")


def test_hybrid_blockers_no_tags() -> None:
    blockers = hybrid_start_blockers(categories_count=0, clip_ready=True)
    assert any("набор" in b.lower() or "тег" in b.lower() for b in blockers)


def test_hybrid_blockers_clip_missing() -> None:
    blockers = hybrid_start_blockers(categories_count=10, clip_ready=False)
    assert len(blockers) == 1
    assert "pip install" in blockers[0]


def test_build_tag_mode_hint_hybrid_includes_clip_status() -> None:
    hint_ok = build_tag_mode_hint("hybrid", clip_ready=True)
    hint_bad = build_tag_mode_hint("hybrid", clip_ready=False)
    assert "CLIP" in hint_ok
    assert "CLIP" in hint_bad
    assert hint_ok != hint_bad

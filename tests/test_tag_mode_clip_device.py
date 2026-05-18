# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP device UI labels."""

from __future__ import annotations

from app.tag_mode_ui import CLIP_DEVICE_LABELS, CLIP_DEVICE_VALUES, build_tag_mode_hint


def test_clip_device_label_roundtrip() -> None:
    assert CLIP_DEVICE_VALUES[CLIP_DEVICE_LABELS["cuda"]] == "cuda"


def test_hybrid_hint_includes_device() -> None:
    hint = build_tag_mode_hint("hybrid", clip_ready=True, clip_device="cuda")
    assert "GPU" in hint or "cuda" in hint.lower() or "Устройство" in hint

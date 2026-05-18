# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Custom tag lists use raw CLIP scores without preset reweighting."""

from __future__ import annotations

from app.fast_classify.priority import pick_tag


def test_pick_tag_without_preset_rules() -> None:
    scores = {"cat": 0.4, "dog": 0.35, "uncategorized": 0.1}
    wl = frozenset(scores)
    tag, conf, _ = pick_tag(scores, whitelist=wl, apply_preset_rules=False)
    assert tag == "cat"
    assert conf == 0.4

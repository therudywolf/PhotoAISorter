# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Confidence margin for single-tag CLIP setups."""

from __future__ import annotations

import numpy as np

from app.fast_classify.scoring import confidence_from_probs, needs_review


def test_single_tag_margin_uses_top_prob() -> None:
    probs = np.array([0.91], dtype=np.float64)
    top, margin = confidence_from_probs(probs)
    assert top == 0.91
    assert margin == 0.91
    assert not needs_review(top, margin, min_prob=0.55, min_margin=0.06)

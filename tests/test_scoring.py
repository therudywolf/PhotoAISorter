# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

import numpy as np
from app.fast_classify.scoring import confidence_from_probs, needs_review, softmax_probs


def test_softmax_probs_sum_to_one() -> None:
    sims = np.array([0.2, 0.25, 0.1], dtype=np.float32)
    probs = softmax_probs(sims, temperature=0.05)
    assert abs(sum(probs) - 1.0) < 1e-5


def test_needs_review_low_margin() -> None:
    assert needs_review(0.4, 0.03, min_prob=0.28, min_margin=0.06) is True
    assert needs_review(0.4, 0.15, min_prob=0.28, min_margin=0.06) is False


def test_confidence_from_probs_top1() -> None:
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    top, margin = confidence_from_probs(probs)
    assert top == 0.5
    assert abs(margin - 0.2) < 1e-6

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np
import pytest
from app.fast_classify.scoring import raw_similarity_margin, topk_softmax_probs


def test_topk_softmax_concentrates_mass() -> None:
    sims = np.array([0.1, 0.35, 0.12, 0.34, 0.05], dtype=np.float32)
    full = topk_softmax_probs(sims, temperature=0.07, top_k=2)
    assert abs(float(full.sum()) - 1.0) < 1e-5
    assert full[1] + full[3] > 0.95


def test_raw_margin() -> None:
    sims = np.array([0.3, 0.28, 0.1], dtype=np.float32)
    top, margin = raw_similarity_margin(sims)
    assert float(top) == pytest.approx(0.3)
    assert float(margin) == pytest.approx(0.02)


def test_build_custom_prompts_all_tags() -> None:
    from app.context_tags import Tag, TagSet, build_custom_prompts

    ts = TagSet(
        name="t",
        tags=[Tag(key="dog", description=""), Tag(key="cat", description="ginger cat")],
    )
    p = build_custom_prompts(ts)
    assert p["dog"] == "dog"
    assert "ginger" in p["cat"]

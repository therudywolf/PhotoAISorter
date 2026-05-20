# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np
import pytest
from app.fast_classify.config import FastClassifySettings
from app.fast_classify.pipeline import FastClassifier


def test_exemplar_additive_not_multiplicative() -> None:
    """Exemplar match should add a small delta, not replace text score with sim*boost."""
    settings = FastClassifySettings.from_dict(
        {
            "exemplar_boost": 1.38,
            "min_exemplar_similarity": 0.32,
            "exemplar_max_delta": 0.12,
        }
    )
    clf = FastClassifier.__new__(FastClassifier)
    clf.settings = settings
    clf._tags = ["a", "b"]
    clf._text_matrix = np.eye(2, dtype=np.float32)
    clf._prompt_matrix = np.zeros((0, 2), dtype=np.float32)
    clf._prompt_owner = np.zeros((0,), dtype=np.int32)
    clf._exemplar_matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    clf._exemplar_owner = np.array([0], dtype=np.int32)

    feat = np.array([[0.85, 0.52]], dtype=np.float32)
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    combined, text = clf._raw_sims_batch(feat)

    assert float(text[0, 0]) == pytest.approx(0.85, abs=0.02)
    assert float(combined[0, 0]) == pytest.approx(0.97, abs=0.02)
    assert float(combined[0, 0]) < 0.85 * 1.22

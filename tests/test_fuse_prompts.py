# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np
from app.fast_classify.pipeline import _fuse_prompt_vectors


def test_fuse_prompt_vectors_unit_norm() -> None:
    chunk = np.random.randn(4, 8).astype(np.float32)
    v = _fuse_prompt_vectors(chunk, fusion_weight=0.5)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

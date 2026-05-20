# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Minimal FastClassifier stub for unit tests (no torch / open_clip)."""

from __future__ import annotations

import numpy as np
from app.fast_classify.config import FastClassifySettings
from app.fast_classify.pipeline import FastClassifier


def stub_classifier(
    tags: list[str],
    *,
    settings: FastClassifySettings | None = None,
    text_matrix: np.ndarray | None = None,
    exemplar_matrix: np.ndarray | None = None,
    exemplar_owner: np.ndarray | None = None,
    apply_preset_rules: bool = False,
) -> FastClassifier:
    settings = settings or FastClassifySettings.from_dict({})
    dim = int(text_matrix.shape[1]) if text_matrix is not None else 4
    if text_matrix is None:
        n = len(tags)
        text_matrix = np.eye(n, dim, dtype=np.float32) if n <= dim else np.eye(dim, dtype=np.float32)
    if exemplar_matrix is None:
        exemplar_matrix = np.zeros((0, dim), dtype=np.float32)
    if exemplar_owner is None:
        exemplar_owner = np.zeros((0,), dtype=np.int32)

    clf = FastClassifier.__new__(FastClassifier)
    clf.settings = settings
    clf._tags = list(tags)
    clf._whitelist = frozenset(tags)
    clf._apply_preset_rules = apply_preset_rules
    clf._text_matrix = text_matrix.astype(np.float32)
    clf._prompt_matrix = np.zeros((0, dim), dtype=np.float32)
    clf._prompt_owner = np.zeros((0,), dtype=np.int32)
    clf._exemplar_matrix = exemplar_matrix.astype(np.float32)
    clf._exemplar_owner = exemplar_owner.astype(np.int32)
    return clf

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Fast classify settings persistence."""

from __future__ import annotations

from app.fast_classify.config import FastClassifySettings, load_fast_classify_settings


def test_fast_classify_settings_roundtrip() -> None:
    raw = {
        "batch_size": 64,
        "confidence_threshold": 0.31,
        "vlm_fallback": False,
        "device": "cpu",
    }
    s = FastClassifySettings.from_dict(raw)
    assert s.batch_size == 64
    assert s.confidence_threshold == 0.31
    assert s.min_margin == 0.06
    assert s.softmax_temperature == 0.05
    assert s.vlm_fallback is False
    assert s.to_dict()["device"] == "cpu"


def test_load_fast_classify_from_gui_blob() -> None:
    s = load_fast_classify_settings({"fast_classify": {"vlm_fallback": False, "batch_size": 16}})
    assert s.vlm_fallback is False
    assert s.batch_size == 16

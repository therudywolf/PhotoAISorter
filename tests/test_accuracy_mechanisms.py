# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""
Regression tests for CLIP exemplar scoring, strict thresholds, and VLM merge.

These guard the accuracy fixes in pipeline v7 (additive exemplars, conservative VLM).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from app.category_aliases import resolve_storage_category
from app.classification_result import parse_classification_result
from app.constants import UNCATEGORIZED
from app.fast_classify.config import FastClassifySettings
from app.fast_classify.pipeline import _reject_outlier_vectors
from app.fast_classify.priority import pick_tag
from app.fast_classify.quality import finalize_fast_classify_settings
from app.fast_classify.scoring import topk_softmax_probs
from app.sort_hybrid import _clip_needs_vlm_fallback, _merge_vlm_with_clip
from tests.helpers_clip_stub import stub_classifier

# --- Exemplar delta mechanics ---


def test_exemplar_no_delta_below_min_similarity() -> None:
    settings = FastClassifySettings.from_dict(
        {"min_exemplar_similarity": 0.32, "exemplar_boost": 1.22, "exemplar_max_delta": 0.12}
    )
    clf = stub_classifier(
        ["with_ref", "plain"],
        settings=settings,
        text_matrix=np.eye(2, dtype=np.float32),
        exemplar_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        exemplar_owner=np.array([0], dtype=np.int32),
    )
    # Weak match to exemplar (cosine ~0.25) but stronger text on plain.
    feat = np.array([[0.25, 0.97]], dtype=np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    combined, text = clf._raw_sims_batch(feat)
    assert float(combined[0, 0]) == pytest.approx(float(text[0, 0]), abs=1e-5)
    assert float(combined[0, 1]) > float(combined[0, 0])


def test_exemplar_combined_similarity_never_above_one() -> None:
    settings = FastClassifySettings.from_dict(
        {"min_exemplar_similarity": 0.20, "exemplar_boost": 1.5, "exemplar_max_delta": 0.12}
    )
    clf = stub_classifier(
        ["a", "b"],
        settings=settings,
        text_matrix=np.eye(2, dtype=np.float32),
        exemplar_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        exemplar_owner=np.array([0], dtype=np.int32),
    )
    feat = np.array([[1.0, 0.0]], dtype=np.float32)
    combined, _text = clf._raw_sims_batch(feat)
    assert float(combined.max()) <= 1.0 + 1e-6


def test_exemplar_delta_capped_at_max() -> None:
    settings = FastClassifySettings.from_dict(
        {"min_exemplar_similarity": 0.20, "exemplar_boost": 2.0, "exemplar_max_delta": 0.08}
    )
    clf = stub_classifier(
        ["a", "b"],
        settings=settings,
        text_matrix=np.eye(2, dtype=np.float32),
        exemplar_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        exemplar_owner=np.array([0], dtype=np.int32),
    )
    feat = np.array([[0.82, 0.45]], dtype=np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    combined, text = clf._raw_sims_batch(feat)
    delta = float(combined[0, 0] - text[0, 0])
    assert delta == pytest.approx(0.08, abs=1e-4)


def test_text_wins_over_weak_exemplar_on_other_tag() -> None:
    """Tag with refs must not beat a tag with clearly higher text cosine."""
    settings = FastClassifySettings.from_dict(
        {"min_exemplar_similarity": 0.32, "exemplar_boost": 1.38, "exemplar_max_delta": 0.12}
    )
    clf = stub_classifier(
        ["iam", "landscape"],
        settings=settings,
        text_matrix=np.eye(2, dtype=np.float32),
        exemplar_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        exemplar_owner=np.array([0], dtype=np.int32),
    )
    feat = np.array([[0.15, 0.92]], dtype=np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    combined, _text = clf._raw_sims_batch(feat)
    assert float(combined[0, 1]) > float(combined[0, 0])


def test_exemplar_only_top_tag_goes_to_review() -> None:
    settings = FastClassifySettings.from_dict(
        {
            "confidence_threshold": 0.20,
            "min_margin": 0.05,
            "min_raw_similarity": 0.15,
            "min_raw_margin": 0.01,
            "softmax_temperature": 0.07,
            "top_k_softmax": 5,
        }
    )
    clf = stub_classifier(["dog", "cat"], settings=settings, apply_preset_rules=False)
    # dog wins combined mainly via exemplar bump; text still weak for dog.
    sims = np.array([0.32, 0.28], dtype=np.float32)
    text = np.array([0.17, 0.26], dtype=np.float32)
    result = clf._result_from_sims_row(sims, text_sims_row=text)
    assert result.category == "dog"
    assert result.needs_review is True


# --- Strict CLIP thresholds ---


def test_clip_low_similarity_returns_uncategorized() -> None:
    settings = FastClassifySettings.from_dict({"min_raw_similarity": 0.25})
    clf = stub_classifier(["a", "b"], settings=settings, apply_preset_rules=False)
    sims = np.array([0.12, 0.10], dtype=np.float32)
    result = clf._result_from_sims_row(sims, text_sims_row=sims)
    assert result.category == UNCATEGORIZED
    assert "clip_low_sim" in result.reason_short
    assert result.needs_review is True


def test_topk_softmax_avoids_uniform_guess_among_many_classes() -> None:
    sims = np.zeros(40, dtype=np.float32)
    sims[3] = 0.36
    sims[17] = 0.34
    for i in range(40):
        if i not in (3, 17):
            sims[i] = 0.08 + 0.001 * i
    probs = topk_softmax_probs(sims, temperature=0.06, top_k=8)
    assert float(probs[3]) > 0.45
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_high_margin_text_match_can_auto_sort() -> None:
    settings = FastClassifySettings.from_dict(
        {
            "confidence_threshold": 0.20,
            "min_margin": 0.08,
            "min_raw_similarity": 0.18,
            "min_raw_margin": 0.04,
            "softmax_temperature": 0.06,
            "top_k_softmax": 5,
        }
    )
    clf = stub_classifier(["dog", "cat", "car"], settings=settings, apply_preset_rules=False)
    sims = np.array([0.62, 0.18, 0.15], dtype=np.float32)
    result = clf._result_from_sims_row(sims, text_sims_row=sims)
    assert result.category == "dog"
    assert result.needs_review is False


# --- Priority rules (personal tags) ---


def test_personal_priority_requires_strong_score() -> None:
    wl = frozenset({"human_real_sfw", "iam", "landscape"})
    tag_plain, _, _ = pick_tag(
        {"human_real_sfw": 0.25, "iam": 0.14, "landscape": 0.13},
        whitelist=wl,
        apply_preset_rules=False,
    )
    assert tag_plain == "human_real_sfw"

    tag_boost, _, _ = pick_tag(
        {"human_real_sfw": 0.25, "iam": 0.24, "landscape": 0.13},
        whitelist=wl,
        apply_preset_rules=True,
    )
    assert tag_boost == "iam"


# --- VLM merge & fallback gating ---


def _res(cat: str, conf: float, *, review: bool = False, reason: str = "test") -> object:
    from app.classification_result import ClassificationResult

    return ClassificationResult(cat, [cat], conf, reason, review, "")


def test_vlm_not_used_for_borderline_clip_review() -> None:
    r = _res("my_dog", 0.58, review=True)
    assert _clip_needs_vlm_fallback(r, confidence_threshold=0.55) is False


def test_vlm_used_for_low_clip_confidence() -> None:
    r = _res("my_dog", 0.30, review=True)
    assert _clip_needs_vlm_fallback(r, confidence_threshold=0.55) is True


def test_merge_rejects_vlm_with_defaultish_confidence() -> None:
    clip = _res("iam", 0.48, review=True, reason="clip raw=0.40 margin=0.05")
    vlm = _res("human_real_sfw", 0.55, reason="legacy")
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "iam"


def test_merge_accepts_vlm_on_clear_low_sim_clip() -> None:
    clip = _res(UNCATEGORIZED, 0.08, review=True, reason="clip_low_sim raw=0.10")
    vlm = _res("screenshot", 0.72)
    out = _merge_vlm_with_clip(vlm, clip, confidence_threshold=0.55)
    assert out.category == "screenshot"


def test_parse_structured_missing_confidence_not_inflated() -> None:
    raw = json.dumps({"primary_category": "my_dog", "reasoning": "black lab"})
    result = parse_classification_result(
        raw, mode="hybrid", whitelist=frozenset({"my_dog", "my_cat"})
    )
    assert result.confidence == pytest.approx(0.55, abs=1e-6)
    assert result.confidence < 0.70


# --- Quality profile sanity ---


def test_ultra_profile_exemplar_params_in_safe_range() -> None:
    s = finalize_fast_classify_settings(
        FastClassifySettings.from_dict({"quality": "ultra", "device": "cpu"})
    )
    assert 1.0 <= s.exemplar_boost <= 1.35
    assert 0.28 <= s.min_exemplar_similarity <= 0.40
    assert 0.08 <= s.exemplar_max_delta <= 0.15


# --- Storage aliases for virtual exemplar tags ---


def test_builtin_virtual_tag_aliases() -> None:
    assert resolve_storage_category("iam_face", {}) == "iam"
    assert resolve_storage_category("my_dog_closeup", {}) == "my_dog"


# --- Exemplar outlier rejection ---


def test_reject_outlier_exemplar_vectors() -> None:
    good = np.array([[1.0, 0.0], [0.99, 0.05], [0.98, 0.08]], dtype=np.float32)
    bad = np.array([[0.1, 0.99]], dtype=np.float32)
    feats = np.vstack([good, bad])
    kept = _reject_outlier_vectors(feats)
    assert kept.shape[0] >= 3
    assert kept.shape[0] <= 4

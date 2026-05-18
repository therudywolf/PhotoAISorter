# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Structured classification parsing."""

from app.classification_result import parse_classification_result
from app.constants import UNCATEGORIZED


def test_parse_json_classification_auto_alias() -> None:
    raw = '{"primary_category": "auto/bmw", "candidates": ["car"], "confidence": 0.91, "reason_short": "vehicle"}'
    result = parse_classification_result(raw, mode="auto", aliases={"auto/bmw": "vehicles/bmw"})
    assert result.category == "vehicles/bmw"
    assert result.confidence == 0.91
    assert result.needs_review is False


def test_parse_json_low_confidence_needs_review() -> None:
    raw = '{"primary_category": "nature/forest", "confidence": 0.4}'
    result = parse_classification_result(raw, mode="auto")
    assert result.category == "nature/forest"
    assert result.needs_review is True


def test_parse_legacy_tag_fallback() -> None:
    result = parse_classification_result("vehicles_and_racing", mode="strict")
    assert result.category == "vehicles_and_racing"
    assert result.needs_review is False


def test_parse_general_preset_tag() -> None:
    result = parse_classification_result('{"primary_category": "coding_ide_and_terminal", "confidence": 0.8}', mode="general")
    assert result.category == "coding_ide_and_terminal"
    assert result.needs_review is False


def test_parse_strict_rejects_nsfw_tag() -> None:
    result = parse_classification_result("furry_nsfw_canidae", mode="strict")
    assert result.category == UNCATEGORIZED
    assert result.needs_review is True


def test_parse_empty_falls_back_uncategorized() -> None:
    result = parse_classification_result("", mode="strict")
    assert result.category == UNCATEGORIZED
    assert result.needs_review is True


def test_parse_custom_mode_with_whitelist() -> None:
    custom_wl = frozenset({"portraits", "events", "uncategorized"})
    raw = '{"primary_category": "portraits", "confidence": 0.85}'
    result = parse_classification_result(raw, mode="custom", whitelist=custom_wl)
    assert result.category == "portraits"
    assert result.needs_review is False


def test_parse_custom_mode_rejects_unknown_tag() -> None:
    custom_wl = frozenset({"portraits", "events", "uncategorized"})
    raw = "vehicles_and_racing"
    result = parse_classification_result(raw, mode="custom", whitelist=custom_wl)
    assert result.category == UNCATEGORIZED


def test_parse_hybrid_mode_with_whitelist() -> None:
    wl = frozenset({"iam", "dog", "uncategorized"})
    raw = '{"best_folder_name": "iam", "confidence": 0.77, "top_candidates": [], "reasoning": "selfie"}'
    result = parse_classification_result(raw, mode="hybrid", whitelist=wl)
    assert result.category == "iam"


def test_parse_custom_primary_category_alias_for_best_folder_name() -> None:
    custom_wl = frozenset({"my_dog", "dog", "uncategorized"})
    raw = (
        '{"primary_category": "my_dog", "confidence": 0.8, '
        '"candidates": ["dog"], "reason_short": "black lab"}'
    )
    result = parse_classification_result(raw, mode="custom", whitelist=custom_wl)
    assert result.category == "my_dog"
    assert "dog" in result.candidates


def test_parse_preset_mode_accepts_profile_tag() -> None:
    from app.constants import SearchProfile, categories_for_profile
    nsfw_wl = frozenset(categories_for_profile(SearchProfile.NSFW))
    raw = '{"primary_category": "humans_nsfw_female", "confidence": 0.9}'
    result = parse_classification_result(raw, mode="preset", whitelist=nsfw_wl)
    assert result.category == "humans_nsfw_female"
    assert result.needs_review is False

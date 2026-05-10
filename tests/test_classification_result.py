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


def test_parse_empty_falls_back_uncategorized() -> None:
    result = parse_classification_result("", mode="strict")
    assert result.category == UNCATEGORIZED
    assert result.needs_review is True

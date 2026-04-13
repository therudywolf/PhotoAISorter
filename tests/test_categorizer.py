"""Unit tests for tag normalization and merge."""

from app.categorizer import merge_tags_by_priority, normalize_tag, normalize_tag_free
from app.constants import UNCATEGORIZED


def test_normalize_exact_tag() -> None:
    assert normalize_tag("human_real_sfw") == "human_real_sfw"


def test_normalize_last_line() -> None:
    assert normalize_tag("thinking…\nhuman_sfw") == "human_real_sfw"


def test_merge_priority_cars_over_human_sfw() -> None:
    assert merge_tags_by_priority(["human_sfw", "cars_and_bmw"]) == "vehicles_and_racing"


def test_merge_empty() -> None:
    assert merge_tags_by_priority([]) == UNCATEGORIZED


def test_normalize_tag_with_think_block() -> None:
    raw = "<think>\nreasoning\n</think>\nhuman_real_sfw"
    assert normalize_tag(raw) == "human_real_sfw"


def test_normalize_tag_with_channel_thought_block() -> None:
    raw = "<|channel>thought\ninternal reasoning\n<channel|>\nhuman_real_sfw"
    assert normalize_tag(raw) == "human_real_sfw"


def test_normalize_tag_free_hierarchical() -> None:
    assert normalize_tag_free("Nature / Forest / Sunset") == "nature/forest/sunset"


def test_normalize_tag_free_invalid_fallback() -> None:
    assert normalize_tag_free("###") == UNCATEGORIZED


def test_legacy_alias_human_group_is_canonicalized() -> None:
    assert normalize_tag("human_nsfw_group") == "human_real_nsfw_female"


def test_legacy_alias_cars_is_canonicalized() -> None:
    assert normalize_tag("cars_and_bmw") == "vehicles_and_racing"

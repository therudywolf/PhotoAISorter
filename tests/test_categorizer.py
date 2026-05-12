# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Unit tests for tag normalization and merge."""

from app.categorizer import merge_tags_by_priority, normalize_tag, normalize_tag_auto, normalize_tag_free
from app.constants import GENERAL_CATEGORY_WHITELIST, UNCATEGORIZED


def test_normalize_exact_tag() -> None:
    assert normalize_tag("humans_sfw") == "humans_sfw"


def test_normalize_legacy_alias() -> None:
    assert normalize_tag("human_real_sfw") == "humans_sfw"


def test_normalize_last_line() -> None:
    assert normalize_tag("thinking…\nhuman_sfw") == "humans_sfw"


def test_merge_priority_cars_over_human_sfw() -> None:
    assert merge_tags_by_priority(["humans_sfw", "cars_and_bmw"]) == "vehicles_and_racing"


def test_merge_empty() -> None:
    assert merge_tags_by_priority([]) == UNCATEGORIZED


def test_normalize_tag_with_think_block() -> None:
    raw = "<think>\nreasoning\n</think>\nhumans_sfw"
    assert normalize_tag(raw) == "humans_sfw"


def test_normalize_tag_with_channel_thought_block() -> None:
    raw = "<|channel>thought\ninternal reasoning\n<channel|>\nhumans_sfw"
    assert normalize_tag(raw) == "humans_sfw"


def test_normalize_tag_free_hierarchical() -> None:
    assert normalize_tag_free("Nature / Forest / Sunset") == "nature/forest/sunset"


def test_normalize_tag_free_invalid_fallback() -> None:
    assert normalize_tag_free("###") == UNCATEGORIZED


def test_normalize_tag_free_does_not_substring_match_preset() -> None:
    """Free mode must not map a hierarchical tag to a preset just because it appears as a segment."""
    assert normalize_tag_free("art/humans_sfw/parody") == "art/humans_sfw/parody"


def test_normalize_tag_free_exact_preset_still_works() -> None:
    assert normalize_tag_free("vehicles_and_racing") == "vehicles_and_racing"


def test_normalize_tag_free_prefers_final_tag_line_after_plain_reasoning() -> None:
    raw = "The image appears to be a workstation, so I will use a desk tag.\nfinal: tech/desk/monitors"
    assert normalize_tag_free(raw) == "tech/desk/monitors"


def test_normalize_tag_free_rejects_plain_prose_without_tag() -> None:
    assert normalize_tag_free("The image appears to be a workstation with multiple monitors.") == UNCATEGORIZED


def test_legacy_alias_human_group_is_canonicalized() -> None:
    assert normalize_tag("human_nsfw_group") == "humans_nsfw_female"


def test_legacy_alias_cars_is_canonicalized() -> None:
    assert normalize_tag("cars_and_bmw") == "vehicles_and_racing"


def test_normalize_tag_auto_keeps_known_whitelist_tag() -> None:
    raw = "vehicles_and_racing, road_trip/night"
    assert normalize_tag_auto(raw) == "vehicles_and_racing"


def test_normalize_general_preset_accepts_extended_tag() -> None:
    assert normalize_tag("pc_build_and_hardware", whitelist=GENERAL_CATEGORY_WHITELIST) == "pc_build_and_hardware"


def test_sfw_whitelist_rejects_nsfw_tag() -> None:
    from app.constants import CANONICAL_CATEGORY_WHITELIST
    assert normalize_tag("humans_nsfw_female", whitelist=CANONICAL_CATEGORY_WHITELIST) == UNCATEGORIZED


def test_normalize_tag_auto_does_not_substring_force_preset() -> None:
    raw = "tech/desk/monitors\nexplanation mentions humans_sfw"
    assert normalize_tag_auto(raw) == "tech/desk"


def test_normalize_tag_auto_skips_plain_reasoning_before_final_candidate() -> None:
    raw = "The image shows a modified car at night.\nvehicles/bmw"
    assert normalize_tag_auto(raw) == "vehicles/bmw"


def test_normalize_tag_auto_picks_most_frequent_candidate() -> None:
    raw = "nature/forest/sunset, city/night, nature/forest/sunset"
    assert normalize_tag_auto(raw) == "nature/forest"


def test_normalize_tag_auto_collapses_vehicle_synonyms() -> None:
    raw = "car, cars, vehicle, auto"
    assert normalize_tag_auto(raw) == "vehicles_and_racing"


def test_normalize_tag_auto_keeps_stable_vehicle_detail() -> None:
    raw = "bmw/car, auto/bmw, vehicles/bmw"
    assert normalize_tag_auto(raw) == "vehicles/bmw"


def test_normalize_tag_auto_drops_weak_segments_and_caps_depth() -> None:
    raw = "photo/nature/forest/sunset/evening"
    assert normalize_tag_auto(raw) == "nature/forest"


def test_normalize_tag_auto_tie_keeps_model_order() -> None:
    raw = "city/night, nature/forest"
    assert normalize_tag_auto(raw) == "city/night"

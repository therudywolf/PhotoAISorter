# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Category alias storage and normalization."""

from pathlib import Path

from app.category_aliases import load_category_aliases, resolve_storage_category, save_category_aliases


def test_category_aliases_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "aliases.json"
    save_category_aliases({"Auto / BMW": "Vehicles / BMW", "###": "bad"}, p)
    assert load_category_aliases(p) == {"auto/bmw": "vehicles/bmw"}


def test_resolve_storage_category_maps_virtual_tags() -> None:
    aliases = {"iam_face": "iam", "my_dog_closeup": "my_dog"}
    assert resolve_storage_category("iam_face", aliases) == "iam"
    assert resolve_storage_category("my_dog", aliases) == "my_dog"
    assert resolve_storage_category("car", aliases) == "car"


def test_builtin_alt_aliases_merge_into_canonical_folder() -> None:
    """Legacy *_alt tags fold back into their canonical folder with no user aliases."""
    assert resolve_storage_category("my_dog_alt", {}) == "my_dog"
    assert resolve_storage_category("personal_user_sfw_alt", {}) == "personal_user_sfw"
    assert resolve_storage_category("personal_user_nsfw_alt", {}) == "personal_user_nsfw"
    assert resolve_storage_category("puppy_play_alt", {}) == "puppy_play"
    assert resolve_storage_category("uncategorized_alt", {}) == "uncategorized"


def test_user_alias_overrides_builtin() -> None:
    assert resolve_storage_category("iam_face", {"iam_face": "portraits"}) == "portraits"


def test_clean_alias_tag_normalizes_separators_and_case() -> None:
    from app.category_aliases import clean_alias_tag

    assert clean_alias_tag("  BMW / Cars  ") == "bmw/cars"
    assert clean_alias_tag("a — b") == "a_b"
    assert clean_alias_tag("///x///") == "x"
    assert clean_alias_tag("") == ""


def test_unknown_tag_passes_through_unchanged() -> None:
    assert resolve_storage_category("forest", {}) == "forest"

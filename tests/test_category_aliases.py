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

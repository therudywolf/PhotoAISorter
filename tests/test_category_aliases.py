"""Category alias storage and normalization."""

from pathlib import Path

from app.category_aliases import load_category_aliases, save_category_aliases


def test_category_aliases_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "aliases.json"
    save_category_aliases({"Auto / BMW": "Vehicles / BMW", "###": "bad"}, p)
    assert load_category_aliases(p) == {"auto/bmw": "vehicles/bmw"}

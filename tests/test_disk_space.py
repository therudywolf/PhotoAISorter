# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Disk space checks before copy."""

from pathlib import Path
from unittest.mock import patch

from app.worker import has_disk_space_for_copy


def test_has_disk_space_when_enough_free(tmp_path: Path) -> None:
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x" * 100)
    dest = tmp_path / "out"
    dest.mkdir()
    with patch("app.worker.shutil.disk_usage", return_value=type("U", (), {"free": 200_000_000})()):
        assert has_disk_space_for_copy(dest, src) is True


def test_has_disk_space_false_when_full(tmp_path: Path) -> None:
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x" * 1000)
    dest = tmp_path / "out"
    dest.mkdir()
    with patch("app.worker.shutil.disk_usage", return_value=type("U", (), {"free": 10})()):
        assert has_disk_space_for_copy(dest, src) is False


def test_has_disk_space_fail_closed_on_stat_error(tmp_path: Path) -> None:
    src = tmp_path / "missing.jpg"
    dest = tmp_path / "out"
    dest.mkdir()
    assert has_disk_space_for_copy(dest, src) is False


def test_has_disk_space_fail_closed_on_disk_usage_error(tmp_path: Path) -> None:
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x")
    dest = tmp_path / "out"
    dest.mkdir()
    with patch("app.worker.shutil.disk_usage", side_effect=OSError("no disk info")):
        assert has_disk_space_for_copy(dest, src) is False

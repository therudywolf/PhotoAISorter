# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""unique_dest_path naming and crash-recovery idempotency."""

from __future__ import annotations

from pathlib import Path

from app.images import file_sha256
from app.worker import unique_dest_path


def test_free_name_returned_as_is(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    assert unique_dest_path(dest, "photo.jpg") == dest / "photo.jpg"


def test_name_collision_with_different_content_gets_suffix(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "photo.jpg").write_bytes(b"already here")
    src_digest = "deadbeef"  # not equal to the existing file's hash
    assert unique_dest_path(dest, "photo.jpg", source_digest=src_digest) == dest / "photo_1.jpg"


def test_identical_existing_file_is_reused_not_duplicated(tmp_path: Path) -> None:
    """Crash recovery: a byte-identical copy already on disk must be reused.

    Simulates a hard crash that copied the file but lost the DB record, so the
    file is reprocessed on resume. It must NOT become photo_1.jpg.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    content = b"the real photo bytes"
    existing = dest / "photo.jpg"
    existing.write_bytes(content)
    digest = file_sha256(existing)
    assert unique_dest_path(dest, "photo.jpg", source_digest=digest) == existing


def test_identical_match_found_among_suffixed_variants(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "photo.jpg").write_bytes(b"a different earlier file")
    content = b"our file from before the crash"
    variant = dest / "photo_1.jpg"
    variant.write_bytes(content)
    digest = file_sha256(variant)
    assert unique_dest_path(dest, "photo.jpg", source_digest=digest) == variant


def test_without_digest_collision_still_suffixes(tmp_path: Path) -> None:
    """Back-compat: callers that pass no digest keep the old suffixing behaviour."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "photo.jpg").write_bytes(b"x")
    assert unique_dest_path(dest, "photo.jpg") == dest / "photo_1.jpg"

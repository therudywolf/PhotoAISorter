"""Duplicate finder: grouping, signature cache, regroup."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.duplicate_finder import (
    DuplicateFinderOptions,
    FileDupInfo,
    build_groups_from_records,
    compute_one_signature,
    load_records_from_db,
    regroup_from_cached_records,
)
from app.signature_db import SIG_CACHE_VERSION, SignatureDatabase


def _solid_png(path: Path, rgb: tuple[int, int, int] = (200, 100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), rgb).save(path, format="PNG")


def test_exact_duplicate_group(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _solid_png(a)
    b.write_bytes(a.read_bytes())

    db = SignatureDatabase(tmp_path / "s.sqlite3")
    opts = DuplicateFinderOptions(
        include_exact=True,
        include_perceptual=False,
        parallel_workers=2,
    )

    def log(_: str) -> None:
        pass

    ra = compute_one_signature(a, db, opts, force_recompute=True, on_log=log)
    rb = compute_one_signature(b, db, opts, force_recompute=True, on_log=log)
    assert ra is not None and rb is not None
    assert ra.sha256 == rb.sha256

    groups = build_groups_from_records([ra, rb], opts)
    assert len(groups) == 1
    assert len(groups[0].paths) == 2
    assert groups[0].is_exact is True


def test_perceptual_similar_group(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    im = Image.new("RGB", (120, 90), (10, 140, 200))
    im.save(a, format="JPEG", quality=95)
    im.save(b, format="JPEG", quality=40)

    db = SignatureDatabase(tmp_path / "s2.sqlite3")
    opts = DuplicateFinderOptions(
        include_exact=False,
        include_perceptual=True,
        phash_max_hamming=14,
        hash_max_side=256,
        parallel_workers=2,
    )

    def log(_: str) -> None:
        pass

    ra = compute_one_signature(a, db, opts, force_recompute=True, on_log=log)
    rb = compute_one_signature(b, db, opts, force_recompute=True, on_log=log)
    assert ra is not None and rb is not None
    groups = build_groups_from_records([ra, rb], opts)
    assert len(groups) >= 1
    assert any(len(g.paths) == 2 for g in groups)


def test_signature_cache_hit_and_force_recompute(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    _solid_png(p, (1, 2, 3))
    db = SignatureDatabase(tmp_path / "s3.sqlite3")
    opts = DuplicateFinderOptions(include_exact=True, include_perceptual=True, parallel_workers=1)

    def log(_: str) -> None:
        pass

    compute_one_signature(p, db, opts, force_recompute=False, on_log=log)
    row1 = db.get_row(str(p.resolve()))
    assert row1 is not None
    assert row1["sig_version"] == SIG_CACHE_VERSION
    sha1 = row1["sha256"]

    compute_one_signature(p, db, opts, force_recompute=False, on_log=log)
    row2 = db.get_row(str(p.resolve()))
    assert row2 is not None
    assert row2["sha256"] == sha1

    _solid_png(p, (200, 200, 200))
    compute_one_signature(p, db, opts, force_recompute=True, on_log=log)
    row3 = db.get_row(str(p.resolve()))
    assert row3 is not None
    assert row3["sha256"] != sha1


def test_load_records_from_db_and_regroup(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _solid_png(a)
    b.write_bytes(a.read_bytes())

    db = SignatureDatabase(tmp_path / "s4.sqlite3")
    opts = DuplicateFinderOptions(include_exact=True, include_perceptual=True, parallel_workers=2)

    def log(_: str) -> None:
        pass

    compute_one_signature(a, db, opts, force_recompute=True, on_log=log)
    compute_one_signature(b, db, opts, force_recompute=True, on_log=log)

    recs = load_records_from_db([a, b], db, opts)
    assert len(recs) == 2

    opts_loose = DuplicateFinderOptions(
        include_exact=True,
        include_perceptual=True,
        phash_max_hamming=20,
        parallel_workers=2,
    )
    groups = regroup_from_cached_records(recs, opts_loose)
    assert len(groups) == 1


def test_build_groups_no_false_positive_different_images(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _solid_png(a, (255, 0, 0))
    _solid_png(b, (0, 0, 255))

    ra = FileDupInfo(
        path=a,
        path_norm=str(a.resolve()),
        size_bytes=a.stat().st_size,
        mtime_ns=int(a.stat().st_mtime_ns),
        width=80,
        height=60,
        sha256="x",
        phash=None,
        dhash=None,
    )
    rb = FileDupInfo(
        path=b,
        path_norm=str(b.resolve()),
        size_bytes=b.stat().st_size,
        mtime_ns=int(b.stat().st_mtime_ns),
        width=80,
        height=60,
        sha256="y",
        phash=None,
        dhash=None,
    )
    opts = DuplicateFinderOptions(include_exact=False, include_perceptual=True, phash_max_hamming=4)
    groups = build_groups_from_records([ra, rb], opts)
    assert groups == []

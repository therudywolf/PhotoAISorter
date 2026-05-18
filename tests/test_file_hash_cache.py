# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Path+mtime file hash cache."""

from __future__ import annotations

from pathlib import Path

from app.file_hash_cache import FileHashCache


def test_hash_cache_avoids_reread(monkeypatch: object, tmp_path: Path) -> None:
    db = tmp_path / "hashes.sqlite3"
    cache = FileHashCache(db)
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    st = f.stat()
    d1 = cache.sha256_for_file(f, mtime_ns=st.st_mtime_ns, size_bytes=st.st_size)
    f.write_bytes(b"changed")
    d2 = cache.sha256_for_file(f, mtime_ns=st.st_mtime_ns, size_bytes=st.st_size)
    assert d1 == d2
    st2 = f.stat()
    d3 = cache.sha256_for_file(f, mtime_ns=st2.st_mtime_ns, size_bytes=st2.st_size)
    assert d3 != d1
    cache.close()

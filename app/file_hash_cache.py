# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Cache file SHA-256 by (path, mtime, size) to avoid full HDD reads every run."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from app.images import file_sha256
from app.paths import file_hash_cache_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_hashes (
    path_norm TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (path_norm, mtime_ns, size_bytes)
);
CREATE INDEX IF NOT EXISTS idx_file_hashes_sha ON file_hashes(sha256);
"""

_instance: FileHashCache | None = None
_lock = threading.Lock()


class FileHashCache:
    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or file_hash_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def sha256_for_file(self, path: Path, *, mtime_ns: int, size_bytes: int) -> str:
        path_norm = str(path.resolve())
        with self._lock:
            row = self._conn.execute(
                "SELECT sha256 FROM file_hashes WHERE path_norm=? AND mtime_ns=? AND size_bytes=?",
                (path_norm, int(mtime_ns), int(size_bytes)),
            ).fetchone()
        if row:
            return str(row[0])
        digest = file_sha256(path)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO file_hashes(path_norm,mtime_ns,size_bytes,sha256,updated_at) "
                "VALUES (?,?,?,?,?)",
                (path_norm, int(mtime_ns), int(size_bytes), digest, time.time()),
            )
        return digest

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def get_file_hash_cache() -> FileHashCache:
    global _instance
    with _lock:
        if _instance is None:
            _instance = FileHashCache()
        return _instance

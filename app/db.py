"""SQLite persistence for file hashes and processing state."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

Status = Literal["pending", "processed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    sha256 TEXT PRIMARY KEY NOT NULL,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processed')),
    category TEXT,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
"""


def default_db_path() -> Path:
    """Store DB under %APPDATA%/PhotoAISorter on Windows, else ~/.local/share/PhotoAISorter."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "PhotoAISorter" / "state.sqlite3"
    return Path.home() / ".local" / "share" / "PhotoAISorter" / "state.sqlite3"


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._pending_writes = 0
        self._last_commit_ts = time.monotonic()
        self._max_pending_writes = 64
        self._max_commit_interval_sec = 0.8
        self._configure_connection()
        self._init_schema()
        self._migrate_pipeline_version()

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.commit()

    def _maybe_commit(self, force: bool = False) -> None:
        now = time.monotonic()
        should_commit = force or self._pending_writes >= self._max_pending_writes or (
            self._pending_writes > 0 and (now - self._last_commit_ts) >= self._max_commit_interval_sec
        )
        if should_commit:
            self._conn.commit()
            self._pending_writes = 0
            self._last_commit_ts = now

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate_pipeline_version(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(files)").fetchall()}
        if "pipeline_version" not in cols:
            with self._lock:
                self._conn.execute("ALTER TABLE files ADD COLUMN pipeline_version TEXT")
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._maybe_commit(force=True)
        self._conn.close()

    def upsert_file_record(
        self, sha256: str, source_path: str, pipeline_version: str
    ) -> str | None:
        """
        Insert or update path for hash. Returns 'skip' if уже обработано с тем же pipeline_version.
        Если версия пайплайна изменилась — сброс в pending и повторная обработка.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT status, pipeline_version FROM files WHERE sha256 = ?", (sha256,)
            ).fetchone()
            now = time.time()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO files (sha256, source_path, status, category, updated_at, pipeline_version)
                    VALUES (?, ?, 'pending', NULL, ?, NULL)
                    """,
                    (sha256, source_path, now),
                )
                self._pending_writes += 1
                self._maybe_commit()
                return None
            if row["status"] == "processed":
                stored = row["pipeline_version"]
                if stored == pipeline_version:
                    self._conn.execute(
                        "UPDATE files SET source_path = ?, updated_at = ? WHERE sha256 = ?",
                        (source_path, now, sha256),
                    )
                    self._pending_writes += 1
                    self._maybe_commit()
                    return "skip"
                self._conn.execute(
                    """
                    UPDATE files SET source_path = ?, status = 'pending', category = NULL,
                    pipeline_version = NULL, updated_at = ? WHERE sha256 = ?
                    """,
                    (source_path, now, sha256),
                )
                self._pending_writes += 1
                self._maybe_commit()
                return None
            self._conn.execute(
                "UPDATE files SET source_path = ?, updated_at = ? WHERE sha256 = ?",
                (source_path, now, sha256),
            )
            self._pending_writes += 1
            self._maybe_commit()
            return None

    def mark_processed(self, sha256: str, category: str, pipeline_version: str) -> None:
        with self._lock:
            now = time.time()
            self._conn.execute(
                """
                UPDATE files SET status = 'processed', category = ?, updated_at = ?, pipeline_version = ?
                WHERE sha256 = ?
                """,
                (category, now, pipeline_version, sha256),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def is_processed(self, sha256: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM files WHERE sha256 = ?", (sha256,)
            ).fetchone()
            return row is not None and row["status"] == "processed"

    def clear_all_records(self) -> int:
        """Удалить все строки (хеши / состояние). Возвращает число удалённых записей."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
            n = int(row[0]) if row else 0
            self._conn.execute("DELETE FROM files")
            self._pending_writes += 1
            self._maybe_commit(force=True)
            return n

    def count_records(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
            return int(row[0]) if row else 0

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""SQLite persistence for file hashes and processing state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Literal

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

CREATE TABLE IF NOT EXISTS sort_sessions (
    session_key TEXT PRIMARY KEY NOT NULL,
    source_dir TEXT NOT NULL,
    dest_dir TEXT NOT NULL,
    media_mode TEXT NOT NULL,
    tag_mode TEXT NOT NULL,
    review_first INTEGER NOT NULL,
    pipeline_version TEXT NOT NULL,
    total_files INTEGER NOT NULL,
    done_files INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT,
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sort_sessions_status_updated ON sort_sessions(status, updated_at);

CREATE TABLE IF NOT EXISTS sort_session_items (
    session_key TEXT NOT NULL,
    path_norm TEXT NOT NULL,
    status TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT,
    category TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (session_key, path_norm)
);
CREATE INDEX IF NOT EXISTS idx_sort_items_session_status ON sort_session_items(session_key, status);
"""


def default_db_path() -> Path:
    """Sort/session SQLite under project tmp/ (fast disk); override via PHOTO_AI_SORTER_TMP."""
    from app.paths import app_state_dir, migrate_app_state_to_project_tmp

    migrate_app_state_to_project_tmp()
    return app_state_dir() / "state.sqlite3"


def make_sort_session_key(
    source_dir: str,
    dest_dir: str,
    media_mode: str,
    tag_mode: str,
    review_first: bool,
    pipeline_version: str,
) -> str:
    raw = "|".join(
        [
            str(Path(source_dir).resolve()),
            str(Path(dest_dir).resolve()),
            str(media_mode),
            str(tag_mode),
            "review" if review_first else "copy",
            str(pipeline_version),
        ]
    ).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


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
            self._conn.execute("DELETE FROM sort_sessions")
            self._conn.execute("DELETE FROM sort_session_items")
            self._pending_writes += 3
            self._maybe_commit(force=True)
            return n

    def count_records(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
            return int(row[0]) if row else 0

    def get_sort_session(self, session_key: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM sort_sessions WHERE session_key = ?",
                (session_key,),
            ).fetchone()

    def latest_incomplete_sort_session(self) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                """
                SELECT * FROM sort_sessions
                WHERE status IN ('running', 'interrupted', 'stopped', 'error')
                  AND (total_files <= 0 OR done_files < total_files)
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()

    def mark_running_sort_sessions_interrupted(self) -> int:
        with self._lock:
            rows = self._conn.execute("SELECT COUNT(*) FROM sort_sessions WHERE status = 'running'").fetchone()
            n = int(rows[0]) if rows else 0
            if n:
                self._conn.execute(
                    "UPDATE sort_sessions SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
                    (time.time(),),
                )
                self._pending_writes += 1
                self._maybe_commit(force=True)
            return n

    def upsert_sort_session(
        self,
        *,
        session_key: str,
        source_dir: str,
        dest_dir: str,
        media_mode: str,
        tag_mode: str,
        review_first: bool,
        pipeline_version: str,
        total_files: int,
        done_files: int,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sort_sessions (
                    session_key, source_dir, dest_dir, media_mode, tag_mode, review_first,
                    pipeline_version, total_files, done_files, status, payload_json,
                    started_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    source_dir = excluded.source_dir,
                    dest_dir = excluded.dest_dir,
                    media_mode = excluded.media_mode,
                    tag_mode = excluded.tag_mode,
                    review_first = excluded.review_first,
                    pipeline_version = excluded.pipeline_version,
                    total_files = excluded.total_files,
                    done_files = excluded.done_files,
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_key,
                    source_dir,
                    dest_dir,
                    media_mode,
                    tag_mode,
                    1 if review_first else 0,
                    pipeline_version,
                    int(total_files),
                    int(done_files),
                    status,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def sort_session_item_status(
        self,
        session_key: str,
        path_norm: str,
        *,
        mtime_ns: int,
        size_bytes: int,
    ) -> str | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT status FROM sort_session_items
                WHERE session_key = ? AND path_norm = ? AND mtime_ns = ? AND size_bytes = ?
                """,
                (session_key, path_norm, int(mtime_ns), int(size_bytes)),
            ).fetchone()
            return str(row["status"]) if row else None

    def mark_sort_session_item(
        self,
        session_key: str,
        path_norm: str,
        *,
        status: str,
        mtime_ns: int,
        size_bytes: int,
        sha256: str | None = None,
        category: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sort_session_items (
                    session_key, path_norm, status, mtime_ns, size_bytes,
                    sha256, category, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_key, path_norm) DO UPDATE SET
                    status = excluded.status,
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    category = excluded.category,
                    updated_at = excluded.updated_at
                """,
                (
                    session_key,
                    path_norm,
                    status,
                    int(mtime_ns),
                    int(size_bytes),
                    sha256,
                    category,
                    time.time(),
                ),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def clear_sort_session(self, session_key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sort_sessions WHERE session_key = ?", (session_key,))
            self._conn.execute("DELETE FROM sort_session_items WHERE session_key = ?", (session_key,))
            self._pending_writes += 2
            self._maybe_commit(force=True)

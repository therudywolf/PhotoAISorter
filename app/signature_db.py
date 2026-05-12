# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""SQLite cache for duplicate-finder signatures and resumable sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from app.db import default_db_path

SIG_CACHE_VERSION = "2026-05-10-v4"


def signatures_db_path() -> Path:
    return default_db_path().parent / "signatures.sqlite3"


def make_session_key(root_path: str, media_mode: str, strictness: str) -> str:
    raw = f"{root_path}|{media_mode}|{strictness}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


class SignatureDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or signatures_db_path()
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

    def close(self) -> None:
        with self._lock:
            self._maybe_commit(force=True)
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS file_signatures (
                path_norm TEXT PRIMARY KEY NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                sha256 TEXT,
                phash_hex TEXT,
                dhash_hex TEXT,
                colorhash_hex TEXT,
                sig_version TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sig_mtime ON file_signatures(mtime_ns);
            CREATE INDEX IF NOT EXISTS idx_sig_sha ON file_signatures(sha256);

            CREATE TABLE IF NOT EXISTS dup_sessions (
                session_key TEXT PRIMARY KEY NOT NULL,
                root_path TEXT NOT NULL,
                media_mode TEXT NOT NULL,
                strictness TEXT NOT NULL,
                stage TEXT NOT NULL,
                total_files INTEGER NOT NULL,
                done_files INTEGER NOT NULL,
                llm_total_pairs INTEGER NOT NULL,
                llm_done_pairs INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dup_session_items (
                session_key TEXT NOT NULL,
                path_norm TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (session_key, path_norm)
            );
            CREATE INDEX IF NOT EXISTS idx_dup_items_session_status ON dup_session_items(session_key, status);

            CREATE TABLE IF NOT EXISTS dup_session_llm_pairs (
                session_key TEXT NOT NULL,
                path_a TEXT NOT NULL,
                path_b TEXT NOT NULL,
                decision INTEGER NOT NULL,
                PRIMARY KEY (session_key, path_a, path_b)
            );
            """
        )
        self._conn.commit()
        self._migrate_file_signatures_columns()

    def _migrate_file_signatures_columns(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(file_signatures)").fetchall()}
        if "phash_frames_json" not in cols:
            self._conn.execute("ALTER TABLE file_signatures ADD COLUMN phash_frames_json TEXT")
            self._conn.commit()
        if "colorhash_hex" not in cols:
            self._conn.execute("ALTER TABLE file_signatures ADD COLUMN colorhash_hex TEXT")
            self._conn.commit()

    def get_row(self, path_norm: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM file_signatures WHERE path_norm = ?", (path_norm,)).fetchone()

    def upsert_signature(
        self,
        path_norm: str,
        mtime_ns: int,
        size_bytes: int,
        width: int | None,
        height: int | None,
        sha256: str | None,
        phash_hex: str | None,
        dhash_hex: str | None,
        colorhash_hex: str | None = None,
        phash_frames_json: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO file_signatures (
                    path_norm, mtime_ns, size_bytes, width, height,
                    sha256, phash_hex, dhash_hex, colorhash_hex, phash_frames_json, sig_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path_norm) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    width = excluded.width,
                    height = excluded.height,
                    sha256 = excluded.sha256,
                    phash_hex = excluded.phash_hex,
                    dhash_hex = excluded.dhash_hex,
                    colorhash_hex = excluded.colorhash_hex,
                    phash_frames_json = excluded.phash_frames_json,
                    sig_version = excluded.sig_version
                """,
                (
                    path_norm,
                    mtime_ns,
                    size_bytes,
                    width,
                    height,
                    sha256,
                    phash_hex,
                    dhash_hex,
                    colorhash_hex,
                    phash_frames_json,
                    SIG_CACHE_VERSION,
                ),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def clear_all(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM file_signatures").fetchone()
            n = int(row[0]) if row else 0
            self._conn.execute("DELETE FROM file_signatures")
            self._pending_writes += 1
            self._maybe_commit(force=True)
            return n

    def count_signatures(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM file_signatures").fetchone()
            return int(row[0]) if row else 0

    def get_session(self, session_key: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM dup_sessions WHERE session_key = ?", (session_key,)).fetchone()

    def upsert_session(
        self,
        *,
        session_key: str,
        root_path: str,
        media_mode: str,
        strictness: str,
        stage: str,
        total_files: int,
        done_files: int,
        llm_total_pairs: int,
        llm_done_pairs: int,
        status: str,
        payload: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dup_sessions (
                    session_key, root_path, media_mode, strictness, stage,
                    total_files, done_files, llm_total_pairs, llm_done_pairs,
                    status, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    stage=excluded.stage,
                    total_files=excluded.total_files,
                    done_files=excluded.done_files,
                    llm_total_pairs=excluded.llm_total_pairs,
                    llm_done_pairs=excluded.llm_done_pairs,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session_key,
                    root_path,
                    media_mode,
                    strictness,
                    stage,
                    int(total_files),
                    int(done_files),
                    int(llm_total_pairs),
                    int(llm_done_pairs),
                    status,
                    json.dumps(payload or {}, ensure_ascii=False),
                    float(time.time()),
                ),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def mark_session_item(self, session_key: str, path_norm: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dup_session_items (session_key, path_norm, status)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key, path_norm) DO UPDATE SET status = excluded.status
                """,
                (session_key, path_norm, status),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def session_item_status(self, session_key: str, path_norm: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM dup_session_items WHERE session_key = ? AND path_norm = ?",
                (session_key, path_norm),
            ).fetchone()
            return str(row["status"]) if row else None

    def save_llm_pair_decision(self, session_key: str, path_a: str, path_b: str, decision: bool) -> None:
        a, b = sorted((path_a, path_b))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dup_session_llm_pairs (session_key, path_a, path_b, decision)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_key, path_a, path_b) DO UPDATE SET decision = excluded.decision
                """,
                (session_key, a, b, 1 if decision else 0),
            )
            self._pending_writes += 1
            self._maybe_commit()

    def get_llm_pair_decision(self, session_key: str, path_a: str, path_b: str) -> bool | None:
        a, b = sorted((path_a, path_b))
        with self._lock:
            row = self._conn.execute(
                "SELECT decision FROM dup_session_llm_pairs WHERE session_key = ? AND path_a = ? AND path_b = ?",
                (session_key, a, b),
            ).fetchone()
            if not row:
                return None
            return bool(int(row["decision"]))

    def list_llm_pair_decisions(self, session_key: str) -> dict[tuple[str, str], bool]:
        """All stored LLM duplicate decisions for a session (sorted path_a, path_b keys)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path_a, path_b, decision FROM dup_session_llm_pairs WHERE session_key = ?",
                (session_key,),
            ).fetchall()
        out: dict[tuple[str, str], bool] = {}
        for row in rows:
            out[(str(row["path_a"]), str(row["path_b"]))] = bool(int(row["decision"]))
        return out

    def clear_session(self, session_key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM dup_sessions WHERE session_key = ?", (session_key,))
            self._conn.execute("DELETE FROM dup_session_items WHERE session_key = ?", (session_key,))
            self._conn.execute("DELETE FROM dup_session_llm_pairs WHERE session_key = ?", (session_key,))
            self._pending_writes += 3
            self._maybe_commit(force=True)

    def session_payload(self, session_key: str) -> dict[str, Any]:
        row = self.get_session(session_key)
        if not row or not row["payload_json"]:
            return {}
        try:
            return json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return {}

    def update_session_payload(self, session_key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE dup_sessions SET payload_json = ?, updated_at = ? WHERE session_key = ?",
                (json.dumps(payload or {}, ensure_ascii=False), float(time.time()), session_key),
            )
            self._pending_writes += 1
            self._maybe_commit(force=True)

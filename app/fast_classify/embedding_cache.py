# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Persistent SQLite cache for CLIP image embeddings keyed by content hash."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from app.paths import clip_embedding_cache_path, migrate_app_state_to_project_tmp

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clip_embeddings (
    sha256 TEXT NOT NULL,
    model_key TEXT NOT NULL,
    image_max_side INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (sha256, model_key, image_max_side)
);
"""


def _cache_path() -> Path:
    migrate_app_state_to_project_tmp()
    return clip_embedding_cache_path()


class EmbeddingCache:
    def __init__(self, model_key: str, image_max_side: int) -> None:
        self._model_key = model_key
        self._image_max_side = int(image_max_side)
        self._lock = threading.Lock()
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def get_many(self, digests: list[str]) -> dict[str, np.ndarray]:
        if not digests:
            return {}
        out: dict[str, np.ndarray] = {}
        with self._lock:
            cur = self._conn.cursor()
            chunk = 400
            for i in range(0, len(digests), chunk):
                part = digests[i : i + chunk]
                qmarks = ",".join("?" * len(part))
                rows = cur.execute(
                    f"SELECT sha256, dim, vec FROM clip_embeddings "
                    f"WHERE model_key=? AND image_max_side=? AND sha256 IN ({qmarks})",
                    (self._model_key, self._image_max_side, *part),
                ).fetchall()
                for sha, dim, blob in rows:
                    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                    if arr.size == dim:
                        out[sha] = arr
        return out

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        import time

        now = time.time()
        rows = []
        for sha, vec in items:
            v16 = vec.astype(np.float16, copy=False)
            rows.append(
                (sha, self._model_key, self._image_max_side, int(v16.size), v16.tobytes(), now)
            )
        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.executemany(
                "INSERT OR REPLACE INTO clip_embeddings "
                "(sha256, model_key, image_max_side, dim, vec, created_at) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def model_cache_key(settings) -> str:  # type: ignore[no-untyped-def]
    device = getattr(settings, "device", "auto")
    fp16 = int(bool(getattr(settings, "use_fp16", True)))
    side = int(getattr(settings, "image_max_side", 384))
    return f"{settings.model_name}|{settings.pretrained}|{device}|fp16={fp16}|side={side}"

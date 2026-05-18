# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Hybrid sort path integration."""

from __future__ import annotations

from pathlib import Path
from queue import Queue

from app.classification_result import ClassificationResult
from app.constants import MediaScanMode
from app.db import Database
from app.tag_config import ResolvedTagConfig, TagMode
from app.worker import SortWorker


def test_worker_hybrid_uses_run_hybrid_sort(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.jpg").write_bytes(b"\xff\xd8\xff\xd8")

    called = {"n": 0}

    def fake_hybrid(*_a, **_k) -> None:
        called["n"] += 1

    monkeypatch.setattr("app.sort_hybrid.run_hybrid_sort", fake_hybrid)

    wl = frozenset({"cat", "uncategorized"})
    cfg = ResolvedTagConfig(
        mode=TagMode.HYBRID,
        categories=tuple(wl),
        prompts={"cat": "cat", "uncategorized": "x"},
        whitelist=wl,
    )
    db = Database(tmp_path / "state.sqlite3")
    q: Queue = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, tag_config=cfg)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)
    assert called["n"] == 1
    db.close()


def test_sort_hybrid_copies_clip_result(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "photo.jpg").write_bytes(b"x")

    class _FakeFast:
        def classify_batch(self, paths: list[Path], *, digests=None) -> list[ClassificationResult]:
            return [
                ClassificationResult("cat", ["cat"], 0.9, "clip_similarity", False, "")
                for _ in paths
            ]

    monkeypatch.setattr("app.sort_hybrid.clip_available", lambda: True)
    monkeypatch.setattr("app.sort_hybrid.get_classifier", lambda *a, **k: _FakeFast())
    from app.fast_classify.config import FastClassifySettings

    monkeypatch.setattr(
        "app.sort_hybrid.load_fast_classify_settings",
        lambda *a, **k: FastClassifySettings(batch_size=8, confidence_threshold=0.2, vlm_fallback=False),
    )

    db = Database(tmp_path / "state.sqlite3")
    q: Queue = Queue()
    wl = frozenset({"cat", "uncategorized"})
    cfg = ResolvedTagConfig(
        mode=TagMode.HYBRID,
        categories=tuple(wl),
        prompts={"cat": "c", "uncategorized": "u"},
        whitelist=wl,
    )
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, tag_config=cfg)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "cat" / "photo.jpg").exists()
    db.close()

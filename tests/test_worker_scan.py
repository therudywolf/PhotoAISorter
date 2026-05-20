# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Media file iteration by scan mode."""

import threading
import time
from pathlib import Path
from queue import Queue

from app.constants import PIPELINE_VERSION, MediaScanMode
from app.db import Database, make_sort_session_key
from app.images import file_sha256
from app.worker import SortWorker, iter_media_files


def test_iter_photos_only_skips_video(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.mp4").write_bytes(b"x")
    out = iter_media_files(tmp_path, MediaScanMode.PHOTOS_ONLY)
    assert len(out) == 1
    assert out[0].suffix.lower() == ".jpg"


def test_iter_video_only(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.mp4").write_bytes(b"x")
    out = iter_media_files(tmp_path, MediaScanMode.VIDEO_ONLY)
    assert len(out) == 1
    assert out[0].suffix.lower() == ".mp4"


def test_sort_worker_clamps_workers_1_to_16(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w0 = SortWorker(db, q, api_base="http://x", model="m", workers=0)
    w9 = SortWorker(db, q, api_base="http://x", model="m", workers=9)
    w_over = SortWorker(db, q, api_base="http://x", model="m", workers=20)
    a0 = SortWorker(db, q, api_base="http://x", model="m", api_workers=0)
    a9 = SortWorker(db, q, api_base="http://x", model="m", api_workers=9)
    a_over = SortWorker(db, q, api_base="http://x", model="m", api_workers=20)
    assert w0.workers == 1
    assert w9.workers == 9
    assert w_over.workers == 16
    assert a0.api_workers == 1
    assert a9.api_workers == 9
    assert a_over.api_workers == 16
    db.close()


def test_sort_worker_strict_mode_forces_uncategorized_for_unknown_tag(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", lambda *_a, **_k: "new/custom/tag")

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, free_tag_mode=False)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "uncategorized" / "a.jpg").exists()
    assert not (dst / "new" / "custom" / "tag" / "a.jpg").exists()
    db.close()


def test_sort_worker_free_mode_allows_model_tag_folder(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", lambda *_a, **_k: "new/custom/tag")

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, free_tag_mode=True)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "new" / "custom" / "tag" / "a.jpg").exists()
    db.close()


def test_sort_worker_auto_mode_chooses_popular_candidate(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr(
        "app.worker.chat_completion_cfg",
        lambda *_a, **_k: "city/night, nature/forest/sunset, nature/forest/sunset",
    )

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, auto_tag_mode=True)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "nature" / "forest" / "a.jpg").exists()
    db.close()


def test_sort_worker_general_mode_accepts_extended_preset(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr(
        "app.worker.chat_completion_cfg",
        lambda *_a, **_k: '{"primary_category": "coding_ide_and_terminal", "confidence": 0.91}',
    )

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, general_tag_mode=True)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "coding_ide_and_terminal" / "a.jpg").exists()
    db.close()


def test_sort_worker_serializes_lm_requests_when_api_workers_is_one(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for i in range(4):
        (src / f"{i}.jpg").write_bytes(f"fake-{i}".encode("ascii"))

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def fake_chat(*_a: object, **_k: object) -> str:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return '{"primary_category": "vehicles_and_racing", "confidence": 0.9}'

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=4, api_workers=1)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert state["max_active"] == 1
    assert len(list((dst / "vehicles_and_racing").glob("*.jpg"))) == 4
    db.close()


def test_sort_worker_retries_uncategorized_then_succeeds(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    attempts = {"n": 0}

    def fake_chat(*_a: object, **_k: object) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return "uncategorized"
        return "vehicles_and_racing"

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, free_tag_mode=False)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert attempts["n"] == 3
    assert (dst / "vehicles_and_racing" / "a.jpg").exists()
    db.close()


def test_sort_worker_video_structured_uses_contact_sheet_once(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.mp4"
    f.write_bytes(b"fake-video")

    calls = {"chat": 0}

    def fake_chat(*_a: object, **_k: object) -> str:
        calls["chat"] += 1
        return '{"primary_category": "vehicles_and_racing", "confidence": 0.86}'

    monkeypatch.setattr("app.worker.extract_frames_reduced", lambda *_a, **_k: [object(), object()])
    monkeypatch.setattr("app.worker.video_contact_sheet_data_uri", lambda _frames: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.VIDEO_ONLY)

    assert calls == {"chat": 1}
    assert (dst / "vehicles_and_racing" / "a.mp4").exists()
    db.close()


def test_sort_worker_video_structured_falls_back_to_single_frame_on_api_error(
    tmp_path: Path, monkeypatch: object
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.mp4"
    f.write_bytes(b"fake-video")

    calls = {"chat": 0}

    def fake_chat(*_a: object, **_k: object) -> str:
        calls["chat"] += 1
        if calls["chat"] == 1:
            raise RuntimeError("400 Bad Request")
        return '{"primary_category": "vehicles_and_racing", "confidence": 0.86}'

    monkeypatch.setattr("app.worker.extract_frames_reduced", lambda *_a, **_k: [object(), object()])
    monkeypatch.setattr("app.worker.video_contact_sheet_data_uri", lambda _frames: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.pil_image_to_jpeg_data_uri", lambda *_a, **_k: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.VIDEO_ONLY)

    assert calls == {"chat": 2}
    assert (dst / "vehicles_and_racing" / "a.mp4").exists()
    db.close()


def test_sort_worker_retries_api_error_then_succeeds(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    attempts = {"n": 0}

    def fake_chat(*_a: object, **_k: object) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Channel Error")
        return "vehicles_and_racing"

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, free_tag_mode=False)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert attempts["n"] == 3
    assert (dst / "vehicles_and_racing" / "a.jpg").exists()
    db.close()


def test_sort_worker_no_space_leaves_file_pending(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", lambda *_a, **_k: "vehicles_and_racing")
    monkeypatch.setattr("app.worker.has_disk_space_for_copy", lambda *_a, **_k: False)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert not (dst / "vehicles_and_racing" / "a.jpg").exists()
    assert db.count_records() == 1
    assert db.is_processed(file_sha256(f)) is False
    db.close()


def test_sort_worker_skips_existing_output_folder_inside_source(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = src / "sorted"
    dst.mkdir(parents=True)
    keep = src / "a.jpg"
    old_output = dst / "old.jpg"
    keep.write_bytes(b"fake-a")
    old_output.write_bytes(b"fake-old")

    attempts = {"n": 0}

    def fake_chat(*_a: object, **_k: object) -> str:
        attempts["n"] += 1
        return "vehicles_and_racing"

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert attempts["n"] == 1
    assert (dst / "vehicles_and_racing" / "a.jpg").exists()
    assert not (dst / "vehicles_and_racing" / "old.jpg").exists()
    db.close()


def test_sort_worker_review_first_writes_manifest_without_copy(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr(
        "app.worker.chat_completion_cfg",
        lambda *_a, **_k: '{"primary_category": "vehicles_and_racing", "confidence": 0.9}',
    )

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, review_first=True)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    manifests = list((dst / "_review_runs").glob("sort-*/manifest.jsonl"))
    assert len(manifests) == 1
    assert "vehicles_and_racing" in manifests[0].read_text(encoding="utf-8")
    assert not (dst / "vehicles_and_racing" / "a.jpg").exists()
    assert db.is_processed(file_sha256(f)) is False
    db.close()


def test_sort_worker_resume_session_skips_done_path(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", lambda *_a, **_k: "vehicles_and_racing")

    db = Database(tmp_path / "state.sqlite3")
    key = make_sort_session_key(
        str(src.resolve()),
        str(dst.resolve()),
        MediaScanMode.PHOTOS_ONLY.value,
        "strict",
        False,
        PIPELINE_VERSION,
    )
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, session_key=key)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)
    assert (dst / "vehicles_and_racing" / "a.jpg").exists()

    def fail_chat(*_a: object, **_k: object) -> str:
        raise AssertionError("resume should skip already completed path before API")

    monkeypatch.setattr("app.worker.chat_completion_cfg", fail_chat)
    q2 = Queue()
    w2 = SortWorker(
        db,
        q2,
        api_base="http://x",
        model="m",
        workers=1,
        session_key=key,
        resume_session=True,
    )
    w2.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert len(list((dst / "vehicles_and_racing").glob("a*.jpg"))) == 1
    db.close()

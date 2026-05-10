"""Media file iteration by scan mode."""

from pathlib import Path
from queue import Queue

from app.constants import MediaScanMode
from app.db import Database
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


def test_sort_worker_clamps_workers_1_to_4(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w0 = SortWorker(db, q, api_base="http://x", model="m", workers=0)
    w9 = SortWorker(db, q, api_base="http://x", model="m", workers=9)
    assert w0.workers == 1
    assert w9.workers == 4
    db.close()


def test_sort_worker_strict_mode_forces_uncategorized_for_unknown_tag(tmp_path: Path, monkeypatch: object) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    f = src / "a.jpg"
    f.write_bytes(b"fake")

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion", lambda *_a, **_k: "new/custom/tag")

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
    monkeypatch.setattr("app.worker.chat_completion", lambda *_a, **_k: "new/custom/tag")

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
        "app.worker.chat_completion",
        lambda *_a, **_k: "city/night, nature/forest/sunset, nature/forest/sunset",
    )

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, auto_tag_mode=True)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "nature" / "forest" / "a.jpg").exists()
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
    monkeypatch.setattr("app.worker.chat_completion", fake_chat)

    db = Database(tmp_path / "state.sqlite3")
    q = Queue()
    w = SortWorker(db, q, api_base="http://x", model="m", workers=1, free_tag_mode=False)
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert attempts["n"] == 3
    assert (dst / "vehicles_and_racing" / "a.jpg").exists()
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
    monkeypatch.setattr("app.worker.chat_completion", fake_chat)

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
    monkeypatch.setattr("app.worker.chat_completion", lambda *_a, **_k: "vehicles_and_racing")
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
    monkeypatch.setattr("app.worker.chat_completion", fake_chat)

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
        "app.worker.chat_completion",
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

from pathlib import Path

from app.constants import PIPELINE_VERSION
from app.db import Database, make_sort_session_key
from app.signature_db import SignatureDatabase, make_session_key


def test_session_roundtrip(tmp_path: Path) -> None:
    db = SignatureDatabase(tmp_path / "sig.sqlite3")
    key = make_session_key("c:/x", "photos_only", "strict")

    db.upsert_session(
        session_key=key,
        root_path="c:/x",
        media_mode="photos_only",
        strictness="strict",
        stage="scan_signatures",
        total_files=10,
        done_files=3,
        llm_total_pairs=0,
        llm_done_pairs=0,
        status="running",
        payload={"paths": ["a", "b"]},
    )
    row = db.get_session(key)
    assert row is not None
    assert row["stage"] == "scan_signatures"
    assert int(row["done_files"]) == 3

    db.mark_session_item(key, "a", "done")
    assert db.session_item_status(key, "a") == "done"

    db.save_llm_pair_decision(key, "a", "b", True)
    assert db.get_llm_pair_decision(key, "a", "b") is True

    payload = db.session_payload(key)
    assert payload.get("paths") == ["a", "b"]

    db.clear_session(key)
    assert db.get_session(key) is None


def test_sort_session_roundtrip(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.sqlite3")
    key = make_sort_session_key("c:/src", "c:/dst", "photos_only", "strict", False, PIPELINE_VERSION)
    db.upsert_sort_session(
        session_key=key,
        source_dir="c:/src",
        dest_dir="c:/dst",
        media_mode="photos_only",
        tag_mode="strict",
        review_first=False,
        pipeline_version=PIPELINE_VERSION,
        total_files=100,
        done_files=12,
        status="running",
        payload={"model": "m"},
    )
    row = db.get_sort_session(key)
    assert row is not None
    assert int(row["done_files"]) == 12
    assert db.mark_running_sort_sessions_interrupted() == 1
    row = db.latest_incomplete_sort_session()
    assert row is not None
    assert row["status"] == "interrupted"

    db.mark_sort_session_item(
        key,
        "c:/src/a.jpg",
        status="done",
        mtime_ns=123,
        size_bytes=456,
        sha256="abc",
        category="vehicles",
    )
    assert db.sort_session_item_status(key, "c:/src/a.jpg", mtime_ns=123, size_bytes=456) == "done"
    assert db.sort_session_item_status(key, "c:/src/a.jpg", mtime_ns=124, size_bytes=456) is None
    db.clear_sort_session(key)
    assert db.get_sort_session(key) is None
    db.close()


def test_clear_sort_cache_removes_sort_sessions(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.sqlite3")
    key = make_sort_session_key("c:/src", "c:/dst", "photos_only", "strict", False, PIPELINE_VERSION)
    db.upsert_sort_session(
        session_key=key,
        source_dir="c:/src",
        dest_dir="c:/dst",
        media_mode="photos_only",
        tag_mode="strict",
        review_first=False,
        pipeline_version=PIPELINE_VERSION,
        total_files=10,
        done_files=1,
        status="stopped",
        payload={},
    )
    db.clear_all_records()
    assert db.get_sort_session(key) is None
    db.close()

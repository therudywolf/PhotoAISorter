from pathlib import Path

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

"""Export duplicate-finder results to disk next to the app database (readable JSON cache)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.db import default_db_path


def duplicate_runs_base_dir() -> Path:
    return default_db_path().parent / "duplicate_runs"


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def export_duplicate_run(
    session_key: str,
    *,
    root_path: str,
    media_mode: str,
    strictness: str,
    groups: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> Path:
    """Write meta.json + result.json under duplicate_runs/<session_key>/. Returns output directory."""
    out = duplicate_runs_base_dir() / session_key
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_key": session_key,
        "root_path": root_path,
        "media_mode": media_mode,
        "strictness": strictness,
        "exported_at": time.time(),
        "groups_count": len(groups),
        "records_count": len(records),
    }
    atomic_write_json(out / "meta.json", meta)
    atomic_write_json(out / "result.json", {"groups": groups, "records": records})
    return out

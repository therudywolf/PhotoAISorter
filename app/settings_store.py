"""Persist GUI fields next to the app data directory (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.db import default_db_path


def settings_json_path() -> Path:
    return default_db_path().parent / "gui_settings.json"


def duplicate_journal_path() -> Path:
    return default_db_path().parent / "duplicate_delete_journal.jsonl"


def load_gui_settings() -> dict[str, Any]:
    p = settings_json_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_gui_settings(data: dict[str, Any]) -> None:
    p = settings_json_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

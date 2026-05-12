# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""GUI settings and secret persistence."""

from pathlib import Path

from app import settings_store


def test_secret_settings_roundtrip(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setattr(settings_store, "default_db_path", lambda: tmp_path / "state.sqlite3")
    settings_store.save_secret_settings({"lm_studio_api_key": "secret"})
    assert settings_store.load_secret_settings() == {"lm_studio_api_key": "secret"}
    settings_store.save_secret_settings({"lm_studio_api_key": ""})
    assert settings_store.load_secret_settings() == {}

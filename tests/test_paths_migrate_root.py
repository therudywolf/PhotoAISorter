# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path

from app.paths import migrate_legacy_project_root


def test_migrate_root_local_presets_to_app_state(
    monkeypatch: object, tmp_path: Path
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "local_presets.json").write_text(
        json.dumps({"sets": [{"name": "A", "tags": [{"key": "dog"}]}], "active_set": "A"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.paths.project_root", lambda: root)
    monkeypatch.setattr("app.paths.project_tmp_dir", lambda: root / "tmp")
    monkeypatch.setattr("app.paths.app_state_dir", lambda: root / "tmp" / "app_state")
    monkeypatch.setattr("app.paths.clip_weights_dir", lambda: root / "data" / "clip_weights")

    migrate_legacy_project_root()

    dst = root / "tmp" / "app_state" / "context_tags.json"
    assert dst.is_file()
    assert not (root / "local_presets.json").is_file()
    assert (root / "local_presets.json.migrated").is_file()

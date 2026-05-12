# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Small .env loader for local, non-committed runtime settings."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_project_env(project_root: Path | None = None) -> list[Path]:
    """Load .env.local/.env without overriding real environment variables."""
    root = project_root or Path(__file__).resolve().parent.parent
    loaded: list[Path] = []
    for name in (".env.local", ".env"):
        path = root / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _strip_quotes(value)
        loaded.append(path)
    return loaded

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""User-editable aliases for smart auto categories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.db import default_db_path

# Virtual CLIP tags (separate refs folders) map to parent on-disk sort folders.
BUILTIN_STORAGE_ALIASES: dict[str, str] = {
    "iam_face": "iam",
    "iam_body": "iam",
    "iam_tattoo": "iam",
    "my_dog_closeup": "my_dog",
    "my_dog_fullbody": "my_dog",
    "my_dog_alt": "my_dog",
    "my_cat_closeup": "my_cat",
    "my_cat_fullbody": "my_cat",
}

_ALIAS_KEY_RE = re.compile(r"[^a-z0-9_/\-\s]+")
_SPACES_DASH_RE = re.compile(r"[\s\-]+")
_UNDERSCORE_RE = re.compile(r"_+")


def category_aliases_path() -> Path:
    return default_db_path().parent / "category_aliases.json"


def _clean_alias_part(raw: str) -> str:
    txt = _ALIAS_KEY_RE.sub(" ", str(raw or "").strip().lower())
    txt = _SPACES_DASH_RE.sub("_", txt)
    txt = _UNDERSCORE_RE.sub("_", txt).strip("_/")
    return txt[:96]


def clean_alias_tag(raw: str) -> str:
    parts = [_clean_alias_part(p) for p in str(raw or "").split("/")]
    return "/".join(p for p in parts if p)[:128].strip("/")


def normalize_aliases(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        key = clean_alias_tag(str(k))
        val = clean_alias_tag(str(v))
        if key and val and key != val:
            out[key] = val
    return out


def load_category_aliases(path: Path | None = None) -> dict[str, str]:
    p = path or category_aliases_path()
    if not p.is_file():
        return {}
    try:
        return normalize_aliases(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def save_category_aliases(aliases: dict[str, str], path: Path | None = None) -> None:
    p = path or category_aliases_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_aliases(aliases)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def resolve_storage_category(category: str, aliases: dict[str, str] | None = None) -> str:
    """Map a model tag to the on-disk folder name (virtual tags → parent folder)."""
    key = clean_alias_tag(category)
    if not key:
        return category
    normalized = normalize_aliases(aliases or {})
    if key in normalized:
        return normalized[key]
    return BUILTIN_STORAGE_ALIASES.get(key, key)


def aliases_to_prompt_lines(aliases: dict[str, str]) -> str:
    normalized = normalize_aliases(aliases)
    if not normalized:
        return ""
    lines = ["Category alias rules (apply before creating new folders):"]
    for src, dst in sorted(normalized.items()):
        lines.append(f"- {src} => {dst}")
    return "\n".join(lines)

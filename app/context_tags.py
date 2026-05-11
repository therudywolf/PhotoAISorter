"""User-defined context tags for AI classification.

Context tags are named descriptions (e.g. 'my_dog', 'me') that get injected
into the system prompt so the model can recognize personal subjects.
Stored locally in context_tags.json alongside other app settings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.db import default_db_path


def _tags_json_path() -> Path:
    return default_db_path().parent / "context_tags.json"


@dataclass
class ContextTag:
    key: str
    label: str
    description: str
    enabled: bool = True


@dataclass
class CustomCategoryList:
    name: str
    categories: list[str] = field(default_factory=list)


@dataclass
class ContextTagStore:
    tags: list[ContextTag] = field(default_factory=list)
    custom_lists: list[CustomCategoryList] = field(default_factory=list)


def load_context_tags() -> ContextTagStore:
    p = _tags_json_path()
    if not p.is_file():
        return ContextTagStore()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return ContextTagStore()
        tags = []
        for t in raw.get("tags", []):
            if isinstance(t, dict) and t.get("key"):
                tags.append(ContextTag(
                    key=str(t["key"]),
                    label=str(t.get("label", t["key"])),
                    description=str(t.get("description", "")),
                    enabled=bool(t.get("enabled", True)),
                ))
        custom_lists = []
        for cl in raw.get("custom_lists", []):
            if isinstance(cl, dict) and cl.get("name"):
                custom_lists.append(CustomCategoryList(
                    name=str(cl["name"]),
                    categories=[str(c) for c in cl.get("categories", []) if c],
                ))
        return ContextTagStore(tags=tags, custom_lists=custom_lists)
    except (OSError, json.JSONDecodeError):
        return ContextTagStore()


def save_context_tags(store: ContextTagStore) -> None:
    p = _tags_json_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "tags": [asdict(t) for t in store.tags],
        "custom_lists": [asdict(cl) for cl in store.custom_lists],
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def build_user_context_from_tags(store: ContextTagStore) -> str:
    """Format enabled context tags into a text block for the system prompt."""
    lines: list[str] = []
    for tag in store.tags:
        if tag.enabled and tag.description.strip():
            lines.append(f"{tag.key}: {tag.description.strip()}")
    return "\n".join(lines)


def get_active_custom_list(store: ContextTagStore, name: str) -> list[str] | None:
    """Return category list by name, or None if not found."""
    for cl in store.custom_lists:
        if cl.name == name:
            return cl.categories
    return None

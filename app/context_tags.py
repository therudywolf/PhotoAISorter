# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""User-defined tag sets for AI classification.

A TagSet is a named collection of tags. Each tag has:
- key: the output category / folder name
- description: optional instruction for the model (what to look for)

When a custom TagSet is active, the model uses ONLY those tags as output categories.
Tags with descriptions get their descriptions injected into the prompt,
giving the model recognition instructions (e.g. "my_pet: Black Labrador, friendly").

Stored locally in context_tags.json.
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
class Tag:
    key: str
    description: str = ""


@dataclass
class TagSet:
    name: str
    tags: list[Tag] = field(default_factory=list)


@dataclass
class TagStore:
    sets: list[TagSet] = field(default_factory=list)
    active_set: str = ""


def load_tag_store() -> TagStore:
    p = _tags_json_path()
    if not p.is_file():
        return TagStore()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return TagStore()
        sets: list[TagSet] = []
        for s in raw.get("sets", []):
            if isinstance(s, dict) and s.get("name"):
                tags = []
                for t in s.get("tags", []):
                    if isinstance(t, dict) and t.get("key"):
                        tags.append(Tag(
                            key=str(t["key"]).strip(),
                            description=str(t.get("description", "")).strip(),
                        ))
                    elif isinstance(t, str) and t.strip():
                        tags.append(Tag(key=t.strip()))
                sets.append(TagSet(name=str(s["name"]), tags=tags))
        # Backward compat: migrate old format (separate tags + custom_lists)
        if not sets and "tags" in raw:
            migrated_tags = []
            for t in raw.get("tags", []):
                if isinstance(t, dict) and t.get("key"):
                    migrated_tags.append(Tag(
                        key=str(t["key"]),
                        description=str(t.get("description", "")),
                    ))
            if migrated_tags:
                sets.append(TagSet(name="My tags", tags=migrated_tags))
            for cl in raw.get("custom_lists", []):
                if isinstance(cl, dict) and cl.get("name"):
                    cl_tags = [Tag(key=c) for c in cl.get("categories", []) if c]
                    if cl_tags:
                        sets.append(TagSet(name=str(cl["name"]), tags=cl_tags))
        active = str(raw.get("active_set", "")).strip()
        return TagStore(sets=sets, active_set=active)
    except (OSError, json.JSONDecodeError):
        return TagStore()


def save_tag_store(store: TagStore) -> None:
    p = _tags_json_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "sets": [asdict(s) for s in store.sets],
        "active_set": store.active_set,
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def get_active_set(store: TagStore) -> TagSet | None:
    """Return the currently active TagSet, or None if none selected."""
    for s in store.sets:
        if s.name == store.active_set:
            return s
    return None


def build_custom_categories(tag_set: TagSet) -> tuple[str, ...]:
    """Extract category keys from a TagSet (the whitelist for the model)."""
    return tuple(t.key for t in tag_set.tags if t.key)


def build_custom_prompts(tag_set: TagSet) -> dict[str, str]:
    """Build category->description map from a TagSet (every tag gets a CLIP description)."""
    out: dict[str, str] = {}
    for t in tag_set.tags:
        if not t.key:
            continue
        desc = (t.description or "").strip()
        if not desc:
            desc = t.key.replace("_", " ")
        out[t.key] = desc
    return out


def build_user_context_from_tags(tag_set: TagSet | None) -> str:
    """Format tag descriptions into USER_CONTEXT block for the system prompt."""
    if not tag_set:
        return ""
    lines: list[str] = []
    for tag in tag_set.tags:
        if tag.description.strip():
            lines.append(f"{tag.key}: {tag.description.strip()}")
    return "\n".join(lines)


# Legacy compatibility aliases
def load_context_tags():
    """Legacy: returns a TagStore (renamed from ContextTagStore)."""
    return load_tag_store()

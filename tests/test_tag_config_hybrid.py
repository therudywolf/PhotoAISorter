# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""TagMode.HYBRID resolution."""

from __future__ import annotations

from app.context_tags import Tag, TagSet, TagStore
from app.tag_config import TagMode, resolve_tag_config


def test_resolve_hybrid_uses_active_tag_set() -> None:
    store = TagStore(
        active_set="forest",
        sets=[TagSet(name="forest", tags=[Tag("iam", "me"), Tag("my_dog", "dog")])],
    )
    cfg = resolve_tag_config(TagMode.HYBRID, tag_store=store)
    assert cfg.mode == TagMode.HYBRID
    assert cfg.categories == ("iam", "my_dog", "uncategorized")
    assert cfg.is_strict_whitelist
    assert cfg.whitelist == frozenset({"iam", "my_dog", "uncategorized"})

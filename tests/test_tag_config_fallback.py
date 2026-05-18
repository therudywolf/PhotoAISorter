# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Custom/hybrid tag config must not silently downgrade to preset."""

from __future__ import annotations

from app.constants import UNCATEGORIZED
from app.context_tags import Tag, TagSet, TagStore
from app.tag_config import TagMode, resolve_tag_config


def test_hybrid_missing_active_set_keeps_mode_empty_categories() -> None:
    store = TagStore(active_set="missing", sets=[])
    cfg = resolve_tag_config(TagMode.HYBRID, tag_store=store)
    assert cfg.mode == TagMode.HYBRID
    assert cfg.categories == ()
    assert cfg.whitelist == frozenset()


def test_hybrid_adds_uncategorized_to_custom_list() -> None:
    store = TagStore(
        active_set="pets",
        sets=[TagSet(name="pets", tags=[Tag("my_dog", "black lab")])],
    )
    cfg = resolve_tag_config(TagMode.HYBRID, tag_store=store)
    assert cfg.mode == TagMode.HYBRID
    assert "my_dog" in cfg.categories
    assert UNCATEGORIZED in cfg.categories
    assert UNCATEGORIZED in cfg.whitelist


def test_custom_empty_tags_keeps_custom_mode() -> None:
    store = TagStore(active_set="x", sets=[TagSet(name="x", tags=[])])
    cfg = resolve_tag_config(TagMode.CUSTOM, tag_store=store)
    assert cfg.mode == TagMode.CUSTOM
    assert cfg.categories == ()

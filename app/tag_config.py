# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Unified tag configuration: TagMode enum, SearchProfile, ResolvedTagConfig, and resolution logic.

This module is the single source of truth for how tag modes map to
categories, descriptions, priority rules, and prompt behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.constants import (
    CATEGORY_PROMPTS,
    PRIORITY_RULES_BLOCK,
    TAG_MERGE_PRIORITY,
    UNCATEGORIZED,
    SearchProfile,
    categories_for_profile,
)


class TagMode(str, Enum):
    PRESET = "preset"
    AUTO = "auto"
    FREE = "free"
    CUSTOM = "custom"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ResolvedTagConfig:
    """Everything the classification pipeline needs to know about tags.

    Built once at sort start, then threaded through the entire pipeline
    instead of separate booleans/dicts/tuples.
    """
    mode: TagMode
    categories: tuple[str, ...]
    prompts: dict[str, str] = field(default_factory=dict)
    priority: tuple[str, ...] = ()
    priority_rules_text: str = ""
    user_context: str = ""
    whitelist: frozenset[str] | None = None
    profile: SearchProfile = SearchProfile.SFW

    @property
    def is_free(self) -> bool:
        return self.mode in (TagMode.FREE, TagMode.AUTO)

    @property
    def is_strict_whitelist(self) -> bool:
        return self.mode in (TagMode.PRESET, TagMode.CUSTOM, TagMode.HYBRID)


def resolve_tag_config(
    mode: TagMode,
    *,
    profile: SearchProfile = SearchProfile.SFW,
    tag_store: Any = None,
    user_context_override: str = "",
) -> ResolvedTagConfig:
    """Build a complete tag config from mode + profile + optional custom TagStore."""
    from app.context_tags import (
        build_custom_categories,
        build_custom_prompts,
        build_user_context_from_tags,
        get_active_set,
    )

    if mode in (TagMode.CUSTOM, TagMode.HYBRID):
        if tag_store is None:
            from app.context_tags import load_tag_store
            tag_store = load_tag_store()
        active = get_active_set(tag_store)
        if active and active.tags:
            cats = _custom_categories_with_fallback(build_custom_categories(active))
            prompts = build_custom_prompts(active)
            if UNCATEGORIZED not in prompts:
                prompts = {**prompts, UNCATEGORIZED: "none of the listed categories"}
            return ResolvedTagConfig(
                mode=mode,
                categories=cats,
                prompts=prompts,
                priority=(),
                priority_rules_text="",
                user_context="",
                whitelist=frozenset(cats) if cats else None,
                profile=profile,
            )
        return _empty_custom_config(mode, profile)

    ctx = user_context_override
    if not ctx and tag_store is not None:
        active = get_active_set(tag_store)
        if active:
            ctx = build_user_context_from_tags(active)

    if mode == TagMode.PRESET:
        cats = categories_for_profile(profile)
        return ResolvedTagConfig(
            mode=TagMode.PRESET,
            categories=cats,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=frozenset(cats),
            profile=profile,
        )
    elif mode == TagMode.AUTO:
        cats = categories_for_profile(profile)
        return ResolvedTagConfig(
            mode=TagMode.AUTO,
            categories=cats,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=None,
            profile=profile,
        )
    else:  # FREE
        cats = categories_for_profile(profile)
        return ResolvedTagConfig(
            mode=TagMode.FREE,
            categories=cats,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=None,
            profile=profile,
        )


def _custom_categories_with_fallback(cats: tuple[str, ...]) -> tuple[str, ...]:
    """Ensure custom/hybrid lists always include uncategorized for low-confidence routing."""
    if UNCATEGORIZED in cats:
        return cats
    return cats + (UNCATEGORIZED,)


def _empty_custom_config(mode: TagMode, profile: SearchProfile) -> ResolvedTagConfig:
    """Missing/empty active tag set — keep requested mode with no categories (GUI blocks start)."""
    return ResolvedTagConfig(
        mode=mode,
        categories=(),
        prompts={},
        priority=(),
        priority_rules_text="",
        user_context="",
        whitelist=frozenset(),
        profile=profile,
    )

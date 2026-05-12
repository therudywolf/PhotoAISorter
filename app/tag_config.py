"""Unified tag configuration: TagMode enum, ResolvedTagConfig, and resolution logic.

This module is the single source of truth for how tag modes map to
categories, descriptions, priority rules, and prompt behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.constants import (
    CATEGORIES,
    CATEGORY_PROMPTS,
    GENERAL_CATEGORIES,
    GENERAL_CATEGORY_WHITELIST,
    CANONICAL_CATEGORY_WHITELIST,
    PRIORITY_RULES_BLOCK,
    TAG_MERGE_PRIORITY,
    UNCATEGORIZED,
)


class TagMode(str, Enum):
    STRICT = "strict"
    GENERAL = "general"
    AUTO = "auto"
    FREE = "free"
    CUSTOM = "custom"


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

    @property
    def is_free(self) -> bool:
        return self.mode in (TagMode.FREE, TagMode.AUTO)

    @property
    def is_strict_whitelist(self) -> bool:
        return self.mode in (TagMode.STRICT, TagMode.GENERAL, TagMode.CUSTOM)


def resolve_tag_config(
    mode: TagMode,
    *,
    tag_store: Any = None,
    user_context_override: str = "",
) -> ResolvedTagConfig:
    """Build a complete tag config from mode + optional custom TagStore.

    For built-in modes (strict/general/auto/free), uses constants.
    For custom mode, uses the active TagSet from the store.
    """
    from app.context_tags import get_active_set, build_custom_categories, build_custom_prompts, build_user_context_from_tags

    if mode == TagMode.CUSTOM:
        if tag_store is None:
            from app.context_tags import load_tag_store
            tag_store = load_tag_store()
        active = get_active_set(tag_store)
        if active and active.tags:
            cats = build_custom_categories(active)
            prompts = build_custom_prompts(active)
            ctx = build_user_context_from_tags(active)
            return ResolvedTagConfig(
                mode=TagMode.CUSTOM,
                categories=cats,
                prompts=prompts,
                priority=(),
                priority_rules_text="",
                user_context=ctx,
                whitelist=frozenset(cats) if cats else None,
            )
        return _fallback_strict_config(user_context_override)

    # For all built-in modes, user_context comes from the active tag set (if any)
    ctx = user_context_override
    if not ctx and tag_store is not None:
        active = get_active_set(tag_store)
        if active:
            ctx = build_user_context_from_tags(active)

    if mode == TagMode.GENERAL:
        return ResolvedTagConfig(
            mode=TagMode.GENERAL,
            categories=GENERAL_CATEGORIES,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=GENERAL_CATEGORY_WHITELIST,
        )
    elif mode == TagMode.AUTO:
        return ResolvedTagConfig(
            mode=TagMode.AUTO,
            categories=GENERAL_CATEGORIES,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=None,
        )
    elif mode == TagMode.FREE:
        return ResolvedTagConfig(
            mode=TagMode.FREE,
            categories=GENERAL_CATEGORIES,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=None,
        )
    else:
        return ResolvedTagConfig(
            mode=TagMode.STRICT,
            categories=CATEGORIES,
            prompts=CATEGORY_PROMPTS,
            priority=TAG_MERGE_PRIORITY,
            priority_rules_text=PRIORITY_RULES_BLOCK,
            user_context=ctx,
            whitelist=CANONICAL_CATEGORY_WHITELIST,
        )


def _fallback_strict_config(user_context: str = "") -> ResolvedTagConfig:
    return ResolvedTagConfig(
        mode=TagMode.STRICT,
        categories=CATEGORIES,
        prompts=CATEGORY_PROMPTS,
        priority=TAG_MERGE_PRIORITY,
        priority_rules_text=PRIORITY_RULES_BLOCK,
        user_context=user_context,
        whitelist=CANONICAL_CATEGORY_WHITELIST,
    )

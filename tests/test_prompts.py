# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Tests for CLIP text prompt generation."""

from __future__ import annotations

from app.fast_classify.prompts import clip_text_prompts_for_tag


def _has_cyrillic(text: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def test_prompts_are_english_only() -> None:
    prompts = clip_text_prompts_for_tag("furry_nsfw_canidae", "anthropomorphic wolf character")
    assert prompts
    assert not any(_has_cyrillic(p) for p in prompts)


def test_prompts_use_readable_label_not_raw_tag() -> None:
    prompts = clip_text_prompts_for_tag("my_dog", "")
    assert all("my_dog" not in p for p in prompts)
    assert any("my dog" in p for p in prompts)


def test_prompts_include_description() -> None:
    prompts = clip_text_prompts_for_tag("iam", "slim young person with light hair")
    assert any("slim young person with light hair" in p for p in prompts)


def test_prompts_are_deduplicated() -> None:
    prompts = clip_text_prompts_for_tag("cat", "")
    assert len(prompts) == len(set(p.lower() for p in prompts))


def test_long_description_is_truncated() -> None:
    prompts = clip_text_prompts_for_tag("x", "word " * 200)
    assert all(len(p) <= 260 for p in prompts)

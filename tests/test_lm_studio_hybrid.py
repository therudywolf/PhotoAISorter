# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Hybrid mode uses expert VLM prompt when fallback runs."""

from __future__ import annotations

from app.lm_studio import CUSTOM_CLASSIFICATION_GUIDANCE, build_system_prompt
from app.tag_config import ResolvedTagConfig, TagMode


def test_build_system_prompt_hybrid_matches_custom_expert_block() -> None:
    cfg = ResolvedTagConfig(
        mode=TagMode.HYBRID,
        categories=("iam", "my_dog"),
        prompts={"iam": "owner", "my_dog": "pet"},
        whitelist=frozenset({"iam", "my_dog"}),
    )
    prompt = build_system_prompt(cfg, structured_output=True)
    assert CUSTOM_CLASSIFICATION_GUIDANCE.splitlines()[0] in prompt
    assert "CLASSES (folder_name: description):" in prompt
    assert "best_folder_name" in prompt

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP text prompt templates (multi-template averaging improves accuracy)."""

from __future__ import annotations


def clip_text_prompts_for_tag(tag: str, description: str) -> list[str]:
    """Return several prompts per tag; embeddings are averaged and re-normalized."""
    label = tag.replace("_", " ")
    desc = (description or label).strip()
    if len(desc) > 220:
        desc = desc[:217] + "..."
    return [
        f"a photo of {label}. {desc}",
        f"a picture in category {tag}. {desc}",
        f"{desc} Category: {tag}.",
        f"фотография категории «{label}». {desc}",
        f"high quality photograph showing {desc}. Folder name: {tag}.",
        f"realistic image, subject: {desc}. Tag {tag}.",
        f"детальная фотография: {desc}. Класс {tag}.",
        f"the main subject matches: {desc}. Label {tag}.",
        f"изображение для сортировки в папку {tag}: {desc}",
    ]

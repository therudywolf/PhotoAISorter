# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP text prompt templates (multi-template averaging improves accuracy)."""

from __future__ import annotations


def clip_text_prompts_for_tag(tag: str, description: str) -> list[str]:
    """Return several English prompts per tag; embeddings are averaged and re-normalized.

    OpenAI CLIP is English-trained, so every template stays English and
    natural-language. Raw tag tokens (e.g. ``furry_nsfw_canidae``) are not real
    words, so prompts use the human-readable label and the user description
    instead of injecting the tag id or "folder name" framing.
    """
    label = tag.replace("_", " ").strip() or tag
    desc = " ".join((description or "").split())
    if len(desc) > 220:
        desc = desc[:217].rstrip() + "..."

    prompts: list[str] = [f"a photo of {label}."]
    if desc:
        prompts.append(f"a photo of {label}, {desc}")
        prompts.append(f"an image showing {desc}")
        prompts.append(f"a picture of {desc}")
        prompts.append(desc)
    else:
        prompts.append(f"a clear, well-lit photo of {label}.")
        prompts.append(f"a picture of {label}.")

    seen: set[str] = set()
    out: list[str] = []
    for p in prompts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out

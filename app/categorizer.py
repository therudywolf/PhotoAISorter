"""Map model output to whitelist category."""

from __future__ import annotations

import re

from app.constants import CATEGORY_WHITELIST, TAG_MERGE_PRIORITY, UNCATEGORIZED

_TAG_PRIORITY_INDEX = {tag: i for i, tag in enumerate(TAG_MERGE_PRIORITY)}
_LEGACY_TAG_ALIASES: dict[str, str] = {
    "cars_and_bmw": "vehicles_and_racing",
    "human_sfw": "human_real_sfw",
    "human_nsfw_solo_male": "human_real_nsfw_male",
    "human_nsfw_solo_female": "human_real_nsfw_female",
    # Group legacy tag is collapsed to female bucket by default for deterministic mapping.
    "human_nsfw_group": "human_real_nsfw_female",
}


def _strip_thinking_sections(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned)
    cleaned = re.sub(r"(?is)<\|channel\>\s*thought\b.*?<channel\|>", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*<think>\s*$", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*</think>\s*$", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _canonical_tag(tag: str) -> str:
    return _LEGACY_TAG_ALIASES.get(tag, tag)


def normalize_tag(raw: str | None) -> str:
    if not raw:
        return UNCATEGORIZED
    raw_clean = _strip_thinking_sections(raw)
    raw_source = raw_clean or raw
    tag = raw_source.strip().lower()
    if tag.startswith("`") and tag.endswith("`"):
        tag = tag[1:-1].strip()
    if tag in CATEGORY_WHITELIST:
        return _canonical_tag(tag)
    lines = [ln.strip().lower() for ln in raw_source.splitlines() if ln.strip()]
    if lines:
        if lines[0] in CATEGORY_WHITELIST:
            return _canonical_tag(lines[0])
        if lines[-1] in CATEGORY_WHITELIST:
            return _canonical_tag(lines[-1])
    # Reasoning-модели: тег может быть где угодно в длинном тексте (ищем самые длинные совпадения первыми)
    blob = raw_source.lower()
    for cat in sorted(CATEGORY_WHITELIST, key=len, reverse=True):
        if cat in blob:
            return _canonical_tag(cat)
    return UNCATEGORIZED


def normalize_tag_free(raw: str | None) -> str:
    """
    Free mode: keep whitelist tags as-is, otherwise return a safe hierarchical tag.
    Format example: nature/forest/sunset
    """
    base = normalize_tag(raw)
    if raw and base != UNCATEGORIZED:
        return base
    if not raw:
        return UNCATEGORIZED
    txt = _strip_thinking_sections(raw).strip().lower()
    if not txt:
        txt = raw.strip().lower()
    txt = txt.replace("`", " ")
    # Keep hierarchy delimiters while normalizing each segment safely.
    txt = re.sub(r"[^a-z0-9_/\-\s]+", " ", txt)
    txt = re.sub(r"/{2,}", "/", txt).strip("/")
    segments: list[str] = []
    for seg in txt.split("/"):
        seg = seg.strip()
        if not seg:
            continue
        seg = re.sub(r"[\s\-]+", "_", seg)
        seg = re.sub(r"_+", "_", seg).strip("_")
        if seg:
            segments.append(seg[:32])
    if not segments:
        return UNCATEGORIZED
    segments = segments[:4]
    out = "/".join(segments)
    if not out:
        return UNCATEGORIZED
    if out in CATEGORY_WHITELIST:
        return _canonical_tag(out)
    return out[:128].strip("/")


def merge_tags_by_priority(tags: list[str]) -> str:
    """При конфликте тегов с разных кадров выбрать один по TAG_MERGE_PRIORITY (меньший индекс = важнее)."""
    if not tags:
        return UNCATEGORIZED
    best = UNCATEGORIZED
    best_rank = len(TAG_MERGE_PRIORITY) + 1
    for raw in tags:
        norm = normalize_tag(raw)
        rank = _TAG_PRIORITY_INDEX.get(norm, len(TAG_MERGE_PRIORITY))
        if rank < best_rank:
            best_rank = rank
            best = norm
    return best

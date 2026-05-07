"""Map model output to whitelist category."""

from __future__ import annotations

import re

from app.constants import CATEGORY_WHITELIST, TAG_MERGE_PRIORITY, UNCATEGORIZED

_TAG_PRIORITY_INDEX = {tag: i for i, tag in enumerate(TAG_MERGE_PRIORITY)}
_WHITELIST_BY_LEN_DESC: tuple[str, ...] = tuple(sorted(CATEGORY_WHITELIST, key=len, reverse=True))
_LEGACY_TAG_ALIASES: dict[str, str] = {
    "cars_and_bmw": "vehicles_and_racing",
    "human_sfw": "human_real_sfw",
    "human_nsfw_solo_male": "human_real_nsfw_male",
    "human_nsfw_solo_female": "human_real_nsfw_female",
    # Group legacy tag is collapsed to female bucket by default for deterministic mapping.
    "human_nsfw_group": "human_real_nsfw_female",
}
_FREE_SANITIZE_RE = re.compile(r"[^a-z0-9_/\-\s]+")
_FREE_SPACES_DASH_RE = re.compile(r"[\s\-]+")
_FREE_UNDERSCORE_RE = re.compile(r"_+")
_AUTO_SPLIT_RE = re.compile(r"[\n,;|]+")


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
    for cat in _WHITELIST_BY_LEN_DESC:
        if cat in blob:
            return _canonical_tag(cat)
    return UNCATEGORIZED


def _safe_hierarchical_tag(raw: str | None) -> str:
    if not raw:
        return UNCATEGORIZED
    txt = _strip_thinking_sections(raw).strip().lower()
    if not txt:
        txt = raw.strip().lower()
    txt = txt.replace("`", " ")
    # Keep hierarchy delimiters while normalizing each segment safely.
    txt = _FREE_SANITIZE_RE.sub(" ", txt)
    txt = re.sub(r"/{2,}", "/", txt).strip("/")
    segments: list[str] = []
    for seg in txt.split("/"):
        seg = seg.strip()
        if not seg:
            continue
        seg = _FREE_SPACES_DASH_RE.sub("_", seg)
        seg = _FREE_UNDERSCORE_RE.sub("_", seg).strip("_")
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


def normalize_tag_free(raw: str | None) -> str:
    """
    Free mode: keep whitelist tags as-is, otherwise return a safe hierarchical tag.
    Format example: nature/forest/sunset
    """
    base = normalize_tag(raw)
    if raw and base != UNCATEGORIZED:
        return base
    return _safe_hierarchical_tag(raw)


def normalize_tag_auto(raw: str | None) -> str:
    """
    Auto mode: accept whitelist tags, otherwise choose the most frequent candidate
    from model output and normalize it as a safe hierarchical tag.
    """
    base = normalize_tag(raw)
    if raw and base != UNCATEGORIZED:
        return base
    if not raw:
        return UNCATEGORIZED
    cleaned = _strip_thinking_sections(raw).lower()
    candidates: list[str] = []
    for chunk in _AUTO_SPLIT_RE.split(cleaned):
        candidate = _safe_hierarchical_tag(chunk)
        if candidate != UNCATEGORIZED:
            candidates.append(candidate)
    if not candidates:
        return _safe_hierarchical_tag(cleaned)
    counts: dict[str, int] = {}
    for tag in candidates:
        counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts, key=lambda k: (-counts[k], len(k), k))[0]


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

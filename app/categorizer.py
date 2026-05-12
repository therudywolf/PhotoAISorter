"""Map model output to whitelist category."""

from __future__ import annotations

import re

from app.constants import (
    FURRY_CATEGORY_WHITELIST,
    GENERAL_CATEGORY_WHITELIST,
    TAG_MERGE_PRIORITY,
    UNCATEGORIZED,
)

_TAG_PRIORITY_INDEX = {tag: i for i, tag in enumerate(TAG_MERGE_PRIORITY)}
_WHITELIST_BY_LEN_FURRY: tuple[str, ...] = tuple(sorted(FURRY_CATEGORY_WHITELIST, key=len, reverse=True))
_WHITELIST_BY_LEN_GENERAL: tuple[str, ...] = tuple(sorted(GENERAL_CATEGORY_WHITELIST, key=len, reverse=True))
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
_TAG_LABEL_RE = re.compile(
    r"(?is)^\s*(?:primary_category|category|tag|label|final(?:\s+tag)?|answer)\s*[:=-]\s*(.+?)\s*$"
)
_PROSE_WORD_RE = re.compile(
    r"\b(?:because|contains|image|looks|photo|picture|shows|there|this|visible|would)\b"
)
_AUTO_SPLIT_RE = re.compile(r"[\n,;|]+")
_AUTO_WEAK_SEGMENTS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "content",
        "file",
        "general",
        "image",
        "images",
        "media",
        "misc",
        "miscellaneous",
        "other",
        "photo",
        "photos",
        "picture",
        "pictures",
        "scene",
        "tag",
        "tags",
        "the",
        "unknown",
    }
)
_AUTO_SEGMENT_ALIASES: dict[str, str] = {
    "airplane": "aviation",
    "airplanes": "aviation",
    "aircraft": "aviation",
    "anime": "illustration",
    "auto": "vehicles",
    "automobile": "vehicles",
    "automobiles": "vehicles",
    "bike": "vehicles",
    "bikes": "vehicles",
    "car": "vehicles",
    "cars": "vehicles",
    "chat": "messaging",
    "cityscape": "city",
    "dog": "animals",
    "dogs": "animals",
    "drawing": "illustration",
    "drawings": "illustration",
    "foodie": "food",
    "funny": "memes",
    "meme": "memes",
    "motorbike": "vehicles",
    "motorbikes": "vehicles",
    "motorcycle": "vehicles",
    "motorcycles": "vehicles",
    "motorsport": "vehicles",
    "motorsports": "vehicles",
    "people": "humans",
    "person": "humans",
    "racing": "vehicles",
    "screen": "screenshots",
    "screen_capture": "screenshots",
    "screencap": "screenshots",
    "screencaps": "screenshots",
    "screenshot": "screenshots",
    "screens": "screenshots",
    "screenshots": "screenshots",
    "selfie": "humans",
    "text": "documents",
    "ui": "screenshots",
    "vehicle": "vehicles",
    "vehicles": "vehicles",
}
_AUTO_ROOTS: frozenset[str] = frozenset(
    {
        "animals",
        "architecture",
        "art",
        "aviation",
        "city",
        "documents",
        "food",
        "humans",
        "illustration",
        "landscape",
        "memes",
        "messaging",
        "nature",
        "objects",
        "screenshots",
        "travel",
        "vehicles",
    }
)
_AUTO_COLLAPSE_TO_PRESET: dict[str, str] = {
    "memes": "memes_and_screenshots",
    "screenshots": "memes_and_screenshots",
}


from app.text_utils import strip_thinking_sections as _strip_thinking_sections


def _canonical_tag(tag: str) -> str:
    return _LEGACY_TAG_ALIASES.get(tag, tag)


def _strip_candidate_label(text: str) -> str:
    candidate = str(text or "").strip().strip("`\"'")
    m = _TAG_LABEL_RE.match(candidate)
    if m:
        candidate = m.group(1).strip().strip("`\"'")
    return candidate


def _looks_like_tag_candidate(text: str) -> bool:
    candidate = _strip_candidate_label(text).lower()
    if not candidate or len(candidate) > 160:
        return False
    if any(ch in candidate for ch in "{}[]:"):
        return False
    has_separator = any(ch in candidate for ch in ("/", "_", "-"))
    if not has_separator and " " in candidate:
        return False
    if _PROSE_WORD_RE.search(candidate) and not has_separator:
        return False
    return bool(re.search(r"[a-z0-9]", candidate))


def _whitelist_by_len_desc(whitelist: frozenset[str]) -> tuple[str, ...]:
    if whitelist is FURRY_CATEGORY_WHITELIST:
        return _WHITELIST_BY_LEN_FURRY
    if whitelist is GENERAL_CATEGORY_WHITELIST:
        return _WHITELIST_BY_LEN_GENERAL
    return tuple(sorted(whitelist, key=len, reverse=True))


def normalize_tag_exact(raw: str | None, *, whitelist: frozenset[str]) -> str:
    """Match only exact tag lines (no substring search). Used by free-tag mode."""
    if not raw:
        return UNCATEGORIZED
    raw_clean = _strip_thinking_sections(raw)
    raw_source = raw_clean or raw
    tag = _strip_candidate_label(raw_source).lower()
    if tag.startswith("`") and tag.endswith("`"):
        tag = tag[1:-1].strip()
    if tag in whitelist:
        return _canonical_tag(tag)
    lines = [_strip_candidate_label(ln).lower() for ln in raw_source.splitlines() if ln.strip()]
    if lines:
        if lines[0] in whitelist:
            return _canonical_tag(lines[0])
        if lines[-1] in whitelist:
            return _canonical_tag(lines[-1])
    return UNCATEGORIZED


def normalize_tag(raw: str | None, *, whitelist: frozenset[str] | None = None) -> str:
    if not raw:
        return UNCATEGORIZED
    wl = FURRY_CATEGORY_WHITELIST if whitelist is None else whitelist
    raw_clean = _strip_thinking_sections(raw)
    raw_source = raw_clean or raw
    tag = _strip_candidate_label(raw_source).lower()
    if tag.startswith("`") and tag.endswith("`"):
        tag = tag[1:-1].strip()
    if tag in wl:
        return _canonical_tag(tag)
    lines = [_strip_candidate_label(ln).lower() for ln in raw_source.splitlines() if ln.strip()]
    if lines:
        if lines[0] in wl:
            return _canonical_tag(lines[0])
        if lines[-1] in wl:
            return _canonical_tag(lines[-1])
    # Reasoning-модели: тег может быть где угодно в длинном тексте (ищем самые длинные совпадения первыми)
    blob = raw_source.lower()
    for cat in _whitelist_by_len_desc(wl):
        if cat in blob:
            return _canonical_tag(cat)
    return UNCATEGORIZED


def _safe_hierarchical_tag(raw: str | None, *, whitelist: frozenset[str] | None = None) -> str:
    if not raw:
        return UNCATEGORIZED
    txt = _strip_candidate_label(_strip_thinking_sections(raw)).strip().lower()
    if not txt:
        txt = _strip_candidate_label(raw).strip().lower()
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
    wl = whitelist if whitelist is not None else GENERAL_CATEGORY_WHITELIST
    if out in wl:
        return _canonical_tag(out)
    return out[:128].strip("/")


def _auto_segment(seg: str, extra_aliases: dict[str, str] | None = None) -> str:
    seg = _FREE_SPACES_DASH_RE.sub("_", seg.strip().lower())
    seg = _FREE_UNDERSCORE_RE.sub("_", seg).strip("_")
    if not seg:
        return ""
    if extra_aliases and seg in extra_aliases:
        aliased = _safe_hierarchical_tag(extra_aliases[seg])
        if aliased != UNCATEGORIZED:
            return aliased.split("/")[0]
    if seg in _AUTO_SEGMENT_ALIASES:
        return _AUTO_SEGMENT_ALIASES[seg]
    if len(seg) > 4 and seg.endswith("ies"):
        singular = seg[:-3] + "y"
        return _AUTO_SEGMENT_ALIASES.get(singular, singular)
    if len(seg) > 3 and seg.endswith("s") and not seg.endswith("ss"):
        singular = seg[:-1]
        return _AUTO_SEGMENT_ALIASES.get(singular, singular)
    return seg


def _optimized_auto_tag(
    raw: str | None,
    extra_aliases: dict[str, str] | None = None,
    *,
    whitelist: frozenset[str] | None = None,
) -> str:
    """
    Auto categories are intentionally conservative: normalize synonyms, keep a
    stable root, and cap depth so large libraries do not explode into folders.
    """
    wl = whitelist if whitelist is not None else GENERAL_CATEGORY_WHITELIST
    safe = _safe_hierarchical_tag(raw, whitelist=wl)
    if safe == UNCATEGORIZED:
        return UNCATEGORIZED
    if safe in wl:
        return _canonical_tag(safe)
    if extra_aliases and safe in extra_aliases:
        aliased = _safe_hierarchical_tag(extra_aliases[safe])
        if aliased != UNCATEGORIZED:
            return _optimized_auto_tag(aliased, None)

    raw_segments = [s for s in safe.split("/") if s]
    segments: list[str] = []
    for raw_seg in raw_segments:
        seg = _auto_segment(raw_seg, extra_aliases)
        if not seg or seg in _AUTO_WEAK_SEGMENTS:
            continue
        if "/" in seg:
            for part in seg.split("/"):
                if part and part not in segments and part not in _AUTO_WEAK_SEGMENTS:
                    segments.append(part)
            continue
        if seg not in segments:
            segments.append(seg)
    if not segments:
        return UNCATEGORIZED

    root_idx = next((i for i, s in enumerate(segments) if s in _AUTO_ROOTS), None)
    if root_idx is None:
        root = segments[0]
        details = segments[1:]
    else:
        root = segments[root_idx]
        details = segments[:root_idx] + segments[root_idx + 1 :]

    if root in _AUTO_COLLAPSE_TO_PRESET and not details:
        return _AUTO_COLLAPSE_TO_PRESET[root]
    if root == "vehicles":
        vehicle_details = [d for d in details if d not in {"vehicles", "racing", "motorsport"}]
        if not vehicle_details:
            return "vehicles_and_racing"
        details = vehicle_details

    compact = [root]
    for detail in details:
        if detail == root or detail in _AUTO_WEAK_SEGMENTS:
            continue
        compact.append(detail)
        if len(compact) >= 2:
            break

    out = "/".join(compact)
    if out in wl:
        return _canonical_tag(out)
    return out[:96].strip("/") or UNCATEGORIZED


def normalize_tag_free(
    raw: str | None,
    *,
    exact_whitelist: frozenset[str] | None = None,
) -> str:
    """
    Free mode: map to whitelist only on exact tag lines (not substring), else hierarchical tag.
    Format example: nature/forest/sunset
    """
    wl = exact_whitelist if exact_whitelist is not None else GENERAL_CATEGORY_WHITELIST
    base = normalize_tag_exact(raw, whitelist=wl)
    if raw and base != UNCATEGORIZED:
        return base
    cleaned = _strip_thinking_sections(raw or "")
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _looks_like_tag_candidate(line):
            candidate = _safe_hierarchical_tag(_strip_candidate_label(line), whitelist=wl)
            if candidate != UNCATEGORIZED:
                return candidate
    if lines:
        return UNCATEGORIZED
    return _safe_hierarchical_tag(raw, whitelist=wl)


def normalize_tag_auto(
    raw: str | None,
    *,
    extra_aliases: dict[str, str] | None = None,
    substring_whitelist: frozenset[str] | None = None,
) -> str:
    """
    Auto mode: accept whitelist tags, otherwise choose the most frequent candidate
    from model output and normalize it as a safe hierarchical tag.
    """
    swl = substring_whitelist if substring_whitelist is not None else GENERAL_CATEGORY_WHITELIST
    exact = normalize_tag_exact(raw, whitelist=swl)
    if raw and exact != UNCATEGORIZED:
        return exact
    if not raw:
        return UNCATEGORIZED
    cleaned = _strip_thinking_sections(raw).lower()
    candidates: list[str] = []
    for chunk in _AUTO_SPLIT_RE.split(cleaned):
        if not _looks_like_tag_candidate(chunk):
            continue
        candidate = _optimized_auto_tag(chunk, extra_aliases, whitelist=swl)
        if candidate != UNCATEGORIZED:
            candidates.append(candidate)
    if not candidates:
        return _optimized_auto_tag(cleaned, extra_aliases, whitelist=swl)
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for idx, tag in enumerate(candidates):
        counts[tag] = counts.get(tag, 0) + 1
        first_seen.setdefault(tag, idx)
    return sorted(counts, key=lambda k: (-counts[k], first_seen[k], len(k), k))[0]


def merge_tags_by_priority(
    tags: list[str],
    *,
    whitelist: frozenset[str] | None = None,
) -> str:
    """При конфликте тегов с разных кадров выбрать один по TAG_MERGE_PRIORITY (меньший индекс = важнее)."""
    if not tags:
        return UNCATEGORIZED
    wl = FURRY_CATEGORY_WHITELIST if whitelist is None else whitelist
    best = UNCATEGORIZED
    best_rank = len(TAG_MERGE_PRIORITY) + 1
    for raw in tags:
        norm = normalize_tag(raw, whitelist=wl)
        rank = _TAG_PRIORITY_INDEX.get(norm, len(TAG_MERGE_PRIORITY))
        if rank < best_rank:
            best_rank = rank
            best = norm
    return best

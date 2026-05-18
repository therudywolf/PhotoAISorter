# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Structured classification result parsing with legacy tag fallback."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from app.categorizer import normalize_tag, normalize_tag_auto, normalize_tag_free
from app.constants import CANONICAL_CATEGORY_WHITELIST, GENERAL_CATEGORY_WHITELIST
from app.constants import UNCATEGORIZED

TagMode = Literal["strict", "general", "auto", "free", "preset", "custom", "hybrid"]

_JSON_FENCE_RE = re.compile(r"(?is)```(?:json)?\s*(\{.*?\})\s*```")


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    candidates: list[str]
    confidence: float
    reason_short: str
    needs_review: bool
    raw_text: str


def _clamp_confidence(value: Any, fallback: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    if f > 1.0 and f <= 100.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidates = [text]
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.insert(0, text[first : last + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _normalize(
    raw: str | None,
    mode: TagMode,
    aliases: dict[str, str] | None = None,
    *,
    whitelist: frozenset[str] | None = None,
) -> str:
    if mode == "auto":
        return normalize_tag_auto(raw, extra_aliases=aliases)
    if mode == "free":
        return normalize_tag_free(raw)
    if whitelist is not None or mode in ("custom", "hybrid"):
        wl = whitelist if whitelist is not None else frozenset()
        if wl:
            return normalize_tag(raw, whitelist=wl)
    if mode == "strict":
        return normalize_tag(raw, whitelist=CANONICAL_CATEGORY_WHITELIST)
    if mode == "general":
        return normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)
    return normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)


def _candidate_values(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    top = obj.get("top_candidates")
    if isinstance(top, list):
        for item in top:
            if isinstance(item, dict):
                name = (
                    item.get("folder_name")
                    or item.get("primary_category")
                    or item.get("category")
                    or item.get("tag")
                )
                if name:
                    out.append(str(name).strip())
            elif isinstance(item, str) and item.strip():
                out.append(item.strip())
    raw_candidates = (
        obj.get("candidates")
        or obj.get("candidate_tags")
        or obj.get("tags")
        or obj.get("labels")
        or []
    )
    if isinstance(raw_candidates, str):
        raw_candidates = re.split(r"[\n,;|]+", raw_candidates)
    if isinstance(raw_candidates, list):
        for x in raw_candidates:
            if isinstance(x, dict):
                name = x.get("folder_name") or x.get("category") or x.get("tag")
                if name:
                    out.append(str(name).strip())
            elif str(x).strip():
                out.append(str(x).strip())
    return out


def parse_classification_result(
    raw: str | None,
    *,
    mode: TagMode,
    aliases: dict[str, str] | None = None,
    whitelist: frozenset[str] | None = None,
    review_confidence_threshold: float = 0.55,
    review_margin_threshold: float = 0.12,
) -> ClassificationResult:
    text = str(raw or "")
    obj = _extract_json_obj(text)

    if obj is None:
        category = _normalize(text, mode, aliases, whitelist=whitelist)
        confidence = 0.75 if category != UNCATEGORIZED else 0.0
        return ClassificationResult(
            category=category,
            candidates=[category] if category != UNCATEGORIZED else [],
            confidence=confidence,
            reason_short="legacy_tag_output",
            needs_review=category == UNCATEGORIZED or confidence < review_confidence_threshold,
            raw_text=text,
        )

    raw_primary = (
        obj.get("best_folder_name")
        or obj.get("primary_category")
        or obj.get("folder_name")
        or obj.get("category")
        or obj.get("tag")
        or obj.get("label")
        or ""
    )
    raw_candidates = _candidate_values(obj)
    category = _normalize(str(raw_primary), mode, aliases, whitelist=whitelist)
    candidates: list[str] = []
    for candidate_raw in [str(raw_primary), *raw_candidates]:
        norm = _normalize(candidate_raw, mode, aliases, whitelist=whitelist)
        if norm != UNCATEGORIZED and norm not in candidates:
            candidates.append(norm)
    if category == UNCATEGORIZED and candidates:
        category = candidates[0]
    confidence = _clamp_confidence(obj.get("confidence"), 0.75 if category != UNCATEGORIZED else 0.0)
    reason = str(obj.get("reasoning") or obj.get("reason_short") or obj.get("reason") or "").strip()[:240]
    explicit_review = obj.get("needs_review")
    needs_review_flag = bool(explicit_review) if isinstance(explicit_review, bool) else False
    margin = _candidate_margin(obj, category)
    needs_review_flag = (
        needs_review_flag
        or category == UNCATEGORIZED
        or confidence < review_confidence_threshold
        or margin < review_margin_threshold
    )
    return ClassificationResult(
        category=category,
        candidates=candidates,
        confidence=confidence,
        reason_short=reason,
        needs_review=needs_review_flag,
        raw_text=text,
    )


def _candidate_margin(obj: dict[str, Any], primary: str) -> float:
    """Margin between top-1 and top-2 candidate probabilities when the model supplies them."""
    top = obj.get("top_candidates")
    if not isinstance(top, list) or len(top) < 2:
        return 1.0
    probs: list[float] = []
    for item in top:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("folder_name")
            or item.get("primary_category")
            or item.get("category")
            or ""
        ).strip()
        if not name:
            continue
        probs.append(_clamp_confidence(item.get("confidence"), 0.5))
    if len(probs) < 2:
        return 1.0
    probs.sort(reverse=True)
    return probs[0] - probs[1]

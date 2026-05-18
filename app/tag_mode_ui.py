# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Tag mode labels, hints, and hybrid readiness (testable without GUI)."""

from __future__ import annotations

from app.fast_classify import clip_available, missing_clip_message
from app.ui_texts import t

TAG_MODE_LABELS: dict[str, str] = {
    "preset_sfw": "SFW",
    "preset_nsfw": "NSFW",
    "preset_furry_sfw": "Furry SFW",
    "preset_furry_nsfw": "Furry NSFW",
    "auto": "Авто",
    "free": "Свободно",
    "custom": "Свой список",
    "hybrid": "Быстрая CLIP",
}

TAG_MODE_VALUES: dict[str, str] = {v: k for k, v in TAG_MODE_LABELS.items()}

PRESET_TAG_MODES: tuple[str, ...] = (
    "preset_sfw",
    "preset_nsfw",
    "preset_furry_sfw",
    "preset_furry_nsfw",
)

FLEXIBLE_TAG_MODES: tuple[str, ...] = ("auto", "free", "custom", "hybrid")

MODES_USING_CUSTOM_LIST: frozenset[str] = frozenset({"custom", "hybrid"})


def label_for_mode(mode: str) -> str:
    return TAG_MODE_LABELS.get(mode, TAG_MODE_LABELS["preset_sfw"])


def mode_from_label(label: str) -> str | None:
    return TAG_MODE_VALUES.get(label)


def build_tag_mode_hint(mode: str, *, clip_ready: bool | None = None) -> str:
    if mode == "free":
        return t("folders.tag_mode.hint_free")
    if mode == "auto":
        return t("folders.tag_mode.hint_auto")
    if mode == "custom":
        return t("folders.tag_mode.hint_custom")
    if mode == "hybrid":
        base = t("folders.tag_mode.hint_hybrid")
        ready = clip_available() if clip_ready is None else clip_ready
        status = t("folders.tag_mode.hybrid_clip_ok" if ready else "folders.tag_mode.hybrid_clip_missing")
        return f"{base}\n{status}"
    return t("folders.tag_mode.hint_preset")


def hybrid_start_blockers(*, categories_count: int, clip_ready: bool | None = None) -> list[str]:
    """Human-readable reasons sort cannot start in hybrid mode."""
    blockers: list[str] = []
    if categories_count <= 0:
        blockers.append(t("folders.tag_mode.hybrid_error_no_tags"))
    ready = clip_available() if clip_ready is None else clip_ready
    if not ready:
        blockers.append(missing_clip_message())
    return blockers


def refs_button_enabled(mode: str) -> bool:
    return mode in MODES_USING_CUSTOM_LIST

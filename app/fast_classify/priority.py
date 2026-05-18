# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Score post-processing: personal tags, NSFW over SFW, specific over generic."""

from __future__ import annotations

from app.constants import UNCATEGORIZED

NSFW_TAGS: frozenset[str] = frozenset(
    {
        "dog_zooporn",
        "explicit_zoo_real_animal",
        "furry_nsfw",
        "furry_nsfw_canidae",
        "furry_nsfw_other",
        "furry_wolf_nsfw",
        "gay_nsfw",
        "human_real_nsfw_female",
        "human_real_nsfw_male",
        "human_ai_gen_nsfw_female",
        "human_ai_gen_nsfw_male",
        "personal_user_nsfw",
        "personal_user_nsfw_alt",
    }
)

SFW_HUMAN_TAGS: frozenset[str] = frozenset(
    {
        "human_real_sfw",
        "human_ai_gen_sfw",
        "personal_user_sfw",
        "personal_user_sfw_alt",
        "gay_sfw",
        "woman",
        "kid",
        "iam",
        "iam_face",
        "iam_body",
    }
)

AI_GENERATED_TAGS: frozenset[str] = frozenset(
    {
        "human_ai_gen_sfw",
        "human_ai_gen_nsfw_female",
        "human_ai_gen_nsfw_male",
    }
)

REAL_PHOTO_HUMAN_TAGS: frozenset[str] = frozenset(
    {
        "human_real_sfw",
        "human_real_nsfw_female",
        "human_real_nsfw_male",
        "personal_user_sfw",
        "personal_user_nsfw",
        "iam",
        "iam_face",
        "iam_body",
    }
)

SPECIFIC_OVER_GENERIC: dict[str, tuple[str, ...]] = {
    "dog": ("my_dog", "my_dog_closeup", "my_dog_fullbody", "my_dog_alt"),
    "cat": ("my_cat", "my_cat_closeup", "my_cat_fullbody"),
    "human_real_sfw": ("iam", "iam_face", "iam_body", "personal_user_sfw", "personal_user_sfw_alt"),
    "human_real_nsfw_female": ("personal_user_nsfw", "personal_user_nsfw_alt", "iam", "iam_body"),
    "human_real_nsfw_male": ("personal_user_nsfw", "personal_user_nsfw_alt", "iam", "iam_body"),
    "woman": ("iam", "iam_face", "personal_user_sfw"),
    "real_animals": ("my_dog", "my_cat", "my_dog_alt"),
    "furry_nsfw": ("furry_nsfw_canidae", "furry_nsfw_other", "furry_wolf_nsfw"),
    "furry_sfw": ("furry_sfw_canidae", "furry_sfw_other", "furry_wolf_sfw"),
    "furry_nsfw_canidae": ("furry_wolf_nsfw",),
    "furry_sfw_canidae": ("furry_wolf_sfw",),
    "memes_and_screenshots": ("meme", "screenshot"),
    "screenshot": ("meme",),
    "meme": ("memes_and_screenshots",),
    "vehicles_and_racing": ("car",),
    "car": ("vehicles_and_racing",),
    "uncategorized": ("uncategorized_alt",),
}

# Higher wins ties at equal score.
TAG_SPECIFICITY: dict[str, int] = {
    "iam": 120,
    "iam_face": 118,
    "iam_body": 117,
    "iam_tattoo": 116,
    "personal_user_nsfw": 115,
    "personal_user_sfw": 115,
    "my_dog": 110,
    "my_dog_closeup": 109,
    "my_dog_fullbody": 109,
    "my_cat": 110,
    "my_cat_closeup": 109,
    "my_cat_fullbody": 109,
    "furry_wolf_nsfw": 85,
    "furry_wolf_sfw": 85,
    "furry_nsfw_canidae": 80,
    "furry_sfw_canidae": 80,
    "kid": 75,
    "dog_zooporn": 70,
    "explicit_zoo_real_animal": 70,
    "uncategorized": 0,
}


def pick_tag(
    scores: dict[str, float],
    *,
    whitelist: frozenset[str],
    apply_preset_rules: bool = True,
) -> tuple[str, float, list[str]]:
    """Choose final tag with priority rules; return (tag, confidence, ranked candidates)."""
    filtered = {k: float(v) for k, v in scores.items() if k in whitelist and float(v) > -1e9}
    if not filtered:
        return UNCATEGORIZED, 0.0, []

    working = dict(filtered)

    if not apply_preset_rules:
        ranked = sorted(
            working.items(),
            key=lambda kv: (-kv[1], -TAG_SPECIFICITY.get(kv[0], 1), kv[0]),
        )
        tag, score = ranked[0]
        candidates = [t for t, _ in ranked[:5] if t != UNCATEGORIZED]
        return tag, score, candidates

    for generic, specifics in SPECIFIC_OVER_GENERIC.items():
        if generic not in working:
            continue
        g_score = working[generic]
        best_spec = max((working.get(s, 0.0) for s in specifics if s in working), default=0.0)
        if best_spec >= g_score * 0.35 and best_spec >= max(working.values()) * 0.25:
            working[generic] = min(working[generic], best_spec * 0.45)

    best_sfw = max((working.get(t, 0.0) for t in SFW_HUMAN_TAGS if t in working), default=0.0)
    best_nsfw = max((working.get(t, 0.0) for t in NSFW_TAGS if t in working), default=0.0)
    if best_nsfw >= 0.08 and best_nsfw >= best_sfw * 0.7:
        for t in SFW_HUMAN_TAGS:
            if t in working and t not in NSFW_TAGS:
                working[t] *= 0.5

    personal = (
        "iam",
        "iam_face",
        "iam_body",
        "iam_tattoo",
        "personal_user_sfw",
        "personal_user_nsfw",
        "personal_user_sfw_alt",
        "personal_user_nsfw_alt",
    )
    best_personal = max((working.get(t, 0.0) for t in personal if t in working), default=0.0)
    if best_personal >= 0.12:
        for t in ("human_real_sfw", "human_real_nsfw_female", "human_real_nsfw_male", "woman"):
            if t in working:
                working[t] *= 0.45

    best_ai = max((working.get(t, 0.0) for t in AI_GENERATED_TAGS if t in working), default=0.0)
    best_real = max((working.get(t, 0.0) for t in REAL_PHOTO_HUMAN_TAGS if t in working), default=0.0)
    if best_real >= 0.12 and best_real >= best_ai * 1.05:
        for t in AI_GENERATED_TAGS:
            if t in working:
                working[t] *= 0.55
    elif best_ai >= 0.12 and best_ai >= best_real * 1.05:
        for t in REAL_PHOTO_HUMAN_TAGS:
            if t in working and t not in personal:
                working[t] *= 0.55

    ranked = sorted(
        working.items(),
        key=lambda kv: (-kv[1], -TAG_SPECIFICITY.get(kv[0], 1), kv[0]),
    )
    tag, score = ranked[0]
    candidates = [t for t, _ in ranked[:5] if t != UNCATEGORIZED]
    return tag, score, candidates

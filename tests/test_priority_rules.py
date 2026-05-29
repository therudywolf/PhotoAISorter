# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Behavioural tests for pick_tag preset reweighting rules.

pick_tag drives every preset/hybrid categorization decision through five
reweighting rules with hand-tuned thresholds. These tests lock in the
observable winner for representative and boundary inputs so the thresholds
cannot drift silently.
"""

from __future__ import annotations

from app.constants import UNCATEGORIZED
from app.fast_classify.priority import pick_tag


def _wl(*tags: str) -> frozenset[str]:
    return frozenset(tags)


# --- Edge cases -------------------------------------------------------------

def test_empty_whitelist_returns_uncategorized() -> None:
    tag, conf, cands = pick_tag({"dog": 0.9}, whitelist=_wl())
    assert tag == UNCATEGORIZED
    assert conf == 0.0
    assert cands == []


def test_scores_outside_whitelist_are_ignored() -> None:
    tag, _conf, _ = pick_tag(
        {"dog": 0.9, "cat": 0.1}, whitelist=_wl("cat")
    )
    assert tag == "cat"


def test_candidates_exclude_uncategorized_and_cap_at_five() -> None:
    scores = {f"tag{i}": 0.5 - i * 0.01 for i in range(8)}
    scores[UNCATEGORIZED] = 0.49
    tag, _conf, cands = pick_tag(
        scores, whitelist=frozenset(scores), apply_preset_rules=False
    )
    assert UNCATEGORIZED not in cands
    assert len(cands) <= 5


# --- Rule 1: specific-over-generic -----------------------------------------

def test_specific_beats_generic_when_strong_enough() -> None:
    # my_dog at 0.3 vs dog at 0.5: specific clears both thresholds, generic demoted.
    tag, _conf, _ = pick_tag(
        {"dog": 0.5, "my_dog": 0.3}, whitelist=_wl("dog", "my_dog")
    )
    assert tag == "my_dog"


def test_generic_kept_when_specific_too_weak() -> None:
    # my_dog at 0.30 vs dog at 1.0: 0.30 < 1.0*0.35 → rule does not fire.
    tag, _conf, _ = pick_tag(
        {"dog": 1.0, "my_dog": 0.30}, whitelist=_wl("dog", "my_dog")
    )
    assert tag == "dog"


# --- Rule 2: NSFW over SFW human -------------------------------------------

def test_nsfw_demotes_sfw_human_when_comparable() -> None:
    # nsfw 0.4 >= 0.08 and >= 0.5*0.7 → sfw halved to 0.25, nsfw wins.
    tag, _conf, _ = pick_tag(
        {"human_real_sfw": 0.5, "human_real_nsfw_female": 0.4},
        whitelist=_wl("human_real_sfw", "human_real_nsfw_female"),
    )
    assert tag == "human_real_nsfw_female"


def test_sfw_kept_when_nsfw_far_below_threshold() -> None:
    # nsfw 0.10 < 0.5*0.7=0.35 → NSFW rule does not fire; sfw stays on top.
    tag, _conf, _ = pick_tag(
        {"human_real_sfw": 0.5, "human_real_nsfw_female": 0.10},
        whitelist=_wl("human_real_sfw", "human_real_nsfw_female"),
    )
    assert tag == "human_real_sfw"


# --- Rule 3: personal identity over generic humans -------------------------

def test_personal_tag_beats_generic_woman() -> None:
    tag, _conf, _ = pick_tag(
        {"woman": 0.5, "iam": 0.25}, whitelist=_wl("woman", "iam")
    )
    assert tag == "iam"


# --- Rule 4: AI-generated vs real-photo separation -------------------------

def test_real_photo_demotes_ai_when_dominant() -> None:
    tag, _conf, _ = pick_tag(
        {"human_real_sfw": 0.5, "human_ai_gen_sfw": 0.4},
        whitelist=_wl("human_real_sfw", "human_ai_gen_sfw"),
    )
    assert tag == "human_real_sfw"


def test_ai_demotes_real_when_dominant() -> None:
    tag, _conf, _ = pick_tag(
        {"human_ai_gen_sfw": 0.5, "human_real_sfw": 0.4},
        whitelist=_wl("human_ai_gen_sfw", "human_real_sfw"),
    )
    assert tag == "human_ai_gen_sfw"


# --- Rule 5: specificity tiebreak ------------------------------------------

def test_specificity_breaks_score_ties() -> None:
    # Equal scores; iam (120) outranks iam_face (118) by TAG_SPECIFICITY.
    tag, _conf, _ = pick_tag(
        {"iam": 0.3, "iam_face": 0.3}, whitelist=_wl("iam", "iam_face")
    )
    assert tag == "iam"


# --- apply_preset_rules=False bypasses reweighting -------------------------

def test_custom_mode_uses_raw_scores() -> None:
    # Without preset rules, the raw top score wins even for generic-vs-specific.
    tag, conf, _ = pick_tag(
        {"dog": 0.5, "my_dog": 0.3},
        whitelist=_wl("dog", "my_dog"),
        apply_preset_rules=False,
    )
    assert tag == "dog"
    assert conf == 0.5

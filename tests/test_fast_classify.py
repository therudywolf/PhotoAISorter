# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Fast CLIP pipeline helpers (no GPU required)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.fast_classify.heuristics import heuristic_tag
from app.fast_classify.priority import pick_tag


def test_pick_tag_prefers_my_dog_over_dog() -> None:
    scores = {"dog": 0.35, "my_dog": 0.28, "uncategorized": 0.1}
    wl = frozenset(scores.keys())
    tag, conf, _cands = pick_tag(scores, whitelist=wl)
    assert tag == "my_dog"
    assert conf > 0.2


def test_pick_tag_nsfw_over_sfw_human() -> None:
    scores = {
        "human_real_sfw": 0.4,
        "human_real_nsfw_male": 0.32,
        "uncategorized": 0.05,
    }
    wl = frozenset(scores.keys())
    tag, _conf, _ = pick_tag(scores, whitelist=wl)
    assert tag == "human_real_nsfw_male"


def test_heuristic_screenshot_filename() -> None:
    im = Image.new("RGB", (1080, 1920), (40, 40, 40))
    path = Path("screen_shot_2024.png")
    hit = heuristic_tag(path, im, whitelist=frozenset({"screenshot", "uncategorized"}))
    assert hit is not None
    assert hit[0] == "screenshot"

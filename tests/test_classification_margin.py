# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

from app.classification_result import parse_classification_result
from app.tag_config import TagMode


def test_parse_marks_ambiguous_top_candidates_for_review() -> None:
    raw = """{
      "best_folder_name": "my_dog",
      "confidence": 0.62,
      "top_candidates": [
        {"folder_name": "my_dog", "confidence": 0.62},
        {"folder_name": "dog", "confidence": 0.58}
      ],
      "reasoning": "both plausible"
    }"""
    result = parse_classification_result(
        raw,
        mode=TagMode.CUSTOM,
        whitelist=frozenset({"my_dog", "dog", "uncategorized"}),
        review_margin_threshold=0.12,
    )
    assert result.category == "my_dog"
    assert result.needs_review is True

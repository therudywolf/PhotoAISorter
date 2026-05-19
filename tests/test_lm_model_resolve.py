# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from app.constants import DEFAULT_MODEL
from app.lm_studio import is_valid_ui_model_id, resolve_ui_model


def test_placeholder_not_valid() -> None:
    assert not is_valid_ui_model_id("— нажмите «Обновить список моделей» —")
    assert not is_valid_ui_model_id("")


def test_combo_beats_stale_default_manual() -> None:
    assert (
        resolve_ui_model(
            manual=DEFAULT_MODEL,
            combo="qwen2.5-vl-7b-instruct",
            profile_model="",
        )
        == "qwen2.5-vl-7b-instruct"
    )


def test_manual_non_default_wins() -> None:
    assert (
        resolve_ui_model(
            manual="my-custom-id",
            combo="other-model",
        )
        == "my-custom-id"
    )


def test_saved_selected_fallback() -> None:
    assert (
        resolve_ui_model(
            manual="",
            combo="— placeholder —",
            saved_selected="saved-vlm",
        )
        == "saved-vlm"
    )

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from app.fast_classify.clip_setup import (
    build_fast_classify_gui_block,
    describe_clip_settings,
    resolve_gui_fast_classify_settings,
)


def test_gui_block_resolves_ultra_profile() -> None:
    block = build_fast_classify_gui_block(
        quality_key="ultra",
        device_key="auto",
        vlm_fallback=True,
    )
    assert block["quality"] == "ultra"
    s = resolve_gui_fast_classify_settings({"fast_classify": block})
    assert s.multi_crop_views >= 7
    assert "ViT" in s.model_name


def test_describe_settings_non_empty() -> None:
    s = resolve_gui_fast_classify_settings(None, quality_key="fast", device_key="cpu")
    text = describe_clip_settings(s)
    assert "ViT-B-32" in text
    assert "384" in text

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from app.fast_classify.config import FastClassifySettings
from app.fast_classify.crops import multi_crop_views
from app.fast_classify.quality import finalize_fast_classify_settings


def test_ultra_profile_enables_max_pooling() -> None:
    s = finalize_fast_classify_settings(
        FastClassifySettings.from_dict({"quality": "ultra", "device": "cpu"}, explicit_keys=frozenset({"quality", "device"}))
    )
    assert s.text_prompt_max_pool is True
    assert s.crop_score_max_pool is True
    assert s.multi_crop_views >= 7


def test_max_profile_cpu_uses_b16_not_l14() -> None:
    s = finalize_fast_classify_settings(
        FastClassifySettings.from_dict({"quality": "max", "device": "cpu"}, explicit_keys=frozenset({"quality", "device"}))
    )
    assert s.model_name == "ViT-B-16"
    assert s.multi_crop is True
    assert s.image_max_side >= 448


def test_fast_profile_disables_multi_crop() -> None:
    s = finalize_fast_classify_settings(
        FastClassifySettings.from_dict({"quality": "fast"}, explicit_keys=frozenset({"quality"}))
    )
    assert s.multi_crop is False
    assert s.model_name == "ViT-B-32"


def test_explicit_model_name_preserved() -> None:
    s = FastClassifySettings.from_dict(
        {"quality": "max", "model_name": "ViT-B-32"},
        explicit_keys=frozenset({"quality", "model_name"}),
    )
    assert s.model_name == "ViT-B-32"


def test_multi_crop_views_count() -> None:
    from PIL import Image

    im = Image.new("RGB", (200, 100), color=(10, 20, 30))
    assert len(multi_crop_views(im, views=5)) == 5
    assert len(multi_crop_views(im, views=9)) == 9
    assert len(multi_crop_views(im, views=1)) == 1

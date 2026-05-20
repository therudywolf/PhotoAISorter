# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Custom tag mode: forest-style presets, expert prompt, JSON parsing."""

from __future__ import annotations

from pathlib import Path
from queue import Queue

from app.category_aliases import load_category_aliases, resolve_storage_category
from app.classification_result import parse_classification_result
from app.constants import MediaScanMode
from app.context_tags import Tag, TagSet, TagStore, load_tag_store
from app.db import Database
from app.lm_studio import CUSTOM_CLASSIFICATION_GUIDANCE, build_system_prompt
from app.tag_config import TagMode, resolve_tag_config
from app.worker import SortWorker

FOREST_SAMPLE_WL = frozenset(
    {
        "iam",
        "iam_face",
        "my_dog",
        "my_dog_closeup",
        "cat",
        "my_cat",
        "uncategorized",
    }
)


def test_parse_custom_expert_json_format() -> None:
    raw = (
        '{"best_folder_name": "iam_face", "confidence": 0.88, '
        '"top_candidates": [{"folder_name": "iam", "confidence": 0.4}], '
        '"reasoning": "portrait selfie of the owner"}'
    )
    result = parse_classification_result(raw, mode="custom", whitelist=FOREST_SAMPLE_WL)
    assert result.category == "iam_face"
    assert result.confidence == 0.88
    assert "iam" in result.candidates
    assert "portrait" in result.reason_short.lower()


def test_build_system_prompt_custom_includes_expert_guidance() -> None:
    from app.tag_config import ResolvedTagConfig

    cfg = ResolvedTagConfig(
        mode=TagMode.CUSTOM,
        categories=("iam", "my_dog"),
        prompts={"iam": "owner photo", "my_dog": "black lab"},
        whitelist=frozenset({"iam", "my_dog"}),
    )
    prompt = build_system_prompt(cfg, structured_output=True)
    assert CUSTOM_CLASSIFICATION_GUIDANCE.splitlines()[0] in prompt
    assert "CLASSES (folder_name: description):" in prompt
    assert "iam: owner photo" in prompt
    assert "best_folder_name" in prompt
    assert "personal_user_*" in prompt


def test_resolve_tag_config_hybrid_same_as_custom() -> None:
    store = TagStore(
        active_set="forest",
        sets=[TagSet(name="forest", tags=[Tag("iam", "owner")])],
    )
    cfg = resolve_tag_config(TagMode.HYBRID, tag_store=store)
    assert cfg.mode == TagMode.HYBRID
    assert "iam" in cfg.categories


def test_resolve_tag_config_custom_skips_duplicate_user_context() -> None:
    store = TagStore(
        active_set="forest",
        sets=[
            TagSet(
                name="forest",
                tags=[Tag("iam", "owner"), Tag("my_dog", "pet")],
            )
        ],
    )
    cfg = resolve_tag_config(TagMode.CUSTOM, tag_store=store)
    assert cfg.user_context == ""
    assert cfg.categories == ("iam", "my_dog", "uncategorized")
    assert cfg.prompts["iam"] == "owner"


def test_forest_preset_loaded_when_present() -> None:
    store = load_tag_store()
    if store.active_set != "forest":
        return
    cfg = resolve_tag_config(TagMode.CUSTOM, tag_store=store)
    assert len(cfg.categories) >= 50
    assert "iam" in cfg.categories
    assert "forest" in cfg.categories
    assert cfg.whitelist is not None
    assert "iam_face" in cfg.whitelist


def test_virtual_tag_aliases_map_to_parent_folder() -> None:
    aliases = load_category_aliases()
    assert resolve_storage_category("iam_face", aliases) == "iam"
    assert resolve_storage_category("my_dog_closeup", aliases) == "my_dog"


def test_sort_worker_custom_virtual_tag_uses_parent_folder(
    tmp_path: Path, monkeypatch: object
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.jpg").write_bytes(b"fake")

    from app.tag_config import ResolvedTagConfig

    wl = frozenset({"iam", "iam_face", "uncategorized"})
    cfg = ResolvedTagConfig(
        mode=TagMode.CUSTOM,
        categories=tuple(wl),
        prompts={"iam": "owner", "iam_face": "owner face", "uncategorized": "fallback"},
        whitelist=wl,
    )
    payload = (
        '{"best_folder_name": "iam_face", "confidence": 0.91, '
        '"top_candidates": [], "reasoning": "face portrait"}'
    )

    monkeypatch.setattr("app.worker.image_to_jpeg_base64_data_uri", lambda _p: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr("app.worker.chat_completion_cfg", lambda *_a, **_k: payload)

    db = Database(tmp_path / "state.sqlite3")
    q: Queue = Queue()
    w = SortWorker(
        db,
        q,
        api_base="http://x",
        model="m",
        workers=1,
        tag_config=cfg,
        category_aliases={"iam_face": "iam"},
    )
    w.run_batch(src, dst, "", media_mode=MediaScanMode.PHOTOS_ONLY)

    assert (dst / "iam" / "a.jpg").exists()
    assert not (dst / "iam_face").exists()
    db.close()

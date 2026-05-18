# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP weight download prefers direct URL over Hugging Face."""

from __future__ import annotations

from typing import Any


def test_download_patch_tries_url_before_hf(monkeypatch: object, tmp_path: object) -> None:
    from pathlib import Path

    import open_clip.factory as factory_mod
    import open_clip.pretrained as pretrained_mod

    from app.fast_classify import weights as w

    calls: list[bool] = []

    def fake_download(
        cfg: dict[str, Any],
        prefer_hf_hub: bool = True,
        cache_dir: str | None = None,
    ) -> str:
        del cfg, cache_dir
        calls.append(prefer_hf_hub)
        if prefer_hf_hub:
            raise OSError("'NoneType' object has no attribute 'write'")
        p = Path(tmp_path) / "ViT-B-32.pt"
        p.write_bytes(b"x" * 64)
        return str(p)

    monkeypatch.setattr(pretrained_mod, "download_pretrained", fake_download)
    w._patch_installed = False
    w.install_clip_download_patch()

    path = factory_mod.download_pretrained({"url": "http://example/x.pt"}, cache_dir=str(tmp_path))
    assert path.endswith("ViT-B-32.pt")
    assert calls == [False]


def test_download_patch_falls_back_to_hf(monkeypatch: object, tmp_path: object) -> None:
    from pathlib import Path

    import open_clip.factory as factory_mod

    from app.fast_classify import weights as w

    calls: list[bool] = []

    def fake_download(
        _cfg: dict[str, Any],
        prefer_hf_hub: bool = True,
        cache_dir: str | None = None,
    ) -> str:
        del cache_dir
        calls.append(prefer_hf_hub)
        if not prefer_hf_hub:
            raise OSError("network down")
        p = Path(tmp_path) / "from_hf.bin"
        p.write_bytes(b"y" * 32)
        return str(p)

    import open_clip.pretrained as pretrained_mod

    monkeypatch.setattr(pretrained_mod, "download_pretrained", fake_download)
    w._patch_installed = False
    w.install_clip_download_patch()

    path = factory_mod.download_pretrained({"url": "http://example/x.pt"}, cache_dir=str(tmp_path))
    assert path.endswith("from_hf.bin")
    assert calls == [False, True]

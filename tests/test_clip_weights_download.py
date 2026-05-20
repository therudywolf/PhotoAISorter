# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP weight download (URL-first, GUI-safe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.fast_classify import weights as w
from app.fast_classify.config import FastClassifySettings


def test_ensure_clip_uses_cached_file(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(w, "clip_cache_dir", lambda: tmp_path)
    big = tmp_path / "ViT-B-32.pt"
    big.write_bytes(b"x" * (w._MIN_OPENAI_VITB32_BYTES + 1))

    cfg = {"url": "https://openaipublic.azureedge.net/x/abc/ViT-B-32.pt", "mean": (0.0,), "std": (1.0,)}
    monkeypatch.setattr(w, "_pretrained_cfg", lambda _s: cfg)

    settings = FastClassifySettings.from_dict({"model_name": "ViT-B-32", "quality": "fast"})
    path, pre = w.ensure_clip_weights_file(settings)
    assert path == big
    assert pre is cfg


def test_ensure_clip_downloads_via_url(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(w, "clip_cache_dir", lambda: tmp_path)
    cfg = {
        "url": "https://openaipublic.azureedge.net/clip/models/abc123/ViT-B-32.pt",
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
    }
    monkeypatch.setattr(w, "_pretrained_cfg", lambda _s: cfg)

    def fake_download(url: str, dest: Path, **kwargs: Any) -> None:
        del url, kwargs
        dest.write_bytes(b"z" * (w._MIN_OPENAI_VITB32_BYTES + 1))

    monkeypatch.setattr(w, "_download_url_quiet", fake_download)

    settings = FastClassifySettings.from_dict({"model_name": "ViT-B-32", "quality": "fast"})
    path, _ = w.ensure_clip_weights_file(settings)
    assert path.name == "ViT-B-32.pt"
    assert path.stat().st_size > w._MIN_OPENAI_VITB32_BYTES


def test_download_patch_tries_url_before_hf(monkeypatch: object, tmp_path: object) -> None:
    import open_clip.factory as factory_mod
    import open_clip.pretrained as pretrained_mod

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

    pretrained_mod._download_pretrained_orig = fake_download
    w._patch_installed = False
    w.install_clip_download_patch()

    path = factory_mod.download_pretrained({"url": "http://example/x.pt"}, cache_dir=str(tmp_path))
    assert path.endswith("ViT-B-32.pt")
    assert calls == [False]

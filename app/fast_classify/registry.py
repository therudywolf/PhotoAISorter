# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Reuse FastClassifier instances across sort runs with identical tag config."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable

from app.fast_classify.config import FastClassifySettings
from app.fast_classify.exemplars import refs_fingerprint
from app.fast_classify.pipeline import FastClassifier
from app.tag_config import ResolvedTagConfig

_lock = threading.Lock()
_cache: dict[str, FastClassifier] = {}


def _config_fingerprint(cfg: ResolvedTagConfig, settings: FastClassifySettings) -> str:
    payload = {
        "categories": list(cfg.categories),
        "prompts": cfg.prompts,
        "settings": settings.to_dict(),
        "refs": refs_fingerprint(extra_tags=cfg.categories),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_classifier(
    cfg: ResolvedTagConfig,
    settings: FastClassifySettings,
    *,
    on_log: Callable[[str], None] | None = None,
    force_new: bool = False,
) -> FastClassifier | None:
    key = _config_fingerprint(cfg, settings)
    with _lock:
        if force_new:
            _cache.pop(key, None)
        if key in _cache:
            return _cache[key]
        clf = FastClassifier.try_create(cfg, settings, on_log=on_log)
        if clf is not None:
            _cache[key] = clf
        return clf


def clear_classifier_cache() -> None:
    with _lock:
        _cache.clear()

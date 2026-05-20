# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Leave-one-out accuracy eval for the hybrid CLIP classifier.

Uses the photos in data/refs/<tag>/ as a labelled set. For every reference
image it rebuilds the classifier with that image excluded from the exemplars,
classifies it, and reports per-tag and overall accuracy.

Run:  .venv/Scripts/python scripts/eval_accuracy.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.fast_classify.exemplars as exemplars_mod
from app.fast_classify.config import load_fast_classify_settings
from app.fast_classify.pipeline import FastClassifier
from app.tag_config import TagMode, resolve_tag_config

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _refs_root() -> Path:
    from app.paths import refs_dir

    return refs_dir()


def _collect_labelled(whitelist: frozenset[str]) -> dict[str, list[Path]]:
    root = _refs_root()
    out: dict[str, list[Path]] = {}
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tag = tag_dir.name
        if tag not in whitelist:
            continue
        imgs = sorted(
            p for p in tag_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXT
        )
        if imgs:
            out[tag] = imgs
    return out


def main() -> int:
    cfg = resolve_tag_config(TagMode.HYBRID)
    if not cfg.categories:
        print("No active tag set found (tmp/app_state/context_tags.json). Abort.")
        return 1
    whitelist = cfg.whitelist or frozenset(cfg.categories)
    settings = load_fast_classify_settings()

    labelled = _collect_labelled(whitelist)
    total_imgs = sum(len(v) for v in labelled.values())
    if total_imgs == 0:
        print("No reference photos in data/refs/<tag>/. Add some and retry.")
        return 1

    print(f"Tags in active set: {len(cfg.categories)}")
    print(f"Labelled folders with photos: {len(labelled)} | images: {total_imgs}")
    print("Running leave-one-out (rebuilds classifier per image)...\n")

    real_list = exemplars_mod.list_exemplar_paths
    per_tag: dict[str, list[bool]] = defaultdict(list)
    confusions: list[tuple[str, str, Path]] = []

    for tag, imgs in labelled.items():
        for img in imgs:
            held = img.resolve()

            def patched(t: str, _held: Path = held) -> list[Path]:
                return [p for p in real_list(t) if p.resolve() != _held]

            exemplars_mod.list_exemplar_paths = patched
            try:
                clf = FastClassifier(cfg, settings)
                res = clf.classify_path(img)
            finally:
                exemplars_mod.list_exemplar_paths = real_list

            ok = res.category == tag
            per_tag[tag].append(ok)
            mark = "OK  " if ok else "MISS"
            review = " [review]" if res.needs_review else ""
            print(f"  [{mark}] {tag:<14} {img.name:<32} -> {res.category}{review}")
            if not ok:
                confusions.append((tag, res.category, img))

    print("\n== Per-tag accuracy ==")
    correct = 0
    for tag in sorted(per_tag):
        results = per_tag[tag]
        c = sum(results)
        correct += c
        print(f"  {tag:<16} {c}/{len(results)}  ({100 * c / len(results):.0f}%)")

    print(f"\nOverall: {correct}/{total_imgs}  ({100 * correct / total_imgs:.1f}%)")
    if confusions:
        print("\n== Misclassifications ==")
        for expected, got, img in confusions:
            print(f"  {expected} -> {got}   ({img.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

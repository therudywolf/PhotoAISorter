# SPDX-License-Identifier: AGPL-3.0-only
"""Self-check CLIP exemplar + VLM accuracy mechanics (no GUI, no LM Studio)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.category_aliases import resolve_storage_category
from app.classification_result import ClassificationResult, parse_classification_result
from app.constants import UNCATEGORIZED
from app.fast_classify.config import FastClassifySettings
from app.fast_classify.quality import finalize_fast_classify_settings
from app.sort_hybrid import _clip_needs_vlm_fallback, _merge_vlm_with_clip
from tests.helpers_clip_stub import stub_classifier


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    mark = "OK" if cond else "FAIL"
    extra = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    return cond


def check_exemplar_mechanics() -> bool:
    print("\n== Exemplar scoring ==")
    all_ok = True
    settings = FastClassifySettings.from_dict(
        {
            "min_exemplar_similarity": 0.32,
            "exemplar_boost": 1.22,
            "exemplar_max_delta": 0.12,
        }
    )
    clf = stub_classifier(
        ["iam", "landscape"],
        settings=settings,
        text_matrix=np.eye(2, dtype=np.float32),
        exemplar_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        exemplar_owner=np.array([0], dtype=np.int32),
    )

    # 1) Weak exemplar match must not boost
    feat = np.array([[0.25, 0.97]], dtype=np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    combined, text = clf._raw_sims_batch(feat)
    all_ok &= _ok(
        "weak exemplar: no delta",
        abs(float(combined[0, 0]) - float(text[0, 0])) < 1e-5,
        f"iam combined={combined[0,0]:.3f} text={text[0,0]:.3f}",
    )
    all_ok &= _ok(
        "weak exemplar: landscape wins",
        float(combined[0, 1]) > float(combined[0, 0]),
        f"landscape={combined[0,1]:.3f}",
    )

    # 2) Strong exemplar: capped delta
    feat2 = np.array([[0.90, 0.20]], dtype=np.float32)
    feat2 /= np.linalg.norm(feat2, axis=1, keepdims=True)
    combined2, text2 = clf._raw_sims_batch(feat2)
    delta = float(combined2[0, 0] - text2[0, 0])
    all_ok &= _ok(
        "strong exemplar: delta capped at 0.12",
        delta <= 0.1201,
        f"delta={delta:.4f}",
    )
    all_ok &= _ok(
        "strong exemplar: combined score <= 1.0",
        float(combined2[0, 0]) <= 1.0 + 1e-6,
        f"combined={combined2[0,0]:.3f}",
    )
    all_ok &= _ok(
        "strong exemplar: not multiplicative",
        float(combined2[0, 0]) < float(text2[0, 0]) * 1.22,
        f"combined={combined2[0,0]:.3f}",
    )

    feat3 = np.array([[0.15, 0.92]], dtype=np.float32)
    feat3 /= np.linalg.norm(feat3, axis=1, keepdims=True)
    combined3, text3 = clf._raw_sims_batch(feat3)
    old_style = float(text3[0, 0]) * 1.38
    all_ok &= _ok(
        "landscape-like photo: landscape beats iam",
        float(combined3[0, 1]) > float(combined3[0, 0]),
        f"landscape={combined3[0,1]:.3f} iam={combined3[0,0]:.3f} old_iam_style={old_style:.3f}",
    )
    return all_ok


def check_clip_review_paths() -> bool:
    print("\n== CLIP thresholds & review ==")
    all_ok = True
    settings = FastClassifySettings.from_dict(
        {
            "confidence_threshold": 0.20,
            "min_margin": 0.08,
            "min_raw_similarity": 0.22,
            "min_raw_margin": 0.04,
            "softmax_temperature": 0.06,
            "top_k_softmax": 8,
        }
    )
    clf = stub_classifier(["dog", "cat", "car"], settings=settings, apply_preset_rules=False)

    low = clf._result_from_sims_row(
        np.array([0.15, 0.12, 0.10], dtype=np.float32),
        text_sims_row=np.array([0.15, 0.12, 0.10], dtype=np.float32),
    )
    all_ok &= _ok("low sim -> uncategorized", low.category == UNCATEGORIZED)
    all_ok &= _ok("low sim -> review", low.needs_review)

    ex_only = clf._result_from_sims_row(
        np.array([0.31, 0.27, 0.12], dtype=np.float32),
        text_sims_row=np.array([0.16, 0.25, 0.11], dtype=np.float32),
    )
    all_ok &= _ok("exemplar-led pick -> review", ex_only.needs_review, f"tag={ex_only.category}")

    confident = clf._result_from_sims_row(
        np.array([0.58, 0.17, 0.14], dtype=np.float32),
        text_sims_row=np.array([0.58, 0.17, 0.14], dtype=np.float32),
    )
    all_ok &= _ok("clear text winner -> no review", not confident.needs_review, confident.category)
    return all_ok


def check_vlm_gating() -> bool:
    print("\n== VLM fallback gating & merge ==")
    all_ok = True
    thr = 0.55

    borderline = ClassificationResult("my_dog", ["my_dog"], 0.58, "clip", True, "")
    low = ClassificationResult("my_dog", ["my_dog"], 0.32, "clip", True, "")
    all_ok &= _ok(
        "borderline review: no VLM",
        not _clip_needs_vlm_fallback(borderline, confidence_threshold=thr),
    )
    all_ok &= _ok(
        "low confidence: VLM",
        _clip_needs_vlm_fallback(low, confidence_threshold=thr),
    )

    clip = ClassificationResult("iam", ["iam"], 0.48, "clip raw=0.40", True, "")
    vlm_weak = ClassificationResult("human_real_sfw", ["human_real_sfw"], 0.55, "legacy", False, "")
    merged = _merge_vlm_with_clip(vlm_weak, clip, confidence_threshold=thr)
    all_ok &= _ok("weak VLM 0.55 does not override CLIP iam", merged.category == "iam")

    clip_u = ClassificationResult(UNCATEGORIZED, [], 0.05, "clip_low_sim raw=0.08", True, "")
    vlm_ok = ClassificationResult("screenshot", ["screenshot"], 0.72, "vlm", False, "")
    merged2 = _merge_vlm_with_clip(vlm_ok, clip_u, confidence_threshold=thr)
    all_ok &= _ok("low-sim CLIP + strong VLM -> screenshot", merged2.category == "screenshot")

    raw = '{"primary_category": "my_dog"}'
    parsed = parse_classification_result(
        raw, mode="hybrid", whitelist=frozenset({"my_dog", "my_cat"})
    )
    all_ok &= _ok("VLM JSON without confidence -> 0.55", abs(parsed.confidence - 0.55) < 1e-6)
    return all_ok


def check_quality_and_aliases() -> bool:
    print("\n== Profiles & storage aliases ==")
    all_ok = True
    ultra = finalize_fast_classify_settings(
        FastClassifySettings.from_dict({"quality": "ultra", "device": "cpu"})
    )
    all_ok &= _ok("ultra exemplar_boost <= 1.35", ultra.exemplar_boost <= 1.35)
    all_ok &= _ok("ultra min_exemplar_similarity >= 0.28", ultra.min_exemplar_similarity >= 0.28)
    all_ok &= _ok("iam_face -> iam folder", resolve_storage_category("iam_face", {}) == "iam")
    return all_ok


def main() -> int:
    print("PhotoAISorter accuracy self-check")
    blocks = [
        check_exemplar_mechanics(),
        check_clip_review_paths(),
        check_vlm_gating(),
        check_quality_and_aliases(),
    ]
    if all(blocks):
        print("\nAll self-checks passed.")
        return 0
    print("\nSome self-checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

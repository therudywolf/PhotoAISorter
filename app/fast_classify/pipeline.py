# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Fast hybrid classifier: heuristics + CLIP text/exemplar + priority rules."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from app.classification_result import ClassificationResult
from app.constants import UNCATEGORIZED
from app.fast_classify.config import FastClassifySettings
from app.fast_classify.crops import multi_crop_views
from app.fast_classify.embedding_cache import EmbeddingCache, model_cache_key
from app.fast_classify.exemplars import ensure_refs_layout, list_exemplar_paths, load_exemplar_images
from app.fast_classify.heuristics import heuristic_tag
from app.fast_classify.model import ClipEmbedder, clip_available, missing_clip_message
from app.fast_classify.priority import pick_tag
from app.fast_classify.prompts import clip_text_prompts_for_tag
from app.fast_classify.scoring import (
    confidence_from_probs,
    needs_review,
    raw_similarity_margin,
    topk_softmax_probs,
)
from app.images import load_image_rgb
from app.tag_config import ResolvedTagConfig, TagMode


class FastClassifier:
    def __init__(
        self,
        cfg: ResolvedTagConfig,
        settings: FastClassifySettings,
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        if not clip_available():
            raise ImportError(missing_clip_message())
        self.cfg = cfg
        self.settings = settings
        self._on_log = on_log
        self._whitelist = cfg.whitelist or frozenset(cfg.categories)
        self._tags = list(cfg.categories)
        self._apply_preset_rules = cfg.mode == TagMode.PRESET
        self._embedder = ClipEmbedder(settings, on_log=on_log)
        ensure_refs_layout(on_log=on_log, extra_tags=cfg.categories)
        (
            self._text_matrix,
            self._prompt_matrix,
            self._prompt_owner,
            self._exemplar_matrix,
            self._exemplar_owner,
        ) = self._build_feature_matrices()
        self._cache: EmbeddingCache | None = None
        if getattr(settings, "cache_embeddings", True):
            try:
                self._cache = EmbeddingCache(
                    model_key=model_cache_key(settings),
                    image_max_side=settings.image_max_side,
                )
            except Exception as e:
                if on_log:
                    on_log(f"CLIP cache отключён: {e}")
                self._cache = None
        if on_log:
            ex_count = int(self._exemplar_matrix.shape[0])
            on_log(
                f"CLIP готов: {len(self._tags)} классов, "
                f"эталонов {ex_count}, устройство {self._embedder.device}"
            )

    @classmethod
    def try_create(
        cls,
        cfg: ResolvedTagConfig,
        settings: FastClassifySettings,
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> FastClassifier | None:
        try:
            return cls(cfg, settings, on_log=on_log)
        except ImportError as e:
            if on_log:
                on_log(str(e))
            return None

    def _build_feature_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for tag in self._tags:
            prompts = clip_text_prompts_for_tag(tag, self.cfg.prompts.get(tag, ""))
            start = len(all_texts)
            all_texts.extend(prompts)
            spans.append((start, len(all_texts)))
        encoded = self._embedder.encode_texts(all_texts).astype(np.float32)
        rows: list[np.ndarray] = []
        fusion = float(self.settings.text_prompt_fusion)
        for start, end in spans:
            chunk = encoded[start:end]
            rows.append(_fuse_prompt_vectors(chunk, fusion_weight=fusion))
        text_matrix = np.stack(rows, axis=0)

        prompt_rows: list[np.ndarray] = []
        prompt_owner: list[int] = []
        for tag_idx, (start, end) in enumerate(spans):
            for row in encoded[start:end]:
                norm = float(np.linalg.norm(row))
                if norm <= 1e-8:
                    continue
                prompt_rows.append((row / norm).astype(np.float32))
                prompt_owner.append(tag_idx)
        if prompt_rows:
            prompt_matrix = np.stack(prompt_rows, axis=0)
            prompt_owner_arr = np.asarray(prompt_owner, dtype=np.int32)
        else:
            prompt_matrix = np.zeros((0, text_matrix.shape[1]), dtype=np.float32)
            prompt_owner_arr = np.zeros((0,), dtype=np.int32)

        loader = lambda p: load_image_rgb(p, max_side=self.settings.image_max_side)
        exemplar_rows: list[np.ndarray] = []
        owner_idx: list[int] = []
        per_tag_count: list[tuple[str, int]] = []
        for i, tag in enumerate(self._tags):
            if not list_exemplar_paths(tag):
                continue
            images = load_exemplar_images(
                tag,
                max_side=self.settings.image_max_side,
                loader=loader,
                on_log=self._on_log,
            )
            if not images:
                continue
            feats = self._embedder.encode_images(images)
            kept = _reject_outlier_vectors(feats)
            if kept.shape[0] == 0:
                continue
            for v in kept:
                norm = float(np.linalg.norm(v))
                if norm <= 1e-8:
                    continue
                exemplar_rows.append((v / norm).astype(np.float32))
                owner_idx.append(i)
            per_tag_count.append((tag, kept.shape[0]))
        if exemplar_rows:
            exemplar_matrix = np.stack(exemplar_rows, axis=0)
            exemplar_owner = np.asarray(owner_idx, dtype=np.int32)
        else:
            exemplar_matrix = np.zeros((0, text_matrix.shape[1]), dtype=np.float32)
            exemplar_owner = np.zeros((0,), dtype=np.int32)
        if self._on_log and per_tag_count:
            head = ", ".join(f"{t}={n}" for t, n in per_tag_count[:8])
            tail = "" if len(per_tag_count) <= 8 else f" (+{len(per_tag_count) - 8} ещё)"
            self._on_log(f"Эталоны загружены: {head}{tail}")
        return text_matrix, prompt_matrix, prompt_owner_arr, exemplar_matrix, exemplar_owner

    def _text_similarities(self, feats: np.ndarray) -> np.ndarray:
        if not getattr(self.settings, "text_prompt_max_pool", False):
            return feats @ self._text_matrix.T
        if self._prompt_matrix.shape[0] == 0:
            return feats @ self._text_matrix.T
        prompt_sims = feats @ self._prompt_matrix.T
        out = np.zeros((feats.shape[0], len(self._tags)), dtype=np.float32)
        for tag_idx in range(len(self._tags)):
            mask = self._prompt_owner == tag_idx
            if mask.any():
                out[:, tag_idx] = prompt_sims[:, mask].max(axis=1)
        return out

    def _raw_sims_batch(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity (B, num_tags).

        Exemplars add a small capped delta on top of text similarity (not multiplicative
        replacement), so tags with refs do not dominate tags without refs.
        """
        text_sims = self._text_similarities(feats)
        sims = text_sims.copy()
        if self._exemplar_matrix.shape[0] == 0:
            return sims, text_sims
        min_sim = float(getattr(self.settings, "min_exemplar_similarity", 0.32))
        max_delta = float(getattr(self.settings, "exemplar_max_delta", 0.12))
        scale = max(0.0, float(self.settings.exemplar_boost) - 1.0)
        ex_sims = feats @ self._exemplar_matrix.T
        for tag_idx in range(len(self._tags)):
            mask = self._exemplar_owner == tag_idx
            if not mask.any():
                continue
            best_raw = ex_sims[:, mask].max(axis=1)
            delta = np.maximum(0.0, best_raw - min_sim) * scale
            delta = np.minimum(delta, max_delta)
            sims[:, tag_idx] = np.minimum(1.0, sims[:, tag_idx] + delta)
        return sims, text_sims


    def _result_from_sims_row(
        self,
        sims_row: np.ndarray,
        *,
        text_sims_row: np.ndarray | None = None,
        embedding_failed: bool = False,
    ) -> ClassificationResult:
        if embedding_failed:
            return ClassificationResult(
                UNCATEGORIZED, [], 0.0, "clip_embed_failed", True, ""
            )
        top_raw, raw_margin = raw_similarity_margin(sims_row)
        min_raw = float(getattr(self.settings, "min_raw_similarity", 0.18))
        min_raw_m = float(getattr(self.settings, "min_raw_margin", 0.03))
        top_k = int(getattr(self.settings, "top_k_softmax", 10))

        if top_raw < min_raw:
            return ClassificationResult(
                UNCATEGORIZED,
                [],
                max(0.0, top_raw),
                f"clip_low_sim raw={top_raw:.2f}",
                True,
                "",
            )

        probs = topk_softmax_probs(
            sims_row,
            temperature=self.settings.softmax_temperature,
            top_k=top_k,
        )
        scores = {self._tags[i]: float(probs[i]) for i in range(len(self._tags))}
        tag, _conf, candidates = pick_tag(
            scores,
            whitelist=self._whitelist,
            apply_preset_rules=self._apply_preset_rules,
        )
        top_prob, prob_margin = confidence_from_probs(probs)
        review = needs_review(
            top_prob,
            prob_margin,
            min_prob=self.settings.confidence_threshold,
            min_margin=self.settings.min_margin,
        )
        if raw_margin < min_raw_m:
            review = True
        if top_prob < 0.16 and prob_margin < self.settings.min_margin:
            review = True
        if tag == UNCATEGORIZED:
            top_prob = max(scores.values()) if scores else 0.0
            review = True
        if text_sims_row is not None and tag in self._tags:
            tag_idx = self._tags.index(tag)
            ex_delta = float(sims_row[tag_idx] - text_sims_row[tag_idx])
            text_score = float(text_sims_row[tag_idx])
            if ex_delta >= 0.045 and text_score < 0.24:
                review = True
        return ClassificationResult(
            category=tag,
            candidates=candidates,
            confidence=min(1.0, max(0.0, top_prob)),
            reason_short=f"clip raw={top_raw:.2f} margin={raw_margin:.2f}",
            needs_review=review,
            raw_text="",
        )

    def classify_image(self, path: Path, im: Image.Image) -> ClassificationResult:
        h = heuristic_tag(path, im, whitelist=self._whitelist)
        if h is not None:
            tag, conf, reason = h
            auto_ok = conf >= 0.92
            return ClassificationResult(
                category=tag,
                candidates=[tag],
                confidence=conf,
                reason_short=f"heuristic:{reason}",
                needs_review=not auto_ok
                or needs_review(
                    conf,
                    1.0 if conf >= 0.85 else 0.0,
                    min_prob=self.settings.confidence_threshold,
                    min_margin=self.settings.min_margin,
                ),
                raw_text="",
            )
        sims_row, text_row, feat = self._encode_image_sims_and_feature(im)
        failed = float(np.linalg.norm(feat)) <= 1e-8
        return self._result_from_sims_row(
            sims_row, text_sims_row=text_row, embedding_failed=failed
        )

    def classify_path(self, path: Path) -> ClassificationResult:
        im = load_image_rgb(path, max_side=self.settings.image_max_side)
        return self.classify_image(path, im)

    def _load_image(self, path: Path) -> tuple[Path, Image.Image | None, str | None]:
        try:
            return path, load_image_rgb(path, max_side=self.settings.image_max_side), None
        except OSError as e:
            return path, None, str(e)

    def classify_video_frames(self, path: Path, frames: list[Image.Image]) -> ClassificationResult:
        if not frames:
            return ClassificationResult(UNCATEGORIZED, [], 0.0, "no_frames", True, "")
        if (
            self.settings.multi_crop
            and int(self.settings.multi_crop_views) > 1
            and getattr(self.settings, "crop_score_max_pool", False)
        ):
            frame_sims: list[np.ndarray] = []
            any_feat = False
            frame_text: list[np.ndarray] = []
            for frame in frames:
                sims_row, text_row, feat = self._encode_image_sims_and_feature(frame)
                if float(np.linalg.norm(feat)) > 1e-8:
                    any_feat = True
                frame_sims.append(sims_row)
                frame_text.append(text_row)
            if not frame_sims or not any_feat:
                return ClassificationResult(UNCATEGORIZED, [], 0.0, "no_feats", True, "")
            sims_row = np.max(np.stack(frame_sims, axis=0), axis=0)
            text_row = np.max(np.stack(frame_text, axis=0), axis=0)
            return self._result_from_sims_row(
                sims_row, text_sims_row=text_row, embedding_failed=not any_feat
            )
        feats = self._encode_image_features_batch(frames)
        if feats.shape[0] == 0:
            return ClassificationResult(UNCATEGORIZED, [], 0.0, "no_feats", True, "")
        sims_per_frame, text_per_frame = self._raw_sims_batch(feats)
        sims_row = sims_per_frame.max(axis=0)
        text_row = text_per_frame.max(axis=0)
        failed = bool(_embedding_rows_failed(feats).any())
        return self._result_from_sims_row(
            sims_row, text_sims_row=text_row, embedding_failed=failed
        )

    def classify_batch(
        self,
        paths: list[Path],
        *,
        digests: list[str] | None = None,
    ) -> list[ClassificationResult]:
        if not paths:
            return []
        results: list[ClassificationResult | None] = [None] * len(paths)
        clip_indices: list[int] = []
        clip_images: list[Image.Image] = []
        clip_digests: list[str | None] = []
        cached_feats: dict[int, np.ndarray] = {}
        cached_sims: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        workers = min(max(1, self.settings.prefetch_workers), max(1, len(paths)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(self._load_image, paths))

        digest_lookup: dict[str, list[int]] = {}
        if digests is not None and self._cache is not None and len(digests) == len(paths):
            for i, d in enumerate(digests):
                if d:
                    digest_lookup.setdefault(d, []).append(i)
            hits = self._cache.get_many(list(digest_lookup.keys()))
        else:
            hits = {}

        for i, (path, im, err) in enumerate(loaded):
            if err is not None:
                results[i] = ClassificationResult(UNCATEGORIZED, [], 0.0, "load_error", True, "")
                continue
            assert im is not None
            h = heuristic_tag(path, im, whitelist=self._whitelist)
            if h is not None:
                tag, conf, reason = h
                auto_ok = conf >= 0.92
                results[i] = ClassificationResult(
                    category=tag,
                    candidates=[tag],
                    confidence=conf,
                    reason_short=f"heuristic:{reason}",
                    needs_review=not auto_ok
                    or needs_review(
                        conf,
                        1.0 if conf >= 0.85 else 0.0,
                        min_prob=self.settings.confidence_threshold,
                        min_margin=self.settings.min_margin,
                    ),
                    raw_text="",
                )
                continue
            d = digests[i] if digests is not None and i < len(digests) else None
            if d and d in hits:
                row = hits[d]
                if float(np.linalg.norm(row)) > 1e-8:
                    cached_feats[i] = row
                    continue
            clip_indices.append(i)
            clip_images.append(im)
            clip_digests.append(d)

        new_feats_to_store: list[tuple[str, np.ndarray]] = []
        if clip_images:
            use_crop_pool = bool(
                self.settings.multi_crop
                and int(self.settings.multi_crop_views) > 1
                and getattr(self.settings, "crop_score_max_pool", False)
            )
            if use_crop_pool:
                for j, idx in enumerate(clip_indices):
                    sims_row, text_row, feat = self._encode_image_sims_and_feature(
                        clip_images[j]
                    )
                    cached_sims[idx] = (sims_row, text_row)
                    cached_feats[idx] = feat
                    d = clip_digests[j]
                    if d and float(np.linalg.norm(feat)) > 1e-8:
                        new_feats_to_store.append((d, feat))
            else:
                feats = self._encode_image_features_batch(clip_images)
                embed_failed = _embedding_rows_failed(feats)
                for j, idx in enumerate(clip_indices):
                    d = clip_digests[j]
                    if d and not embed_failed[j]:
                        new_feats_to_store.append((d, feats[j]))
                    cached_feats[idx] = feats[j]

        if cached_sims:
            for idx, (sims_row, text_row) in cached_sims.items():
                results[idx] = self._result_from_sims_row(
                    sims_row, text_sims_row=text_row, embedding_failed=False
                )

        if cached_feats:
            order = sorted(cached_feats.keys())
            stacked = np.stack([cached_feats[i] for i in order], axis=0)
            sims, text_sims = self._raw_sims_batch(stacked)
            failed_rows = _embedding_rows_failed(stacked)
            for k, idx in enumerate(order):
                if results[idx] is not None:
                    continue
                results[idx] = self._result_from_sims_row(
                    sims[k],
                    text_sims_row=text_sims[k],
                    embedding_failed=failed_rows[k],
                )

        if new_feats_to_store and self._cache is not None:
            try:
                self._cache.put_many(new_feats_to_store)
            except Exception as e:
                if self._on_log:
                    self._on_log(f"CLIP cache write failed: {e}")

        return [
            r
            if r is not None
            else ClassificationResult(UNCATEGORIZED, [], 0.0, "missing", True, "")
            for r in results
        ]


    def _encode_image_sims_and_feature(
        self, im: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = self._text_matrix.shape[1]
        n_tags = len(self._tags)
        if not self.settings.multi_crop or int(self.settings.multi_crop_views) <= 1:
            feats = self._embedder.encode_images([im], micro_batch=1)
            if feats.shape[0] == 0:
                z = np.zeros((n_tags,), dtype=np.float32)
                return z, z, np.zeros((dim,), dtype=np.float32)
            sims, text_sims = self._raw_sims_batch(feats)
            return sims[0], text_sims[0], feats[0]
        crops = multi_crop_views(im, views=int(self.settings.multi_crop_views))
        crop_feats = self._embedder.encode_images(crops, micro_batch=self.settings.batch_size)
        if crop_feats.shape[0] == 0:
            z = np.zeros((n_tags,), dtype=np.float32)
            return z, z, np.zeros((dim,), dtype=np.float32)
        sims, text_sims = self._raw_sims_batch(crop_feats.astype(np.float32))
        if getattr(self.settings, "crop_score_max_pool", False) and sims.shape[0] > 1:
            sims_row = sims.max(axis=0)
            text_row = text_sims.max(axis=0)
        else:
            sims_row = sims[0]
            text_row = text_sims[0]
        feat = _fuse_crop_features(crop_feats, sims)
        return sims_row, text_row, feat

    def _encode_image_features_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._text_matrix.shape[1]), dtype=np.float32)
        rows: list[np.ndarray] = []
        for im in images:
            _, _, feat = self._encode_image_sims_and_feature(im)
            rows.append(feat)
        return np.stack(rows, axis=0)


def _fuse_crop_features(crop_feats: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Fuse crop embeddings for cache (top crops by max class score)."""
    if crop_feats.shape[0] == 0:
        return np.zeros((sims.shape[1] if sims.ndim == 2 else 512,), dtype=np.float32)
    if crop_feats.shape[0] == 1:
        return crop_feats[0]
    scores = sims.max(axis=1) if sims.ndim == 2 else np.asarray([float(sims.max())])
    top_k = min(2, crop_feats.shape[0])
    idx = np.argpartition(scores, -top_k)[-top_k:]
    fused = crop_feats[idx].mean(axis=0)
    norm = float(np.linalg.norm(fused))
    return (fused / norm).astype(np.float32) if norm > 1e-8 else crop_feats[int(np.argmax(scores))]


def _fuse_prompt_vectors(chunk: np.ndarray, *, fusion_weight: float) -> np.ndarray:
    """Blend mean and strongest prompt embedding (sharper class prototypes)."""
    chunk = np.atleast_2d(np.asarray(chunk, dtype=np.float32))
    if chunk.shape[0] == 0:
        return np.zeros((512,), dtype=np.float32)
    if chunk.shape[0] == 1:
        v = chunk[0]
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
    norms = np.linalg.norm(chunk, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    unit = chunk / norms
    mean = unit.mean(axis=0)
    mn = float(np.linalg.norm(mean))
    if mn <= 1e-8:
        mean = unit[0]
    else:
        mean = mean / mn
    scores = unit @ mean
    best = unit[int(np.argmax(scores))]
    w = max(0.0, min(1.0, float(fusion_weight)))
    fused = (1.0 - w) * mean + w * best
    fn = float(np.linalg.norm(fused))
    return (fused / fn).astype(np.float32) if fn > 1e-8 else mean.astype(np.float32)


def _embedding_rows_failed(feats: np.ndarray) -> np.ndarray:
    """True per row when the embedding vector is missing (zero norm after encode)."""
    if feats.size == 0:
        return np.zeros((0,), dtype=bool)
    norms = np.linalg.norm(feats, axis=1)
    return norms <= 1e-8


def _reject_outlier_vectors(feats: np.ndarray, *, sigma: float = 2.0) -> np.ndarray:
    """Drop exemplar vectors whose similarity to the centroid is far below median (MAD-based)."""
    if feats.shape[0] <= 3:
        return feats
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    nfeats = feats / norms
    centroid = nfeats.mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    if c_norm <= 1e-8:
        return feats
    centroid = centroid / c_norm
    sims = nfeats @ centroid
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med))) or 1e-3
    keep = sims >= (med - sigma * 1.4826 * mad)
    return feats[keep] if keep.any() else feats

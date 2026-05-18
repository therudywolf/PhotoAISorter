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
from app.fast_classify.embedding_cache import EmbeddingCache, model_cache_key
from app.fast_classify.exemplars import ensure_refs_layout, list_exemplar_paths, load_exemplar_images
from app.fast_classify.heuristics import heuristic_tag
from app.fast_classify.model import ClipEmbedder, clip_available, missing_clip_message
from app.fast_classify.priority import pick_tag
from app.fast_classify.prompts import clip_text_prompts_for_tag
from app.fast_classify.scoring import confidence_from_probs, needs_review, softmax_probs
from app.images import load_image_rgb
from app.tag_config import ResolvedTagConfig


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
        self._embedder = ClipEmbedder(settings, on_log=on_log)
        ensure_refs_layout(on_log=on_log)
        self._text_matrix, self._exemplar_matrix, self._exemplar_tag_index = self._build_feature_matrices()
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
            ex_count = int(np.sum(self._exemplar_tag_index >= 0))
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for tag in self._tags:
            prompts = clip_text_prompts_for_tag(tag, self.cfg.prompts.get(tag, ""))
            start = len(all_texts)
            all_texts.extend(prompts)
            spans.append((start, len(all_texts)))
        encoded = self._embedder.encode_texts(all_texts).astype(np.float32)
        rows: list[np.ndarray] = []
        for start, end in spans:
            chunk = encoded[start:end]
            mean = chunk.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            rows.append((mean / norm).astype(np.float32) if norm > 1e-8 else chunk[0])
        text_matrix = np.stack(rows, axis=0)
        exemplar_matrix = np.zeros_like(text_matrix)
        exemplar_index = np.full(len(self._tags), -1, dtype=np.int32)
        loader = lambda p: load_image_rgb(p, max_side=self.settings.image_max_side)
        ex_row = 0
        for i, tag in enumerate(self._tags):
            if not list_exemplar_paths(tag):
                continue
            images = load_exemplar_images(tag, max_side=self.settings.image_max_side, loader=loader)
            if not images:
                continue
            feats = self._embedder.encode_images(images)
            mean = feats.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 1e-8:
                continue
            if ex_row >= len(self._tags):
                break
            exemplar_matrix[ex_row] = (mean / norm).astype(np.float32)
            exemplar_index[i] = ex_row
            ex_row += 1
        if ex_row < len(self._tags):
            exemplar_matrix = exemplar_matrix[:ex_row]
        return text_matrix, exemplar_matrix, exemplar_index

    def _raw_sims_batch(self, feats: np.ndarray) -> np.ndarray:
        """Cosine similarity (B, num_tags) with exemplar boost applied per tag."""
        text_sims = feats @ self._text_matrix.T
        sims = text_sims.copy()
        boost = self.settings.exemplar_boost
        if self._exemplar_matrix.shape[0] > 0:
            ex_sims = feats @ self._exemplar_matrix.T
            for col, tag in enumerate(self._tags):
                ex_i = int(self._exemplar_tag_index[col])
                if ex_i >= 0:
                    sims[:, col] = np.maximum(sims[:, col], ex_sims[:, ex_i] * boost)
        return sims

    def _result_from_sims_row(self, sims_row: np.ndarray) -> ClassificationResult:
        probs = softmax_probs(
            sims_row.reshape(1, -1),
            temperature=self.settings.softmax_temperature,
        )[0]
        scores = {self._tags[i]: float(probs[i]) for i in range(len(self._tags))}
        tag, _conf, candidates = pick_tag(scores, whitelist=self._whitelist)
        top_prob, margin = confidence_from_probs(probs)
        review = needs_review(
            top_prob,
            margin,
            min_prob=self.settings.confidence_threshold,
            min_margin=self.settings.min_margin,
        )
        if tag == UNCATEGORIZED:
            top_prob = max(scores.values()) if scores else 0.0
            review = True
        return ClassificationResult(
            category=tag,
            candidates=candidates,
            confidence=min(1.0, max(0.0, top_prob)),
            reason_short=f"clip_prob margin={margin:.2f}",
            needs_review=review,
            raw_text="",
        )

    def classify_image(self, path: Path, im: Image.Image) -> ClassificationResult:
        h = heuristic_tag(path, im, whitelist=self._whitelist)
        if h is not None:
            tag, conf, reason = h
            return ClassificationResult(
                category=tag,
                candidates=[tag],
                confidence=conf,
                reason_short=f"heuristic:{reason}",
                needs_review=needs_review(
                    conf,
                    1.0 if conf >= 0.85 else 0.0,
                    min_prob=self.settings.confidence_threshold,
                    min_margin=self.settings.min_margin,
                ),
                raw_text="",
            )
        feat = self._embedder.encode_images([im], micro_batch=1)
        sims = self._raw_sims_batch(feat)
        return self._result_from_sims_row(sims[0])

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
        feats = self._embedder.encode_images(frames, micro_batch=self.settings.batch_size)
        mean = feats.mean(axis=0, keepdims=True)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean = mean / norm
        sims = self._raw_sims_batch(mean.astype(np.float32))
        return self._result_from_sims_row(sims[0])

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
                results[i] = ClassificationResult(
                    category=tag,
                    candidates=[tag],
                    confidence=conf,
                    reason_short=f"heuristic:{reason}",
                    needs_review=needs_review(
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
                cached_feats[i] = hits[d]
                continue
            clip_indices.append(i)
            clip_images.append(im)
            clip_digests.append(d)

        new_feats_to_store: list[tuple[str, np.ndarray]] = []
        if clip_images:
            feats = self._embedder.encode_images(
                clip_images, micro_batch=self.settings.batch_size
            )
            for j, idx in enumerate(clip_indices):
                d = clip_digests[j]
                if d:
                    new_feats_to_store.append((d, feats[j]))
                cached_feats[idx] = feats[j]

        if cached_feats:
            order = sorted(cached_feats.keys())
            stacked = np.stack([cached_feats[i] for i in order], axis=0)
            sims = self._raw_sims_batch(stacked)
            for k, idx in enumerate(order):
                results[idx] = self._result_from_sims_row(sims[k])

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

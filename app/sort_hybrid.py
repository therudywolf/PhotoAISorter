# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Batched CLIP sort path with optional VLM fallback (hybrid tag mode)."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.category_aliases import resolve_storage_category
from app.classification_result import ClassificationResult
from app.constants import GIF_EXTENSION, PIPELINE_VERSION, UNCATEGORIZED, VIDEO_EXTENSIONS
from app.fast_classify import clip_available, load_fast_classify_settings, missing_clip_message
from app.fast_classify.registry import get_classifier
from app.file_hash_cache import get_file_hash_cache
from app.images import image_to_jpeg_base64_data_uri, video_contact_sheet_data_uri
from app.lm_studio import chat_completion_cfg
from app.video_frames import extract_frames_reduced, is_animated_gif
from app.worker import has_disk_space_for_copy, unique_dest_path

if TYPE_CHECKING:
    from app.worker import SortWorker


def _merge_vlm_with_clip(
    vlm: ClassificationResult,
    clip: ClassificationResult,
    *,
    confidence_threshold: float,
) -> ClassificationResult:
    """Prefer CLIP when VLM is weaker or only CLIP failed review margin."""
    if clip.category == UNCATEGORIZED:
        return vlm
    if vlm.category == UNCATEGORIZED:
        return clip
    if clip.confidence >= confidence_threshold and not clip.needs_review:
        if vlm.confidence < clip.confidence + 0.04:
            return clip
    if clip.needs_review and vlm.confidence > clip.confidence + 0.06:
        return vlm
    if vlm.confidence > clip.confidence + 0.08:
        return vlm
    return clip if clip.confidence >= vlm.confidence else vlm


def _vlm_via_uri(
    worker: SortWorker,
    data_uri: str,
    *,
    video_hint: str,
    _prompt_for_request: Callable[[], str],
    _result_from_raw: Callable[[str], ClassificationResult],
    _timed_api_call: Callable[[Callable[[], str]], str],
    _get_api_base: Callable[[], str],
    _timeout: Callable[[], tuple[float, float] | None],
    on_retry: Callable[[str], None],
) -> ClassificationResult:
    extra = _prompt_for_request()
    if video_hint:
        extra = f"{extra}\n\n{video_hint}".strip() if extra else video_hint
    raw = _timed_api_call(
        lambda: chat_completion_cfg(
            data_uri,
            worker.tag_config,
            api_base=_get_api_base(),
            model=worker.model,
            api_key=worker.api_key,
            timeout=_timeout(),
            on_retry=on_retry,
            prompt_extra=extra,
            structured_output=worker.structured_output,
            temperature=worker.temperature,
            max_tokens=worker.max_tokens,
        )
    )
    return _result_from_raw(raw)


def run_hybrid_sort(
    worker: SortWorker,
    source_dir: Path,
    dest_dir: Path,
    files: list[Path],
    *,
    session_key: str,
    metrics: dict[str, int | float],
    metrics_lock: Any,
    manifest: Any,
    complete_task: Callable[[float], None],
    save_session: Callable[..., None],
    _prompt_for_request: Callable[[], str],
    _result_from_raw: Callable[[str], ClassificationResult],
    _timed_api_call: Callable[[Callable[[], str]], str],
    _get_api_base: Callable[[], str],
    _timeout: Callable[[], tuple[float, float] | None],
    _record_review: Callable[..., None],
    _register_api_error: Callable[[Exception], None],
) -> str:
    settings = load_fast_classify_settings()
    if not clip_available():
        worker._emit({"type": "log", "text": missing_clip_message()})
        raise RuntimeError(missing_clip_message())

    fast = get_classifier(
        worker.tag_config,
        settings,
        on_log=lambda m: worker._emit({"type": "log", "text": m}),
    )
    if fast is None:
        raise RuntimeError(missing_clip_message())

    metrics["fast_classify"] = 0
    metrics["vlm_fallback"] = 0

    vlm_enabled = bool(settings.vlm_fallback)
    if vlm_enabled:
        if not _probe_lm_studio(_get_api_base(), timeout=2.5):
            worker._emit(
                {
                    "type": "log",
                    "text": (
                        "LM Studio недоступен — VLM-фоллбэк отключён на эту сессию. "
                        "Сомнительные снимки уйдут в review."
                    ),
                }
            )
            vlm_enabled = False

    crop_note = (
        f", multi-crop ×{settings.multi_crop_views}"
        if settings.multi_crop
        else ""
    )
    worker._emit(
        {
            "type": "log",
            "text": (
                f"Быстрая сортировка: качество={settings.quality}, "
                f"{settings.model_name} @ {settings.image_max_side}px, "
                f"батч {settings.batch_size}{crop_note}, "
                f"порог {settings.confidence_threshold:.2f}, "
                f"VLM fallback={'да' if vlm_enabled else 'нет'}."
            ),
        }
    )

    pending: list[dict[str, Any]] = []
    total_files = len(files)
    worker._emit(
        {
            "type": "log",
            "text": f"CLIP: подготовка {total_files} файлов (хеш и кеш, параллельно)…",
        }
    )

    def _prepare_one(path: Path) -> dict[str, Any] | None:
        if worker._stop.is_set():
            return None
        path_norm = str(path.resolve())
        try:
            st = path.stat()
            mtime_ns = int(st.st_mtime_ns)
            size_bytes = int(st.st_size)
        except OSError as e:
            worker._emit({"type": "log", "text": f"stat error {path}: {e}"})
            return None

        if worker.resume_session and worker.db.sort_session_item_status(
            session_key, path_norm, mtime_ns=mtime_ns, size_bytes=size_bytes
        ) == "done":
            with metrics_lock:
                metrics["cache_skip"] = int(metrics["cache_skip"]) + 1
            complete_task(0.0)
            return None

        try:
            digest = get_file_hash_cache().sha256_for_file(
                path, mtime_ns=mtime_ns, size_bytes=size_bytes
            )
        except OSError as e:
            worker._emit({"type": "log", "text": f"hash error {path}: {e}"})
            return None

        if worker.db.upsert_file_record(digest, str(path), PIPELINE_VERSION) == "skip":
            with metrics_lock:
                metrics["cache_skip"] = int(metrics["cache_skip"]) + 1
            worker.db.mark_sort_session_item(
                session_key,
                path_norm,
                status="done",
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
                sha256=digest,
                category="cached",
            )
            complete_task(0.0)
            return None

        suf = path.suffix.lower()
        use_video = suf in VIDEO_EXTENSIONS or (suf == GIF_EXTENSION and is_animated_gif(path))
        return {
            "path": path,
            "path_norm": path_norm,
            "mtime_ns": mtime_ns,
            "size_bytes": size_bytes,
            "digest": digest,
            "use_video": use_video,
        }

    prep_workers = max(
        4,
        min(16, int(getattr(worker, "workers", 3)) * 2),
    )
    prep_done = 0
    with ThreadPoolExecutor(max_workers=prep_workers) as pool:
        futures = [pool.submit(_prepare_one, path) for path in files]
        for fut in as_completed(futures):
            prep_done += 1
            if prep_done == 1 or prep_done % 500 == 0 or prep_done == total_files:
                worker._emit(
                    {
                        "type": "log",
                        "text": f"CLIP: подготовка {prep_done}/{total_files}…",
                    }
                )
            if worker._stop.is_set():
                break
            item = fut.result()
            if item is not None:
                pending.append(item)

    worker._emit(
        {
            "type": "log",
            "text": (
                f"CLIP: к классификации {len(pending)} файлов "
                f"(пропущено по кешу/сессии: {int(metrics.get('cache_skip', 0))})."
            ),
        }
    )
    if worker._stop.is_set():
        return "stopped"

    def _vlm_classify(
        path: Path,
        *,
        video_hint: str = "",
        frames: list[Any] | None = None,
    ) -> ClassificationResult:
        with metrics_lock:
            metrics["api_calls"] = int(metrics["api_calls"]) + 1
        if frames:
            hint = (
                "VIDEO MODE: The supplied image is a contact sheet made from chronological "
                "frames of the video. Classify the entire video content, not a single frame."
            )
            if video_hint:
                hint = f"{video_hint}\n\n{hint}"
            data_uri = video_contact_sheet_data_uri(frames)
            return _vlm_via_uri(
                worker,
                data_uri,
                video_hint=hint,
                _prompt_for_request=_prompt_for_request,
                _result_from_raw=_result_from_raw,
                _timed_api_call=_timed_api_call,
                _get_api_base=_get_api_base,
                _timeout=_timeout,
                on_retry=lambda msg: worker._emit({"type": "log", "text": msg}),
            )
        return _vlm_via_uri(
            worker,
            image_to_jpeg_base64_data_uri(path),
            video_hint=video_hint,
            _prompt_for_request=_prompt_for_request,
            _result_from_raw=_result_from_raw,
            _timed_api_call=_timed_api_call,
            _get_api_base=_get_api_base,
            _timeout=_timeout,
            on_retry=lambda msg: worker._emit({"type": "log", "text": msg}),
        )

    def _apply_item(item: dict[str, Any], result: ClassificationResult, *, via: str) -> None:
        t0 = time.monotonic()
        path = item["path"]
        category = result.category
        wl = worker.tag_config.whitelist
        if wl is not None and category not in wl:
            category = UNCATEGORIZED
            result = ClassificationResult(
                UNCATEGORIZED,
                result.candidates,
                result.confidence,
                result.reason_short,
                True,
                result.raw_text,
            )

        if result.needs_review:
            with metrics_lock:
                metrics["needs_review"] = int(metrics["needs_review"]) + 1

        storage_category = resolve_storage_category(category, worker.category_aliases)
        tag_dir = dest_dir / storage_category

        if worker.review_first:
            _record_review(path, item["digest"], result)
            worker.db.mark_sort_session_item(
                session_key,
                item["path_norm"],
                status="done",
                mtime_ns=item["mtime_ns"],
                size_bytes=item["size_bytes"],
                sha256=item["digest"],
                category=category,
            )
            worker._emit(
                {
                    "type": "log",
                    "text": f"review [{via}]: {path.name} -> {category} ({result.confidence:.2f})",
                }
            )
            complete_task(time.monotonic() - t0)
            return

        if not has_disk_space_for_copy(dest_dir, path):
            with metrics_lock:
                metrics["no_space"] = int(metrics["no_space"]) + 1
            complete_task(time.monotonic() - t0)
            return

        try:
            with worker._io_lock:
                dest_file = unique_dest_path(tag_dir, path.name)
                shutil.copy2(path, dest_file)
            worker.db.mark_processed(item["digest"], category, PIPELINE_VERSION)
            worker.db.mark_sort_session_item(
                session_key,
                item["path_norm"],
                status="done",
                mtime_ns=item["mtime_ns"],
                size_bytes=item["size_bytes"],
                sha256=item["digest"],
                category=category,
            )
            with metrics_lock:
                metrics["copied"] = int(metrics["copied"]) + 1
            _record_review(path, item["digest"], result, copied_to=str(dest_file))
            worker._emit(
                {
                    "type": "log",
                    "text": f"[{via} {result.confidence:.2f}] {path.name} -> {storage_category}",
                }
            )
        except OSError as e:
            with metrics_lock:
                metrics["copy_errors"] = int(metrics["copy_errors"]) + 1
            worker._emit({"type": "log", "text": f"copy error {path}: {e!s}"})
        complete_task(time.monotonic() - t0)

    still = [x for x in pending if not x["use_video"]]
    videos = [x for x in pending if x["use_video"]]
    batch_size = settings.batch_size
    video_timeout_s = 90.0

    last_progress_ts = time.monotonic()

    def _maybe_progress() -> None:
        nonlocal last_progress_ts
        now = time.monotonic()
        if now - last_progress_ts < 10.0:
            return
        last_progress_ts = now
        with metrics_lock:
            done = int(metrics.get("fast_classify", 0))
            vlm = int(metrics.get("vlm_fallback", 0))
            review = int(metrics.get("needs_review", 0))
        total = len(still) + len(videos)
        worker._emit(
            {
                "type": "log",
                "text": (
                    f"Прогресс: {done}/{total} CLIP, VLM {vlm}, review {review}"
                ),
            }
        )

    try:
        for offset in range(0, len(still), batch_size):
            if worker._stop.is_set():
                break
            if not worker._wait_if_paused():
                break
            chunk = still[offset : offset + batch_size]
            paths = [x["path"] for x in chunk]
            digests = [x["digest"] for x in chunk]
            if paths:
                worker._emit({"type": "current", "path": str(paths[0])})
            try:
                results = fast.classify_batch(paths, digests=digests)
            except Exception as e:
                worker._emit(
                    {
                        "type": "log",
                        "text": f"CLIP batch error: {e!s} — батч пропущен, файлы в review",
                    }
                )
                results = [
                    ClassificationResult(UNCATEGORIZED, [], 0.0, "batch_error", True, "")
                    for _ in chunk
                ]
            with metrics_lock:
                metrics["fast_classify"] = int(metrics["fast_classify"]) + len(chunk)

            vlm_items: list[tuple[dict[str, Any], ClassificationResult]] = []
            for item, result in zip(chunk, results):
                if vlm_enabled and (
                    result.needs_review or result.category == UNCATEGORIZED
                ):
                    vlm_items.append((item, result))
                else:
                    try:
                        _apply_item(item, result, via="clip")
                    except Exception as e:
                        worker._emit(
                            {"type": "log", "text": f"apply error {item['path']}: {e!s}"}
                        )

            if vlm_items and vlm_enabled:
                with ThreadPoolExecutor(max_workers=max(1, worker.api_workers)) as pool:
                    futures = {
                        pool.submit(_vlm_classify, it["path"]): (it, clip_res)
                        for it, clip_res in vlm_items
                    }
                    for fut in as_completed(futures):
                        item, clip_res = futures[fut]
                        try:
                            vlm_res = fut.result()
                            with metrics_lock:
                                metrics["vlm_fallback"] = int(metrics["vlm_fallback"]) + 1
                            result = _merge_vlm_with_clip(
                                vlm_res,
                                clip_res,
                                confidence_threshold=settings.confidence_threshold,
                            )
                            via = "vlm" if result is not clip_res else "clip"
                        except Exception as e:
                            _register_api_error(e)
                            result = clip_res
                            via = "clip"
                        try:
                            _apply_item(item, result, via=via)
                        except Exception as e:
                            worker._emit(
                                {"type": "log", "text": f"apply error {item['path']}: {e!s}"}
                            )

            with worker.db._lock:
                worker.db._maybe_commit(force=True)
            _maybe_progress()

        for item in videos:
            if worker._stop.is_set():
                break
            if not worker._wait_if_paused():
                break
            path = item["path"]
            worker._emit({"type": "current", "path": str(path)})
            n_frames = max(1, int(getattr(settings, "video_frames", 3)))
            frames: list[Any] = []
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        extract_frames_reduced,
                        path,
                        n_frames,
                        on_log=lambda m: worker._emit({"type": "log", "text": m}),
                    )
                    frames = fut.result(timeout=video_timeout_s) or []
            except Exception as e:
                worker._emit(
                    {"type": "log", "text": f"video frame error {path.name}: {e!s}"}
                )

            try:
                if frames:
                    clip_res = fast.classify_video_frames(path, frames)
                else:
                    clip_res = ClassificationResult(
                        UNCATEGORIZED, [], 0.0, "no_frames", True, ""
                    )
            except Exception as e:
                worker._emit(
                    {"type": "log", "text": f"video clip error {path.name}: {e!s}"}
                )
                clip_res = ClassificationResult(
                    UNCATEGORIZED, [], 0.0, "video_clip_error", True, ""
                )

            result = clip_res
            via = "clip"
            if vlm_enabled and (
                clip_res.needs_review or clip_res.category == UNCATEGORIZED
            ):
                try:
                    vlm_res = _vlm_classify(
                        path,
                        video_hint="VIDEO MODE: classify entire video content.",
                        frames=frames if frames else None,
                    )
                    with metrics_lock:
                        metrics["vlm_fallback"] = int(metrics["vlm_fallback"]) + 1
                    result = _merge_vlm_with_clip(
                        vlm_res,
                        clip_res,
                        confidence_threshold=settings.confidence_threshold,
                    )
                    via = "vlm" if result is not clip_res else "clip"
                except Exception as e:
                    _register_api_error(e)
            try:
                _apply_item(item, result, via=via)
            except Exception as e:
                worker._emit(
                    {"type": "log", "text": f"apply error {path}: {e!s}"}
                )
            with worker.db._lock:
                worker.db._maybe_commit(force=True)
            _maybe_progress()
    finally:
        try:
            with worker.db._lock:
                worker.db._maybe_commit(force=True)
        except Exception:
            pass
    return "stopped" if worker._stop.is_set() else "completed"


def _probe_lm_studio(api_base: str, *, timeout: float = 2.5) -> bool:
    try:
        parts = urlsplit(api_base or "")
        if not parts.scheme or not parts.netloc:
            return False
        base = f"{parts.scheme}://{parts.netloc}"
        req = Request(f"{base}/v1/models", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False

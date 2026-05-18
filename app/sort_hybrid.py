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

from app.category_aliases import resolve_storage_category
from app.classification_result import ClassificationResult
from app.constants import GIF_EXTENSION, PIPELINE_VERSION, UNCATEGORIZED, VIDEO_EXTENSIONS
from app.fast_classify import clip_available, load_fast_classify_settings, missing_clip_message
from app.fast_classify.registry import get_classifier
from app.images import file_sha256, image_to_jpeg_base64_data_uri
from app.lm_studio import chat_completion_cfg
from app.video_frames import extract_frames_reduced, is_animated_gif
from app.worker import has_disk_space_for_copy, unique_dest_path

if TYPE_CHECKING:
    from app.worker import SortWorker


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
) -> None:
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

    worker._emit(
        {
            "type": "log",
            "text": (
                f"Быстрая сортировка: CLIP батч {settings.batch_size}, "
                f"порог {settings.confidence_threshold:.2f}, "
                f"VLM fallback={'да' if settings.vlm_fallback else 'нет'}."
            ),
        }
    )

    pending: list[dict[str, Any]] = []

    for path in files:
        if worker._stop.is_set() or not worker._wait_if_paused():
            break
        path_norm = str(path.resolve())
        try:
            st = path.stat()
            mtime_ns = int(st.st_mtime_ns)
            size_bytes = int(st.st_size)
        except OSError as e:
            worker._emit({"type": "log", "text": f"stat error {path}: {e}"})
            continue

        if worker.resume_session and worker.db.sort_session_item_status(
            session_key, path_norm, mtime_ns=mtime_ns, size_bytes=size_bytes
        ) == "done":
            with metrics_lock:
                metrics["cache_skip"] = int(metrics["cache_skip"]) + 1
            continue

        try:
            digest = file_sha256(path)
        except OSError as e:
            worker._emit({"type": "log", "text": f"hash error {path}: {e}"})
            continue

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
            continue

        suf = path.suffix.lower()
        use_video = suf in VIDEO_EXTENSIONS or (suf == GIF_EXTENSION and is_animated_gif(path))
        pending.append(
            {
                "path": path,
                "path_norm": path_norm,
                "mtime_ns": mtime_ns,
                "size_bytes": size_bytes,
                "digest": digest,
                "use_video": use_video,
            }
        )

    def _vlm_classify(path: Path, *, video_hint: str = "") -> ClassificationResult:
        with metrics_lock:
            metrics["api_calls"] = int(metrics["api_calls"]) + 1
        extra = _prompt_for_request()
        if video_hint:
            extra = f"{extra}\n\n{video_hint}".strip() if extra else video_hint
        data_uri = image_to_jpeg_base64_data_uri(path)
        raw = _timed_api_call(
            lambda: chat_completion_cfg(
                data_uri,
                worker.tag_config,
                api_base=_get_api_base(),
                model=worker.model,
                api_key=worker.api_key,
                timeout=_timeout(),
                on_retry=lambda msg: worker._emit({"type": "log", "text": msg}),
                prompt_extra=extra,
                structured_output=worker.structured_output,
                temperature=worker.temperature,
                max_tokens=worker.max_tokens,
            )
        )
        return _result_from_raw(raw)

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
            worker._emit({"type": "log", "text": f"[{via}] {path.name} -> {storage_category}"})
        except OSError as e:
            with metrics_lock:
                metrics["copy_errors"] = int(metrics["copy_errors"]) + 1
            worker._emit({"type": "log", "text": f"copy error {path}: {e!s}"})
        complete_task(time.monotonic() - t0)

    still = [x for x in pending if not x["use_video"]]
    videos = [x for x in pending if x["use_video"]]
    batch_size = settings.batch_size

    for offset in range(0, len(still), batch_size):
        if worker._stop.is_set():
            break
        chunk = still[offset : offset + batch_size]
        paths = [x["path"] for x in chunk]
        if paths:
            worker._emit({"type": "current", "path": str(paths[0])})
        results = fast.classify_batch(paths)
        with metrics_lock:
            metrics["fast_classify"] = int(metrics["fast_classify"]) + len(chunk)

        vlm_items: list[tuple[dict[str, Any], ClassificationResult]] = []
        for item, result in zip(chunk, results):
            if settings.vlm_fallback and (
                result.needs_review or result.category == UNCATEGORIZED
            ):
                vlm_items.append((item, result))
            else:
                _apply_item(item, result, via="clip")

        if vlm_items and settings.vlm_fallback:
            with ThreadPoolExecutor(max_workers=max(1, worker.api_workers)) as pool:
                futures = {
                    pool.submit(_vlm_classify, it["path"]): (it, clip_res)
                    for it, clip_res in vlm_items
                }
                for fut in as_completed(futures):
                    item, clip_res = futures[fut]
                    try:
                        result = fut.result()
                        with metrics_lock:
                            metrics["vlm_fallback"] = int(metrics["vlm_fallback"]) + 1
                        if result.category == UNCATEGORIZED and clip_res.category != UNCATEGORIZED:
                            result = clip_res
                    except Exception as e:
                        _register_api_error(e)
                        result = clip_res
                    _apply_item(item, result, via="vlm")

    for item in videos:
        if worker._stop.is_set():
            break
        path = item["path"]
        worker._emit({"type": "current", "path": str(path)})
        try:
            frames = extract_frames_reduced(
                path, 1, on_log=lambda m: worker._emit({"type": "log", "text": m})
            )
            if frames:
                result = fast.classify_image(path, frames[len(frames) // 2])
            else:
                result = ClassificationResult(UNCATEGORIZED, [], 0.0, "no_frames", True, "")
        except Exception as e:
            worker._emit({"type": "log", "text": f"video frame error {path.name}: {e!s}"})
            result = ClassificationResult(UNCATEGORIZED, [], 0.0, "video_error", True, "")

        if settings.vlm_fallback and (
            result.needs_review or result.category == UNCATEGORIZED
        ):
            try:
                result = _vlm_classify(
                    path,
                    video_hint="VIDEO MODE: classify entire video content.",
                )
                with metrics_lock:
                    metrics["vlm_fallback"] = int(metrics["vlm_fallback"]) + 1
            except Exception as e:
                _register_api_error(e)
        _apply_item(item, result, via="clip")

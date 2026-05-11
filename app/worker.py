"""Background worker: scan, classify via API, copy files, update DB."""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from app.categorizer import normalize_tag, normalize_tag_auto, normalize_tag_free
from app.category_aliases import aliases_to_prompt_lines
from app.classification_result import ClassificationResult, parse_classification_result
from app.constants import (
    CANONICAL_CATEGORY_WHITELIST,
    CLASSIFY_FILE_MAX_ATTEMPTS,
    COPY_FREE_MARGIN_BYTES,
    DEFAULT_API_KEY,
    ETA_ROLLING_WINDOW,
    GENERAL_CATEGORY_WHITELIST,
    GIF_EXTENSION,
    MediaScanMode,
    PIPELINE_VERSION,
    STILL_IMAGE_EXTENSIONS,
    UNCATEGORIZED,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
)
from app.db import Database, make_sort_session_key
from app.images import file_sha256, image_to_jpeg_base64_data_uri, pil_image_to_jpeg_data_uri
from app.lm_studio import CHAT_COMPLETION_MAX_TOKENS, chat_completion, chat_completion_multi, classify_frames
from app.review_manifest import SortReviewManifest
from app.task_state import TaskState
from app.video_frames import extract_frames_reduced, is_animated_gif


def iter_media_files(root: Path, mode: MediaScanMode) -> list[Path]:
    out: list[Path] = []
    root = root.resolve()
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if mode == MediaScanMode.PHOTOS_ONLY:
            if suf in STILL_IMAGE_EXTENSIONS:
                out.append(p)
        elif mode == MediaScanMode.VIDEO_ONLY:
            if suf in VIDEO_EXTENSIONS or suf == GIF_EXTENSION:
                out.append(p)
        else:
            if suf in STILL_IMAGE_EXTENSIONS or suf in VIDEO_EXTENSIONS or suf == GIF_EXTENSION:
                out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def iter_image_files(root: Path) -> list[Path]:
    """Совместимость: только фото (статичные расширения)."""
    return iter_media_files(root, MediaScanMode.PHOTOS_ONLY)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _disk_usage_root(path: Path) -> Path:
    p = path.resolve()
    cur = p
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur if cur.exists() else p


def has_disk_space_for_copy(dest_dir: Path, source_file: Path, margin: int = COPY_FREE_MARGIN_BYTES) -> bool:
    try:
        need = int(source_file.stat().st_size)
    except OSError:
        return True
    root = _disk_usage_root(dest_dir)
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        return True
    return free >= need + margin


def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).name
    candidate = dest_dir / base
    if not candidate.exists():
        return candidate
    stem = Path(base).stem
    ext = Path(base).suffix
    for i in range(1, 10_000):
        c = dest_dir / f"{stem}_{i}{ext}"
        if not c.exists():
            return c
    raise OSError("could not find free destination name")


class SortWorker:
    def __init__(
        self,
        db: Database,
        out_queue: queue.Queue,
        *,
        api_base: str,
        model: str,
        api_key: str | None = None,
        workers: int = 3,
        api_workers: int = 1,
        free_tag_mode: bool = False,
        auto_tag_mode: bool = False,
        general_tag_mode: bool = False,
        prompt_extra: str = "",
        structured_output: bool = True,
        review_first: bool = False,
        category_aliases: dict[str, str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = CHAT_COMPLETION_MAX_TOKENS,
        request_timeout_sec: float | None = None,
        session_key: str | None = None,
        resume_session: bool = False,
    ) -> None:
        self.db = db
        self.out_queue = out_queue
        self.api_base = api_base
        self.model = model
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self.workers = max(1, min(4, int(workers)))
        self.api_workers = max(1, min(4, int(api_workers)))
        self.free_tag_mode = bool(free_tag_mode)
        self.auto_tag_mode = bool(auto_tag_mode)
        self.general_tag_mode = bool(general_tag_mode)
        self.prompt_extra = str(prompt_extra or "")
        self.structured_output = bool(structured_output)
        self.review_first = bool(review_first)
        self.category_aliases = dict(category_aliases or {})
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.max_tokens = max(1, min(4096, int(max_tokens)))
        self.request_timeout_sec = request_timeout_sec
        self.session_key = session_key
        self.resume_session = bool(resume_session)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # when set, NOT paused (worker runs)
        self._io_lock = threading.Lock()
        self._run_id: str | None = None

    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause.clear()
            self._emit({"type": "state_changed", "state": TaskState.PAUSED.value})
        else:
            self._pause.set()
            self._emit({"type": "state_changed", "state": TaskState.RUNNING.value})

    def request_stop(self) -> None:
        self._stop.set()
        self._emit({"type": "state_changed", "state": TaskState.STOPPING.value})

    def reset_stop(self) -> None:
        self._stop.clear()

    def _emit(self, msg: dict[str, Any]) -> None:
        if self._run_id is not None and "run_id" not in msg:
            msg["run_id"] = self._run_id
        try:
            self.out_queue.put_nowait(msg)
        except queue.Full:
            pass

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def _wait_if_paused(self) -> bool:
        """Wait while paused. Returns False if should abort."""
        while not self._pause.is_set():
            if self._stop.is_set():
                return False
            threading.Event().wait(0.1)
        return not self._stop.is_set()

    def run_batch(
        self,
        source_dir: Path,
        dest_dir: Path,
        user_context: str,
        *,
        media_mode: MediaScanMode = MediaScanMode.PHOTOS_ONLY,
    ) -> None:
        finish_reason = "completed"
        started_ts = time.monotonic()
        metrics = {
            "cache_skip": 0,
            "api_errors": 0,
            "copied": 0,
            "copy_errors": 0,
            "no_space": 0,
            "api_calls": 0,
            "api_latency_sec": 0.0,
            "needs_review": 0,
            "review_written": 0,
        }
        try:
            self._emit({"type": "state_changed", "state": TaskState.RUNNING.value})
            source_dir = source_dir.resolve()
            dest_dir = dest_dir.resolve()
            files = iter_media_files(source_dir, media_mode)
            if source_dir != dest_dir and _is_relative_to(dest_dir, source_dir):
                before = len(files)
                files = [p for p in files if not _is_relative_to(p, dest_dir)]
                skipped_outputs = before - len(files)
                if skipped_outputs > 0:
                    self._emit(
                        {
                            "type": "log",
                            "text": (
                                f"Папка результата внутри источника: пропущено уже лежащих "
                                f"в результате файлов: {skipped_outputs}."
                            ),
                        }
                    )
            total = len(files)
            self._emit({"type": "scan_done", "total": total})
            tag_mode = (
                "auto"
                if self.auto_tag_mode
                else ("free" if self.free_tag_mode else ("general" if self.general_tag_mode else "strict"))
            )
            session_key = self.session_key or make_sort_session_key(
                str(source_dir),
                str(dest_dir),
                media_mode.value,
                tag_mode,
                self.review_first,
                PIPELINE_VERSION,
            )

            def _session_payload() -> dict[str, Any]:
                return {
                    "model": self.model,
                    "workers": self.workers,
                    "api_workers": self.api_workers,
                    "structured_output": self.structured_output,
                    "prompt_extra": self.prompt_extra,
                    "auto_tag_mode": self.auto_tag_mode,
                    "free_tag_mode": self.free_tag_mode,
                    "general_tag_mode": self.general_tag_mode,
                }

            def save_session(status: str, done_files: int, *, force: bool = False) -> None:
                try:
                    self.db.upsert_sort_session(
                        session_key=session_key,
                        source_dir=str(source_dir),
                        dest_dir=str(dest_dir),
                        media_mode=media_mode.value,
                        tag_mode=tag_mode,
                        review_first=self.review_first,
                        pipeline_version=PIPELINE_VERSION,
                        total_files=total,
                        done_files=done_files,
                        status=status,
                        payload=_session_payload(),
                    )
                except Exception as e:
                    if force:
                        self._emit({"type": "log", "text": f"session save error: {e!s}"})

            save_session("running", 0)
            self._emit(
                {
                    "type": "session",
                    "session_key": session_key,
                    "status": "running",
                    "done": 0,
                    "total": total,
                }
            )
            manifest = (
                SortReviewManifest(
                    dest_dir,
                    source_dir=source_dir,
                    media_mode=media_mode.value,
                    tag_mode=tag_mode,
                    model=self.model,
                )
                if self.review_first
                else None
            )
            if manifest is not None:
                self._emit({"type": "log", "text": f"Review-first manifest: {manifest.path}"})

            done = 0
            prog_lock = threading.Lock()
            durations: deque[float] = deque(maxlen=ETA_ROLLING_WINDOW)
            times_lock = threading.Lock()
            error_times: deque[float] = deque()
            storm_guard_until = 0.0
            storm_lock = threading.Lock()
            api_slots = threading.BoundedSemaphore(self.api_workers)
            if self.workers > self.api_workers:
                self._emit(
                    {
                        "type": "log",
                        "text": (
                            f"Файловых потоков: {self.workers}; одновременных запросов к LM: {self.api_workers}. "
                            "Это безопаснее для LM Studio, который часто нестабилен при параллельных vision-запросах."
                        ),
                    }
                )

            def _run_api_call(fn: Callable[[], str]) -> str:
                while not self._stop.is_set():
                    if api_slots.acquire(timeout=0.1):
                        break
                else:
                    raise RuntimeError("stopped before API request")
                try:
                    return fn()
                finally:
                    api_slots.release()

            def _register_api_error(exc: Exception) -> None:
                nonlocal storm_guard_until
                now = time.monotonic()
                with storm_lock:
                    error_times.append(now)
                    window_sec = 12.0
                    while error_times and now - error_times[0] > window_sec:
                        error_times.popleft()
                    threshold = max(6, self.workers * 3)
                    if len(error_times) >= threshold:
                        cooldown = 1.2
                        if now + cooldown > storm_guard_until:
                            storm_guard_until = now + cooldown
                            self._emit(
                                {
                                    "type": "log",
                                    "text": (
                                        "Channel Error detected, applying transient backoff "
                                        f"(~{cooldown:.1f}s) to prevent retry storm."
                                    ),
                                }
                            )

            def _storm_guard_sleep_if_needed() -> None:
                with storm_lock:
                    wait_for = max(0.0, storm_guard_until - time.monotonic())
                if wait_for > 0:
                    _sleep_or_stopped(wait_for)

            def _sleep_or_stopped(seconds: float) -> bool:
                until = time.monotonic() + max(0.0, seconds)
                while not self._stop.is_set():
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        return True
                    time.sleep(min(0.1, remaining))
                return False

            def complete_task(elapsed_sec: float) -> None:
                nonlocal done
                with times_lock:
                    durations.append(elapsed_sec)
                    avg = sum(durations) / len(durations)
                with prog_lock:
                    done += 1
                    rem = total - done
                    workers_eff = max(1, min(self.workers, self.api_workers, rem)) if rem > 0 else 1
                    eta_sec = (rem * avg) / workers_eff if rem > 0 and durations else 0.0
                    self._emit(
                        {
                            "type": "progress",
                            "done": done,
                            "total": total,
                            "eta_sec": eta_sec,
                        }
                    )
                    if done % 20 == 0 or done == total:
                        save_session("running", done)

            def _mode_name() -> str:
                if self.auto_tag_mode:
                    return "auto"
                if self.free_tag_mode:
                    return "free"
                if self.general_tag_mode:
                    return "general"
                return "strict"

            def _prompt_for_request() -> str:
                alias_block = aliases_to_prompt_lines(self.category_aliases)
                if alias_block and self.prompt_extra.strip():
                    return self.prompt_extra.strip() + "\n\n" + alias_block
                return alias_block or self.prompt_extra

            def _timeout() -> tuple[float, float] | None:
                if self.request_timeout_sec is None:
                    return None
                return (30.0, max(30.0, float(self.request_timeout_sec)))

            def _result_from_raw(raw: str) -> ClassificationResult:
                return parse_classification_result(
                    raw,
                    mode=_mode_name(),  # type: ignore[arg-type]
                    aliases=self.category_aliases,
                )

            def _emit_health(last_api_sec: float | None = None) -> None:
                avg = (
                    metrics["api_latency_sec"] / metrics["api_calls"]
                    if metrics["api_calls"] > 0
                    else 0.0
                )
                self._emit(
                    {
                        "type": "health",
                        "payload": {
                            "model": self.model,
                            "api_calls": metrics["api_calls"],
                            "avg_api_sec": round(avg, 2),
                            "last_api_sec": round(last_api_sec or 0.0, 2),
                            "api_errors": metrics["api_errors"],
                            "api_workers": self.api_workers,
                            "needs_review": metrics["needs_review"],
                            "review_written": metrics["review_written"],
                        },
                    }
                )

            def _record_review(
                path: Path,
                digest: str,
                result: ClassificationResult,
                *,
                copied_to: str | None = None,
            ) -> None:
                if manifest is None and not result.needs_review:
                    return
                review_dir = manifest
                if review_dir is None:
                    return
                try:
                    st = path.stat()
                    size_bytes = int(st.st_size)
                except OSError:
                    size_bytes = 0
                review_dir.append(
                    {
                        "source_path": str(path),
                        "sha256": digest,
                        "size_bytes": size_bytes,
                        "category": result.category,
                        "candidates": result.candidates,
                        "confidence": result.confidence,
                        "reason_short": result.reason_short,
                        "needs_review": result.needs_review,
                        "copied_to": copied_to,
                        "raw_model_output": result.raw_text[:4000],
                    }
                )
                metrics["review_written"] += 1

            def process_one(path: Path) -> None:
                if self._stop.is_set():
                    return
                if not self._wait_if_paused():
                    return
                _storm_guard_sleep_if_needed()

                t0 = time.monotonic()
                self._emit({"type": "current", "path": str(path)})
                path_norm = str(path.resolve())
                try:
                    st = path.stat()
                    mtime_ns = int(st.st_mtime_ns)
                    size_bytes = int(st.st_size)
                except OSError as e:
                    self._emit({"type": "log", "text": f"stat error {path}: {e}"})
                    complete_task(time.monotonic() - t0)
                    return

                if self.resume_session and self.db.sort_session_item_status(
                    session_key,
                    path_norm,
                    mtime_ns=mtime_ns,
                    size_bytes=size_bytes,
                ) == "done":
                    metrics["cache_skip"] += 1
                    self._emit({"type": "log", "text": f"resume skip: {path}"})
                    complete_task(time.monotonic() - t0)
                    return

                try:
                    digest = file_sha256(path)
                except OSError as e:
                    self._emit({"type": "log", "text": f"hash error {path}: {e}"})
                    complete_task(time.monotonic() - t0)
                    return

                skip = self.db.upsert_file_record(digest, str(path), PIPELINE_VERSION)
                if skip == "skip":
                    metrics["cache_skip"] += 1
                    self.db.mark_sort_session_item(
                        session_key,
                        path_norm,
                        status="done",
                        mtime_ns=mtime_ns,
                        size_bytes=size_bytes,
                        sha256=digest,
                        category="cached",
                    )
                    self._emit({"type": "log", "text": f"skip (already processed): {path}"})
                    complete_task(time.monotonic() - t0)
                    return

                category = UNCATEGORIZED
                result = ClassificationResult(UNCATEGORIZED, [], 0.0, "", True, "")
                for attempt in range(1, CLASSIFY_FILE_MAX_ATTEMPTS + 1):
                    try:
                        suf = path.suffix.lower()
                        use_video_pipeline = suf in VIDEO_EXTENSIONS or (
                            suf == GIF_EXTENSION and is_animated_gif(path)
                        )

                        if use_video_pipeline:
                            frames = extract_frames_reduced(
                                path,
                                VIDEO_FRAME_COUNT,
                                on_log=lambda m: self._emit({"type": "log", "text": m}),
                            )
                            if not frames:
                                self._emit(
                                    {
                                        "type": "log",
                                        "text": f"rollback: no frames decoded {path}",
                                    }
                                )
                                category = UNCATEGORIZED
                                result = ClassificationResult(UNCATEGORIZED, [], 0.0, "no_frames_decoded", True, "")
                            else:
                                uris = [pil_image_to_jpeg_data_uri(im) for im in frames]
                                api_t0 = time.monotonic()
                                if self.structured_output:
                                    raw = _run_api_call(
                                        lambda: chat_completion_multi(
                                            uris[:VIDEO_FRAME_COUNT],
                                            user_context,
                                            api_base=self.api_base,
                                            model=self.model,
                                            api_key=self.api_key,
                                            timeout=_timeout(),
                                            on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                                            free_mode=self.free_tag_mode or self.auto_tag_mode,
                                            auto_mode=self.auto_tag_mode,
                                            general_mode=self.general_tag_mode,
                                            prompt_extra=_prompt_for_request(),
                                            structured_output=True,
                                            temperature=self.temperature,
                                            max_tokens=self.max_tokens,
                                        )
                                    )
                                    result = _result_from_raw(raw)
                                    category = result.category
                                else:
                                    category = _run_api_call(
                                        lambda: classify_frames(
                                            uris,
                                            user_context,
                                            api_base=self.api_base,
                                            model=self.model,
                                            api_key=self.api_key,
                                            timeout=_timeout(),
                                            on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                                            on_log=lambda m: self._emit({"type": "log", "text": m}),
                                            free_mode=self.free_tag_mode or self.auto_tag_mode,
                                            auto_mode=self.auto_tag_mode,
                                            general_mode=self.general_tag_mode,
                                            prompt_extra=_prompt_for_request(),
                                        )
                                    )
                                    result = ClassificationResult(category, [category], 0.75, "legacy_video_output", category == UNCATEGORIZED, "")
                                api_sec = time.monotonic() - api_t0
                                metrics["api_calls"] += 1
                                metrics["api_latency_sec"] += api_sec
                                _emit_health(api_sec)
                        else:
                            data_uri = image_to_jpeg_base64_data_uri(path)
                            api_t0 = time.monotonic()
                            raw = _run_api_call(
                                lambda: chat_completion(
                                    data_uri,
                                    user_context,
                                    api_base=self.api_base,
                                    model=self.model,
                                    api_key=self.api_key,
                                    timeout=_timeout(),
                                    on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                                    free_mode=self.free_tag_mode or self.auto_tag_mode,
                                    auto_mode=self.auto_tag_mode,
                                    general_mode=self.general_tag_mode,
                                    prompt_extra=_prompt_for_request(),
                                    structured_output=self.structured_output,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens,
                                )
                            )
                            api_sec = time.monotonic() - api_t0
                            metrics["api_calls"] += 1
                            metrics["api_latency_sec"] += api_sec
                            result = _result_from_raw(raw)
                            category = result.category
                            if not self.structured_output and self.auto_tag_mode:
                                category = normalize_tag_auto(raw, extra_aliases=self.category_aliases)
                                result = ClassificationResult(category, [category], 0.75, "legacy_tag_output", category == UNCATEGORIZED, raw)
                            elif not self.structured_output and self.free_tag_mode:
                                category = normalize_tag_free(raw)
                                result = ClassificationResult(category, [category], 0.75, "legacy_tag_output", category == UNCATEGORIZED, raw)
                            elif not self.structured_output and self.general_tag_mode:
                                category = normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)
                                result = ClassificationResult(category, [category], 0.75, "legacy_tag_output", category == UNCATEGORIZED, raw)
                            elif not self.structured_output:
                                category = normalize_tag(raw)
                                result = ClassificationResult(category, [category], 0.75, "legacy_tag_output", category == UNCATEGORIZED, raw)
                            _emit_health(api_sec)
                    except Exception as e:
                        metrics["api_errors"] += 1
                        _register_api_error(e)
                        self._emit(
                            {
                                "type": "log",
                                "text": (
                                    f"API error {path}: {e!s} "
                                    f"(attempt {attempt}/{CLASSIFY_FILE_MAX_ATTEMPTS})"
                                ),
                            }
                        )
                        category = UNCATEGORIZED
                    strict_whitelist = GENERAL_CATEGORY_WHITELIST if self.general_tag_mode else CANONICAL_CATEGORY_WHITELIST
                    if not (self.free_tag_mode or self.auto_tag_mode) and category not in strict_whitelist:
                        category = UNCATEGORIZED
                        result = ClassificationResult(UNCATEGORIZED, result.candidates, result.confidence, result.reason_short, True, result.raw_text)
                    if category != UNCATEGORIZED:
                        break
                    if attempt < CLASSIFY_FILE_MAX_ATTEMPTS:
                        self._emit(
                            {
                                "type": "log",
                                "text": (
                                    f"retry classify {path.name}: uncategorized "
                                    f"(attempt {attempt + 1}/{CLASSIFY_FILE_MAX_ATTEMPTS})"
                                ),
                            }
                        )
                        if not _sleep_or_stopped(0.2 * attempt):
                            break

                if self._stop.is_set():
                    complete_task(time.monotonic() - t0)
                    return
                if result.needs_review:
                    metrics["needs_review"] += 1

                tag_dir = dest_dir / category
                if self.review_first:
                    _record_review(path, digest, result)
                    self.db.mark_sort_session_item(
                        session_key,
                        path_norm,
                        status="done",
                        mtime_ns=mtime_ns,
                        size_bytes=size_bytes,
                        sha256=digest,
                        category=category,
                    )
                    self._emit(
                        {
                            "type": "log",
                            "text": f"review: {path.name} -> {category} ({result.confidence:.2f})",
                        }
                    )
                    complete_task(time.monotonic() - t0)
                    return
                try:
                    if not has_disk_space_for_copy(dest_dir, path):
                        metrics["no_space"] += 1
                        self._emit(
                            {
                                "type": "log",
                                "text": (
                                    f"Мало места на диске: {path.name} не скопирован. "
                                    "Файл оставлен pending и будет повторён в следующем запуске."
                                ),
                            }
                        )
                    else:
                        with self._io_lock:
                            dest_file = unique_dest_path(tag_dir, path.name)
                            shutil.copy2(path, dest_file)
                        self.db.mark_processed(digest, category, PIPELINE_VERSION)
                        self.db.mark_sort_session_item(
                            session_key,
                            path_norm,
                            status="done",
                            mtime_ns=mtime_ns,
                            size_bytes=size_bytes,
                            sha256=digest,
                            category=category,
                        )
                        metrics["copied"] += 1
                        _record_review(path, digest, result, copied_to=str(dest_file))
                        self._emit({"type": "log", "text": f"{path.name} -> {category}"})
                except OSError as e:
                    metrics["copy_errors"] += 1
                    self._emit(
                        {
                            "type": "log",
                            "text": (
                                f"copy error {path}: {e!s}. "
                                "Файл оставлен pending и будет повторён в следующем запуске."
                            ),
                        }
                    )

                complete_task(time.monotonic() - t0)

            if not files:
                save_session("completed", 0, force=True)
                return

            workers = max(1, min(self.workers, len(files)))
            executor = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = [executor.submit(process_one, p) for p in files]
                try:
                    for fut in as_completed(futures):
                        if self._stop.is_set():
                            finish_reason = "stopped"
                            break
                        try:
                            fut.result()
                        except Exception as e:
                            self._emit({"type": "log", "text": f"task error: {e!s}"})
                finally:
                    if self._stop.is_set():
                        finish_reason = "stopped"
                        for fut in futures:
                            fut.cancel()
            finally:
                executor.shutdown(wait=True, cancel_futures=self._stop.is_set())
            if finish_reason == "stopped" and done >= total:
                finish_reason = "completed"

        except Exception as e:
            self._emit({"type": "log", "text": f"fatal: {e!s}"})
            finish_reason = "error"
        finally:
            elapsed = max(0.001, time.monotonic() - started_ts)
            if "save_session" in locals():
                session_status = "completed" if finish_reason == "completed" else finish_reason
                save_session(session_status, done if "done" in locals() else 0, force=True)
                self._emit(
                    {
                        "type": "session",
                        "session_key": session_key if "session_key" in locals() else "",
                        "status": session_status,
                        "done": done if "done" in locals() else 0,
                        "total": total if "total" in locals() else 0,
                    }
                )
            self._emit(
                {
                    "type": "metric",
                    "name": "sort_summary",
                    "payload": {
                        "elapsed_sec": round(elapsed, 2),
                        "files_per_sec": round(done / elapsed, 2) if "done" in locals() else 0.0,
                        "cache_skip": metrics["cache_skip"],
                        "api_errors": metrics["api_errors"],
                        "copied": metrics["copied"],
                        "copy_errors": metrics["copy_errors"],
                        "no_space": metrics["no_space"],
                        "api_calls": metrics["api_calls"],
                        "avg_api_sec": round(
                            metrics["api_latency_sec"] / metrics["api_calls"],
                            2,
                        )
                        if metrics["api_calls"]
                        else 0.0,
                        "needs_review": metrics["needs_review"],
                        "review_written": metrics["review_written"],
                    },
                }
            )
            if finish_reason == "stopped":
                self._emit({"type": "state_changed", "state": TaskState.STOPPED.value})
            elif finish_reason == "error":
                self._emit({"type": "state_changed", "state": TaskState.ERROR.value})
            else:
                self._emit({"type": "state_changed", "state": TaskState.FINISHED.value})
            self._emit({"type": "finished", "reason": finish_reason})

    def start_in_thread(
        self,
        source_dir: Path,
        dest_dir: Path,
        user_context: str,
        *,
        media_mode: MediaScanMode = MediaScanMode.PHOTOS_ONLY,
        on_complete: Callable[[], None] | None = None,
        session_key: str | None = None,
        resume_session: bool = False,
    ) -> None:
        self._run_id = uuid.uuid4().hex
        if session_key:
            self.session_key = session_key
        self.resume_session = bool(resume_session)

        def target() -> None:
            try:
                self.run_batch(source_dir, dest_dir, user_context, media_mode=media_mode)
            finally:
                if on_complete:
                    on_complete()
                self._run_id = None

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

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
from app.constants import (
    CANONICAL_CATEGORY_WHITELIST,
    CLASSIFY_FILE_MAX_ATTEMPTS,
    COPY_FREE_MARGIN_BYTES,
    DEFAULT_API_KEY,
    ETA_ROLLING_WINDOW,
    GIF_EXTENSION,
    MediaScanMode,
    PIPELINE_VERSION,
    STILL_IMAGE_EXTENSIONS,
    UNCATEGORIZED,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
)
from app.db import Database
from app.images import file_sha256, image_to_jpeg_base64_data_uri, pil_image_to_jpeg_data_uri
from app.lm_studio import chat_completion, classify_frames
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
        free_tag_mode: bool = False,
        auto_tag_mode: bool = False,
        prompt_extra: str = "",
    ) -> None:
        self.db = db
        self.out_queue = out_queue
        self.api_base = api_base
        self.model = model
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self.workers = max(1, min(4, int(workers)))
        self.free_tag_mode = bool(free_tag_mode)
        self.auto_tag_mode = bool(auto_tag_mode)
        self.prompt_extra = str(prompt_extra or "")

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
        metrics = {"cache_skip": 0, "api_errors": 0, "copied": 0, "copy_errors": 0, "no_space": 0}
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

            done = 0
            prog_lock = threading.Lock()
            durations: deque[float] = deque(maxlen=ETA_ROLLING_WINDOW)
            times_lock = threading.Lock()
            error_times: deque[float] = deque()
            storm_guard_until = 0.0
            storm_lock = threading.Lock()

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
                    workers_eff = max(1, min(self.workers, rem)) if rem > 0 else 1
                    eta_sec = (rem * avg) / workers_eff if rem > 0 and durations else 0.0
                    self._emit(
                        {
                            "type": "progress",
                            "done": done,
                            "total": total,
                            "eta_sec": eta_sec,
                        }
                    )

            def process_one(path: Path) -> None:
                if self._stop.is_set():
                    return
                if not self._wait_if_paused():
                    return
                _storm_guard_sleep_if_needed()

                t0 = time.monotonic()
                self._emit({"type": "current", "path": str(path)})

                try:
                    digest = file_sha256(path)
                except OSError as e:
                    self._emit({"type": "log", "text": f"hash error {path}: {e}"})
                    complete_task(time.monotonic() - t0)
                    return

                skip = self.db.upsert_file_record(digest, str(path), PIPELINE_VERSION)
                if skip == "skip":
                    metrics["cache_skip"] += 1
                    self._emit({"type": "log", "text": f"skip (already processed): {path}"})
                    complete_task(time.monotonic() - t0)
                    return

                category = UNCATEGORIZED
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
                            else:
                                uris = [pil_image_to_jpeg_data_uri(im) for im in frames]
                                category = classify_frames(
                                    uris,
                                    user_context,
                                    api_base=self.api_base,
                                    model=self.model,
                                    api_key=self.api_key,
                                    on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                                    on_log=lambda m: self._emit({"type": "log", "text": m}),
                                    free_mode=self.free_tag_mode or self.auto_tag_mode,
                                    auto_mode=self.auto_tag_mode,
                                    prompt_extra=self.prompt_extra,
                                )
                        else:
                            data_uri = image_to_jpeg_base64_data_uri(path)
                            raw = chat_completion(
                                data_uri,
                                user_context,
                                api_base=self.api_base,
                                model=self.model,
                                api_key=self.api_key,
                                on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                                free_mode=self.free_tag_mode or self.auto_tag_mode,
                                auto_mode=self.auto_tag_mode,
                                prompt_extra=self.prompt_extra,
                            )
                            if self.auto_tag_mode:
                                category = normalize_tag_auto(raw)
                            elif self.free_tag_mode:
                                category = normalize_tag_free(raw)
                            else:
                                category = normalize_tag(raw)
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
                    if not (self.free_tag_mode or self.auto_tag_mode) and category not in CANONICAL_CATEGORY_WHITELIST:
                        category = UNCATEGORIZED
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

                tag_dir = dest_dir / category
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
                        metrics["copied"] += 1
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

        except Exception as e:
            self._emit({"type": "log", "text": f"fatal: {e!s}"})
            finish_reason = "error"
        finally:
            elapsed = max(0.001, time.monotonic() - started_ts)
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
    ) -> None:
        self._run_id = uuid.uuid4().hex

        def target() -> None:
            try:
                self.run_batch(source_dir, dest_dir, user_context, media_mode=media_mode)
            finally:
                if on_complete:
                    on_complete()
                self._run_id = None

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()



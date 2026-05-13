# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Background worker: scan folder, compute signatures, emit duplicate groups with staged progress."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.constants import DEFAULT_API_KEY, GIF_EXTENSION, MediaScanMode, STILL_IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.duplicate_finder import (
    DuplicateFinderOptions,
    build_groups_from_records,
    collect_llm_verification_pairs,
    compute_one_signature,
    hamming_matrix_stats,
    iter_dup_scan_paths,
    load_records_from_db,
    max_duplicate_workers,
)
from app.images import image_to_jpeg_base64_data_uri, pil_image_to_jpeg_data_uri
from app.lm_studio import pair_images_duplicate_decision
from app.signature_db import SignatureDatabase, make_session_key
from app.task_state import TaskState
from app.video_frames import extract_frames_reduced, is_animated_gif


def _exact_only_candidates(paths: list[Path]) -> tuple[list[Path], int]:
    """For exact duplicates, files with unique sizes cannot match."""
    by_size: dict[int, list[Path]] = {}
    for p in paths:
        try:
            size = int(p.stat().st_size)
        except OSError:
            continue
        by_size.setdefault(size, []).append(p)
    candidates: list[Path] = []
    skipped = 0
    for bucket in by_size.values():
        if len(bucket) > 1:
            candidates.extend(bucket)
        else:
            skipped += 1
    return candidates, skipped


def path_to_jpeg_data_uri(path: Path, video_frames: int, on_log) -> str | None:
    suf = path.suffix.lower()
    try:
        if suf in STILL_IMAGE_EXTENSIONS:
            return image_to_jpeg_base64_data_uri(path)
        if suf == GIF_EXTENSION:
            if is_animated_gif(path):
                frames = extract_frames_reduced(path, max(1, video_frames), on_log=on_log)
                return pil_image_to_jpeg_data_uri(frames[0]) if frames else None
            return image_to_jpeg_base64_data_uri(path)
        if suf in VIDEO_EXTENSIONS:
            frames = extract_frames_reduced(path, max(1, video_frames), on_log=on_log)
            return pil_image_to_jpeg_data_uri(frames[0]) if frames else None
    except Exception as e:
        if callable(on_log):
            try:
                on_log(f"jpeg preview error {path.name}: {e!s}")
            except Exception:
                pass
        return None
    return None


class DuplicateFinderWorker:
    def __init__(
        self,
        sig_db: SignatureDatabase,
        out_queue: queue.Queue,
        *,
        api_base: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.sig_db = sig_db
        self.out_queue = out_queue
        self.api_base = api_base
        self.model = model
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._run_id: str | None = None

    def request_stop(self) -> None:
        self._stop.set()
        self._emit({"type": "state_changed", "state": TaskState.STOPPING.value})

    def reset_stop(self) -> None:
        self._stop.clear()

    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause.clear()
            self._emit({"type": "state_changed", "state": TaskState.PAUSED.value})
        else:
            self._pause.set()
            self._emit({"type": "state_changed", "state": TaskState.RUNNING.value})

    def _wait_if_paused(self) -> bool:
        while not self._pause.is_set():
            if self._stop.is_set():
                return False
            threading.Event().wait(0.1)
        return not self._stop.is_set()

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

    def _emit_stage(self, name: str, done: int, total: int) -> None:
        self._emit({"type": "dup_stage", "stage": name})
        self._emit({"type": "dup_stage_progress", "stage": name, "done": done, "total": total})

    def run_scan(
        self,
        root: Path,
        options: DuplicateFinderOptions,
        *,
        media_mode: MediaScanMode = MediaScanMode.PHOTOS_ONLY,
        force_recompute: bool = False,
        resume: bool = False,
    ) -> None:
        finish_reason = "completed"
        wall_t0 = time.monotonic()
        metrics = {"cache_hits": 0, "sig_computed": 0, "llm_pairs": 0}
        try:
            self._emit({"type": "state_changed", "state": TaskState.RUNNING.value})
            root = root.resolve()
            session_key = make_session_key(str(root), media_mode.value, options.strictness)
            if force_recompute:
                self.sig_db.clear_session(session_key)
            paths = iter_dup_scan_paths(root, media_mode)
            scanned_total = len(paths)
            exact_only = bool(options.include_exact and not options.include_perceptual and not options.include_semantic and not options.use_llm_pairs)
            if exact_only:
                paths, skipped_unique_sizes = _exact_only_candidates(paths)
                if skipped_unique_sizes:
                    self._emit(
                        {
                            "type": "log",
                            "text": (
                                "Дубликаты: быстрый точный режим пропустил "
                                f"{skipped_unique_sizes} файлов с уникальным размером без чтения содержимого."
                            ),
                        }
                    )
            total = len(paths)
            self._emit({"type": "dup_scan_done", "total": scanned_total})
            self._emit_stage("scan_signatures", 0, total)
            if not paths:
                self._emit({"type": "dup_groups_ready", "groups": [], "records": [], "records_count": 0})
                return

            records: list[Any] = []
            done = 0
            lock = threading.Lock()
            workers = max(1, min(max_duplicate_workers(), options.parallel_workers, len(paths)))

            def process_one(p: Path) -> None:
                nonlocal done
                if self._stop.is_set():
                    return
                if not self._wait_if_paused():
                    return
                pnorm = str(p.resolve())
                if resume and not force_recompute and self.sig_db.session_item_status(session_key, pnorm) == "done":
                    cached = load_records_from_db([p], self.sig_db, options)
                    with lock:
                        done += 1
                        metrics["cache_hits"] += 1
                        if cached:
                            records.append(cached[0])
                        self._emit({"type": "dup_progress", "done": done, "total": total, "path": str(p)})
                        self._emit_stage("scan_signatures", done, total)
                    return

                info = compute_one_signature(
                    p,
                    self.sig_db,
                    options,
                    force_recompute=force_recompute,
                    on_log=lambda m: self._emit({"type": "log", "text": m}),
                )
                with lock:
                    done += 1
                    metrics["sig_computed"] += 1
                    if info is not None:
                        records.append(info)
                    self.sig_db.mark_session_item(session_key, pnorm, "done")
                    self._emit({"type": "dup_progress", "done": done, "total": total, "path": str(p)})
                    self._emit_stage("scan_signatures", done, total)
                    if done % 20 == 0 or done == total:
                        self.sig_db.upsert_session(
                            session_key=session_key,
                            root_path=str(root),
                            media_mode=media_mode.value,
                            strictness=options.strictness,
                            stage="scan_signatures",
                            total_files=total,
                            done_files=done,
                            llm_total_pairs=0,
                            llm_done_pairs=0,
                            status="running",
                            payload={"paths": [str(x.path) for x in records]},
                        )

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, p) for p in paths]
                for fut in as_completed(futs):
                    if self._stop.is_set():
                        finish_reason = "stopped"
                        break
                    try:
                        fut.result()
                    except Exception as e:
                        self._emit({"type": "log", "text": f"dup task: {e!s}"})

            if self._stop.is_set():
                finish_reason = "stopped"

            records.sort(key=lambda r: str(r.path).lower())

            llm_cache: dict[tuple[str, str], bool] = {}
            if options.use_llm_pairs and records:
                max_pairs = 15000
                pairs = collect_llm_verification_pairs(records, options, max_pairs)
                self._emit(
                    {
                        "type": "log",
                        "text": f"LLM: кандидатов пар (по корзинам размеров, max {max_pairs}): {len(pairs)}",
                    }
                )

                llm_total = len(pairs)
                llm_done = 0
                self._emit_stage("llm_verify", 0, llm_total if llm_total > 0 else 1)

                def one_pair(pair: tuple[Any, Any]) -> tuple[str, str, bool]:
                    a, b = pair
                    pa, pb = sorted((str(a.path.resolve()), str(b.path.resolve())))
                    cached = self.sig_db.get_llm_pair_decision(session_key, pa, pb)
                    if cached is not None:
                        return pa, pb, cached
                    ua = path_to_jpeg_data_uri(Path(pa), options.video_sample_frames, lambda _m: None)
                    ub = path_to_jpeg_data_uri(Path(pb), options.video_sample_frames, lambda _m: None)
                    if not ua or not ub:
                        return pa, pb, False
                    dec = pair_images_duplicate_decision(
                        ua,
                        ub,
                        api_base=self.api_base,
                        model=self.model,
                        api_key=self.api_key,
                        on_retry=lambda msg: self._emit({"type": "log", "text": msg}),
                    )
                    self.sig_db.save_llm_pair_decision(session_key, pa, pb, dec)
                    return pa, pb, dec

                if llm_total > 0:
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = [ex.submit(one_pair, p) for p in pairs]
                        for fut in as_completed(futs):
                            if self._stop.is_set():
                                finish_reason = "stopped"
                                break
                            try:
                                pa, pb, dec = fut.result()
                                llm_cache[(pa, pb)] = dec
                                metrics["llm_pairs"] += 1
                            except Exception as e:
                                self._emit({"type": "log", "text": f"llm pair: {e!s}"})
                            llm_done += 1
                            self._emit_stage("llm_verify", llm_done, llm_total)
                            if llm_done % 10 == 0 or llm_done == llm_total:
                                self.sig_db.upsert_session(
                                    session_key=session_key,
                                    root_path=str(root),
                                    media_mode=media_mode.value,
                                    strictness=options.strictness,
                                    stage="llm_verify",
                                    total_files=total,
                                    done_files=done,
                                    llm_total_pairs=llm_total,
                                    llm_done_pairs=llm_done,
                                    status="running",
                                    payload={"paths": [str(x.path) for x in records]},
                                )

            merged_llm: dict[tuple[str, str], bool] | None = None
            if options.use_llm_pairs and records:
                merged_llm = self.sig_db.list_llm_pair_decisions(session_key)
                merged_llm.update(llm_cache)

            self._emit({"type": "log", "text": "Дубликаты: сборка групп…"})
            self._emit_stage("grouping", 0, 1)

            groups = build_groups_from_records(
                records,
                options,
                on_log=lambda m: self._emit({"type": "log", "text": m}),
                llm_pair_fn=None,
                llm_pair_decisions=merged_llm,
            )
            self._emit_stage("grouping", 1, 1)

            pairs_est, max_b = hamming_matrix_stats(records)
            self._emit({"type": "log", "text": f"Дубликаты: групп {len(groups)}, файлов {len(records)}, пар ~{pairs_est}, max корзина {max_b}"})
            serial_groups = [{"paths": [str(x) for x in g.paths], "suggested_keep": str(g.suggested_keep), "is_exact": g.is_exact} for g in groups]
            serial_records = [{"path": str(r.path), "path_norm": r.path_norm, "size_bytes": r.size_bytes, "mtime_ns": r.mtime_ns, "width": r.width, "height": r.height, "sha256": r.sha256} for r in records]
            self._emit({"type": "dup_groups_ready", "groups": serial_groups, "records": serial_records, "records_count": len(records)})

            self.sig_db.upsert_session(
                session_key=session_key,
                root_path=str(root),
                media_mode=media_mode.value,
                strictness=options.strictness,
                stage="done",
                total_files=total,
                done_files=done,
                llm_total_pairs=0,
                llm_done_pairs=0,
                status="completed" if finish_reason == "completed" else finish_reason,
                payload={"paths": [str(x.path) for x in records], "groups": serial_groups},
            )
        except Exception as e:
            self._emit({"type": "log", "text": f"dup fatal: {e!s}"})
            finish_reason = "error"
        finally:
            elapsed = max(0.001, time.monotonic() - wall_t0)
            self._emit(
                {
                    "type": "metric",
                    "name": "duplicate_summary",
                    "payload": {
                        "elapsed_sec": round(elapsed, 2),
                        "files_per_sec": round(done / elapsed, 2) if "done" in locals() else 0.0,
                        "cache_hits": metrics["cache_hits"],
                        "sig_computed": metrics["sig_computed"],
                        "llm_pairs": metrics["llm_pairs"],
                    },
                }
            )
            if finish_reason == "stopped":
                self._emit({"type": "state_changed", "state": TaskState.STOPPED.value})
            elif finish_reason == "error":
                self._emit({"type": "state_changed", "state": TaskState.ERROR.value})
            else:
                self._emit({"type": "state_changed", "state": TaskState.FINISHED.value})
            self._emit({"type": "dup_stage", "stage": "done"})
            self._emit({"type": "dup_finished", "reason": finish_reason})

    def start_in_thread(
        self,
        root: Path,
        options: DuplicateFinderOptions,
        *,
        media_mode: MediaScanMode = MediaScanMode.PHOTOS_ONLY,
        force_recompute: bool = False,
        resume: bool = False,
    ) -> None:
        self._run_id = uuid.uuid4().hex

        def target() -> None:
            self.run_scan(root, options, media_mode=media_mode, force_recompute=force_recompute, resume=resume)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

"""CustomTkinter main window."""

from __future__ import annotations

import queue
import threading
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import json
from pathlib import Path

import customtkinter as ctk

from app.cache_service import CacheService
from app.category_aliases import load_category_aliases, save_category_aliases, normalize_aliases
from app.constants import CANONICAL_CATEGORIES, DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL, LOG_MAX_LINES, MediaScanMode
from app.db import Database
from app.gui_duplicates import DuplicatesPane
from app.model_profiles import ModelProfile, merge_profiles, profiles_to_settings
from app.settings_store import load_gui_settings, load_secret_settings, save_gui_settings, save_secret_settings
from app.signature_db import SignatureDatabase
from app.task_state import TaskState
from app.ui_texts import t
from app.lm_studio import (
    benchmark_models,
    find_model_object,
    full_api_self_test,
    list_models,
    vision_hint_from_model_dict,
    vision_self_test,
)
from app.video_frames import resolve_ffmpeg_executable
from app.worker import SortWorker


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    sec = int(round(seconds))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"~{h} ч {m} мин"
    if m > 0:
        return f"~{m} мин {s} с"
    return f"~{s} с"


def _section_label(parent: ctk.CTkFrame, text: str) -> None:
    ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(
        anchor="w", padx=4, pady=(4, 2)
    )


def _format_tag_list_for_display(categories: tuple[str, ...]) -> str:
    return "\n".join(categories)


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Photo AI Sorter")
        self.geometry("920x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._db = Database()
        self._sig_db = SignatureDatabase()
        self._cache_service = CacheService(self._db, self._sig_db)
        self._msg_queue: queue.Queue = queue.Queue()
        self._worker: SortWorker | None = None
        self._running = False
        self._current_run_id: str | None = None
        self._dup_pane: DuplicatesPane | None = None

        saved = load_gui_settings()
        secrets = load_secret_settings()
        self._loaded_settings: dict = saved if isinstance(saved, dict) else {}
        cache_settings = saved.get("cache_settings", {}) if isinstance(saved, dict) else {}
        self._cache_clear_on_start = bool(isinstance(cache_settings, dict) and cache_settings.get("clear_on_start", False))
        self._dup_force_recompute_default = bool(
            isinstance(cache_settings, dict) and cache_settings.get("dup_force_recompute_default", False)
        )
        self._in_var = ctk.StringVar(value=str(saved.get("input_dir", "") or ""))
        self._out_var = ctk.StringVar(value=str(saved.get("output_dir", "") or ""))
        self._api_var = ctk.StringVar(
            value=str(saved.get("api_base", "") or "").strip() or DEFAULT_API_BASE
        )
        self._api_key_var = ctk.StringVar(
            value=str(secrets.get("lm_studio_api_key", "") or DEFAULT_API_KEY).strip()
        )
        self._model_var = ctk.StringVar(value=t("lm.models.placeholder"))
        self._model_manual_var = ctk.StringVar(value=str(saved.get("model_manual", "") or ""))
        self._model_profiles = merge_profiles(
            saved.get("model_profiles", {}),
            api_base=self._api_var.get(),
            model=self._model_manual_var.get().strip() or DEFAULT_MODEL,
        )
        active_profile = str(saved.get("active_model_profile", "") or "classifier")
        if active_profile not in self._model_profiles:
            active_profile = "classifier"
        self._active_model_profile_var = ctk.StringVar(value=active_profile)
        mm = str(saved.get("media_mode", "") or MediaScanMode.PHOTOS_ONLY.value)
        if mm not in {m.value for m in MediaScanMode}:
            mm = MediaScanMode.PHOTOS_ONLY.value
        self._media_mode_var = ctk.StringVar(value=mm)
        self._sort_workers_var = ctk.StringVar(value=str(saved.get("sort_workers", 3) or 3))
        _saved_mode = str(saved.get("tag_mode", "") or "").strip()
        if _saved_mode not in {"strict", "free", "auto"}:
            _saved_free = bool(saved.get("free_tag_mode", False))
            _saved_mode = "free" if _saved_free else "strict"
        self._tag_mode_var = ctk.StringVar(value=_saved_mode)
        self._prompt_extra_var = ctk.StringVar(value=str(saved.get("prompt_extra", "") or ""))
        self._review_first_var = ctk.BooleanVar(value=bool(saved.get("review_first_sort", False)))

        self._build()
        if self._cache_clear_on_start:
            self._cache_service.clear_all()
            self._append_log("Кеш очищен при запуске (по настройке).")
        ctx = str(saved.get("user_context", "") or "")
        if ctx:
            self._context.delete("1.0", "end")
            self._context.insert("1.0", ctx)
        self._in_var.trace_add("write", lambda *_: self._update_start_state())
        self._out_var.trace_add("write", lambda *_: self._update_start_state())
        self._media_mode_var.trace_add("write", lambda *_: self.after(0, self._update_ffmpeg_hint))
        self._update_start_state()
        self._update_ffmpeg_hint()
        self.after(100, self._poll_queue)

    def _model_resolved(self) -> str:
        manual = self._model_manual_var.get().strip()
        if manual:
            return manual
        v = self._model_var.get().strip()
        if v.startswith("—") or v == "":
            return DEFAULT_MODEL
        return v

    def _active_profile(self) -> ModelProfile:
        name = self._active_model_profile_var.get().strip() or "classifier"
        return self._model_profiles.get(name) or self._model_profiles["classifier"]

    def _api_key_resolved(self) -> str:
        return self._api_key_var.get().strip() or DEFAULT_API_KEY

    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}

        self._tabs = ctk.CTkTabview(self)
        self._tabs.pack(fill="both", expand=True, padx=0, pady=0)
        tab_sort = self._tabs.add(t("tabs.sort"))
        tab_dup = self._tabs.add(t("tabs.duplicates"))
        P = tab_sort

        folders = ctk.CTkFrame(P, fg_color=("gray90", "gray16"), corner_radius=8)
        folders.pack(fill="x", **pad)
        _section_label(folders, t("folders.title"))
        row = ctk.CTkFrame(folders, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row, text=t("folders.input"), width=220, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=self._in_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text=t("folders.pick"), width=100, command=self._pick_in).pack(side="right")

        row2 = ctk.CTkFrame(folders, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(row2, text=t("folders.output"), width=220, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self._out_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row2, text=t("folders.pick"), width=100, command=self._pick_out).pack(side="right")

        row_mode = ctk.CTkFrame(folders, fg_color="transparent")
        row_mode.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(row_mode, text=t("folders.process"), width=220, anchor="w").pack(side="left")
        mode_btns = ctk.CTkFrame(row_mode, fg_color="transparent")
        mode_btns.pack(side="left", fill="x", expand=True)
        for val, label in (
            (MediaScanMode.PHOTOS_ONLY.value, "Только фото"),
            (MediaScanMode.PHOTOS_AND_VIDEO.value, "Фото + видео"),
            (MediaScanMode.VIDEO_ONLY.value, "Только видео"),
        ):
            ctk.CTkRadioButton(
                mode_btns,
                text=label,
                variable=self._media_mode_var,
                value=val,
            ).pack(side="left", padx=(0, 12))

        row_w = ctk.CTkFrame(folders, fg_color="transparent")
        row_w.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(row_w, text=t("folders.speed"), width=220, anchor="w").pack(side="left")
        ctk.CTkComboBox(row_w, values=["1", "2", "3", "4"], variable=self._sort_workers_var, width=90, state="readonly").pack(side="left")

        row_ff = ctk.CTkFrame(folders, fg_color="transparent")
        row_ff.pack(fill="x", padx=8, pady=(0, 4))
        self._ffmpeg_hint = ctk.CTkLabel(
            row_ff,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray35", "gray60"),
        )
        self._ffmpeg_hint.pack(anchor="w")

        _section_label(folders, t("folders.tag_mode.section"))
        row_tag_mode = ctk.CTkFrame(folders, fg_color="transparent")
        row_tag_mode.pack(fill="x", padx=8, pady=(0, 4))
        for val, label_key in (
            ("strict", "folders.tag_mode.strict_radio"),
            ("auto", "folders.tag_mode.auto_radio"),
            ("free", "folders.tag_mode.free_radio"),
        ):
            ctk.CTkRadioButton(
                row_tag_mode,
                text=t(label_key),
                variable=self._tag_mode_var,
                value=val,
            ).pack(side="left", padx=(0, 16))
        row_tag_btns = ctk.CTkFrame(folders, fg_color="transparent")
        row_tag_btns.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkButton(
            row_tag_btns,
            text=t("folders.tag_mode.show_list"),
            width=200,
            command=self._show_canonical_tags_dialog,
        ).pack(side="left")
        self._tag_mode_hint = ctk.CTkLabel(
            folders,
            text="",
            anchor="w",
            justify="left",
            text_color=("gray35", "gray65"),
            font=ctk.CTkFont(size=11),
        )
        self._tag_mode_hint.pack(fill="x", padx=8, pady=(0, 8))
        self._tag_mode_var.trace_add("write", lambda *_: self.after(0, self._update_tag_mode_hint))
        self._update_tag_mode_hint()

        lm = ctk.CTkFrame(P, fg_color=("gray90", "gray16"), corner_radius=8)
        lm.pack(fill="x", **pad)
        _section_label(lm, t("lm.title"))

        row_api = ctk.CTkFrame(lm, fg_color="transparent")
        row_api.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_api, text="Базовый URL:", width=120, anchor="w").pack(side="left")
        ctk.CTkEntry(row_api, textvariable=self._api_var, placeholder_text="http://host:port").pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        self._btn_refresh = ctk.CTkButton(row_api, text=t("lm.refresh_models"), width=200, command=self._on_refresh_models)
        self._btn_refresh.pack(side="right")

        row_key = ctk.CTkFrame(lm, fg_color="transparent")
        row_key.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_key, text="API key:", width=120, anchor="w").pack(side="left")
        ctk.CTkEntry(
            row_key,
            textvariable=self._api_key_var,
            show="*",
            placeholder_text="LM Studio token; хранится локально, не в git",
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(
            row_key,
            text="Сохранить ключ",
            width=150,
            command=self._save_lm_secret_clicked,
        ).pack(side="right")

        row_m = ctk.CTkFrame(lm, fg_color="transparent")
        row_m.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_m, text="Модель (список):", width=120, anchor="w").pack(side="left")
        self._model_combo = ctk.CTkComboBox(
            row_m,
            values=[t("lm.models.placeholder")],
            variable=self._model_var,
            width=420,
            state="readonly",
            command=lambda _v: self._on_model_combo_change(),
        )
        self._model_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))

        row_manual = ctk.CTkFrame(lm, fg_color="transparent")
        row_manual.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_manual, text="Вручную (приоритет):", width=120, anchor="w").pack(side="left")
        ctk.CTkEntry(
            row_manual,
            textvariable=self._model_manual_var,
            placeholder_text="Если пусто — берётся из списка выше",
        ).pack(side="left", fill="x", expand=True)

        row_profile = ctk.CTkFrame(lm, fg_color="transparent")
        row_profile.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_profile, text=t("lm.profile"), width=120, anchor="w").pack(side="left")
        self._profile_combo = ctk.CTkComboBox(
            row_profile,
            values=sorted(self._model_profiles.keys()),
            variable=self._active_model_profile_var,
            width=220,
            state="readonly",
            command=lambda _v: self._apply_model_profile(),
        )
        self._profile_combo.pack(side="left", padx=(0, 8))
        self._btn_profile_save = ctk.CTkButton(
            row_profile,
            text=t("lm.profile.save"),
            width=150,
            command=self._save_current_model_profile,
        )
        self._btn_profile_save.pack(side="left", padx=(0, 8))
        self._btn_benchmark = ctk.CTkButton(
            row_profile,
            text=t("lm.benchmark"),
            width=150,
            command=self._on_benchmark_models,
        )
        self._btn_benchmark.pack(side="left")

        row_vis = ctk.CTkFrame(lm, fg_color="transparent")
        row_vis.pack(fill="x", padx=8, pady=(0, 8))
        self._vision_status = ctk.CTkLabel(
            row_vis,
            text=t("lm.vision.unknown"),
            anchor="w",
            text_color=("gray30", "gray70"),
        )
        self._vision_status.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_vision = ctk.CTkButton(row_vis, text="Проверить vision (карта)", width=175, command=self._on_check_vision)
        self._btn_vision.pack(side="left", padx=(0, 8))
        self._btn_vision_file = ctk.CTkButton(
            row_vis, text="Тест с файлом…", width=140, command=self._on_vision_test_file
        )
        self._btn_vision_file.pack(side="left", padx=(0, 8))
        self._btn_selftest = ctk.CTkButton(row_vis, text="Самотест API", width=130, command=self._on_self_test)
        self._btn_selftest.pack(side="left")

        ctk.CTkLabel(P, text="Контекст для ИИ (USER_CONTEXT):", anchor="w").pack(anchor="w", padx=12)
        self._context = ctk.CTkTextbox(P, height=90)
        self._context.pack(fill="x", padx=12, pady=(0, 6))
        self._context.insert("1.0", "")

        prompt_box = ctk.CTkFrame(P, fg_color=("gray90", "gray16"), corner_radius=8)
        prompt_box.pack(fill="x", padx=12, pady=(0, 6))
        _section_label(prompt_box, t("prompt.title"))
        ctk.CTkLabel(
            prompt_box,
            text=t("prompt.extra"),
            anchor="w",
            text_color=("gray35", "gray65"),
        ).pack(anchor="w", padx=8)
        self._prompt_extra = ctk.CTkEntry(prompt_box, textvariable=self._prompt_extra_var)
        self._prompt_extra.pack(fill="x", padx=8, pady=(0, 8))
        prompt_btns = ctk.CTkFrame(prompt_box, fg_color="transparent")
        prompt_btns.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(
            prompt_btns,
            text=t("prompt.composer"),
            width=180,
            command=self._open_prompt_composer,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            prompt_btns,
            text=t("prompt.aliases"),
            width=190,
            command=self._open_category_aliases,
        ).pack(side="left")

        btns = ctk.CTkFrame(P, fg_color="transparent")
        btns.pack(fill="x", **pad)
        self._btn_start = ctk.CTkButton(btns, text=t("buttons.start"), command=self._on_start, state="disabled")
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_pause = ctk.CTkButton(btns, text=t("buttons.pause"), command=self._on_pause, state="disabled")
        self._btn_pause.pack(side="left", padx=(0, 8))
        self._btn_stop = ctk.CTkButton(btns, text=t("buttons.stop"), command=self._on_stop, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))
        self._btn_clear_cache = ctk.CTkButton(
            btns,
            text=t("buttons.clear_cache"),
            width=160,
            fg_color=("gray70", "gray35"),
            hover_color=("gray60", "gray45"),
            command=self._on_clear_hash_cache,
        )
        self._btn_clear_cache.pack(side="left")
        self._chk_review_first = ctk.CTkCheckBox(
            btns,
            text=t("sort.review_first"),
            variable=self._review_first_var,
        )
        self._chk_review_first.pack(side="left", padx=(12, 0))

        prog_row = ctk.CTkFrame(P, fg_color="transparent")
        prog_row.pack(fill="x", **pad)
        self._progress = ctk.CTkProgressBar(prog_row)
        self._progress.pack(fill="x", padx=12, pady=(8, 4))
        self._progress.set(0)
        self._prog_label = ctk.CTkLabel(prog_row, text="0 / 0")
        self._prog_label.pack(pady=(0, 2))
        self._eta_label = ctk.CTkLabel(
            prog_row,
            text=t("eta.left"),
            font=ctk.CTkFont(size=12),
            text_color=("gray35", "gray65"),
        )
        self._eta_label.pack(pady=(0, 8))
        self._health_label = ctk.CTkLabel(
            prog_row,
            text=t("health.idle"),
            font=ctk.CTkFont(size=11),
            text_color=("gray35", "gray65"),
        )
        self._health_label.pack(pady=(0, 8))

        ctk.CTkLabel(P, text=t("log.title"), anchor="w").pack(anchor="w", padx=12)
        self._log = ctk.CTkTextbox(P, height=200)
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._total_files = 0
        self._done_files = 0
        self._probe_busy = False
        self._channel_err_streak = 0
        self._channel_err_suppressed = 0

        self._dup_pane = DuplicatesPane(tab_dup, self)
        self._dup_pane.pack(fill="both", expand=True, padx=4, pady=4)

    def _update_tag_mode_hint(self) -> None:
        mode = self._tag_mode_var.get()
        if mode == "free":
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_free"))
        elif mode == "auto":
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_auto"))
        else:
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_strict"))

    def _show_canonical_tags_dialog(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title(t("folders.tags.dialog_title"))
        win.geometry("480x520")
        win.transient(self)
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(size=12))
        tb.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        tb.insert("1.0", _format_tag_list_for_display(CANONICAL_CATEGORIES))
        tb.configure(state="disabled")
        ctk.CTkButton(win, text=t("buttons.close"), command=win.destroy).pack(pady=(0, 12))

    def _on_model_combo_change(self) -> None:
        self._vision_status.configure(text=t("lm.vision.unknown"), text_color=("gray30", "gray70"))
        # If user explicitly picks model from dropdown, do not keep stale manual override.
        if self._model_manual_var.get().strip():
            self._model_manual_var.set("")
            self._append_log("Ручная модель очищена: используется выбранная из списка.")

    def _set_probe_busy(self, busy: bool) -> None:
        self._probe_busy = busy
        st = "disabled" if busy or self._running else "normal"
        self._btn_refresh.configure(state=st)
        self._btn_vision.configure(state=st)
        self._btn_vision_file.configure(state=st)
        self._btn_selftest.configure(state=st)
        self._btn_clear_cache.configure(state="disabled" if busy or self._running else "normal")
        if hasattr(self, "_btn_benchmark"):
            self._btn_benchmark.configure(state=st)
        if hasattr(self, "_btn_profile_save"):
            self._btn_profile_save.configure(state="disabled" if busy or self._running else "normal")
        if hasattr(self, "_chk_review_first"):
            self._chk_review_first.configure(state="disabled" if self._running else "normal")

    def _on_clear_hash_cache(self) -> None:
        if self._running:
            messagebox.showwarning(t("cache.warn_busy"), t("cache.warn_stop_first"))
            return
        if self._probe_busy:
            messagebox.showwarning(t("cache.warn_busy"), t("cache.warn_probe_wait"))
            return
        try:
            choice = messagebox.askyesnocancel(t("cache.ask.title"), t("cache.ask.text"))
            if choice is True:
                n = self._cache_service.clear_sort_cache()
                self._append_log(t("cache.log.cleared_sort", count=n))
            elif choice is False:
                n = self._cache_service.clear_duplicate_cache()
                self._append_log(t("cache.log.cleared_dup", count=n))
            else:
                # None = «Отмена» в диалоге: очистить оба кеша (см. cache.ask.text)
                a, b = self._cache_service.clear_all()
                self._append_log(t("cache.log.cleared_both", sort_count=a, dup_count=b))
        except OSError as e:
            messagebox.showerror(t("cache.error"), str(e))

    def _on_vision_test_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Файл для проверки vision",
            filetypes=[
                (
                    "Изображения и видео",
                    "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif *.avif "
                    "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.gif",
                ),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        base = self._api_var.get().strip() or DEFAULT_API_BASE
        model = self._model_resolved()

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            try:
                meta = None
                try:
                    meta = find_model_object(base, model, api_key=self._api_key_resolved()) if model else None
                except Exception:
                    meta = None
                hint = vision_hint_from_model_dict(meta) if isinstance(meta, dict) else None
                ok, detail = vision_self_test(
                    base,
                    model,
                    api_key=self._api_key_resolved(),
                    image_path=path,
                )

                def apply() -> None:
                    if hint is True:
                        hint_txt = t("vision.meta.yes")
                    elif hint is False:
                        hint_txt = t("vision.meta.no")
                    else:
                        hint_txt = t("vision.meta.none")
                    if ok:
                        self._vision_status.configure(
                            text=f"Vision: OK ({hint_txt})",
                            text_color=("green", "#6abf69"),
                        )
                        self._append_log(f"Vision OK (файл) [{hint_txt}]: {detail}")
                    else:
                        self._vision_status.configure(
                            text=t("lm.vision.error", hint=hint_txt),
                            text_color=("darkred", "#ff7b7b"),
                        )
                        self._append_log(f"Vision FAIL (файл) [{hint_txt}]: {detail}")
                    self._set_probe_busy(False)

                self.after(0, apply)
            except Exception as e:
                def fail() -> None:
                    self._vision_status.configure(text=t("lm.vision.error_short"), text_color=("darkred", "#ff7b7b"))
                    self._append_log(f"Vision (файл): {e!s}")
                    self._set_probe_busy(False)

                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()
        self._append_log(f"Проверка vision файлом «{path}», модель «{model}»...")

    def _update_ffmpeg_hint(self) -> None:
        try:
            mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            mode = MediaScanMode.PHOTOS_ONLY
        if mode == MediaScanMode.PHOTOS_ONLY:
            self._ffmpeg_hint.configure(text="")
            return
        ff = resolve_ffmpeg_executable()
        if ff:
            short = ff if len(ff) < 72 else "..." + ff[-68:]
            self._ffmpeg_hint.configure(
                text=f"ffmpeg: найден ({short}).",
                text_color=("gray30", "gray65"),
            )
        else:
            self._ffmpeg_hint.configure(
                text=(
                    "ffmpeg не найден — это не ошибка запуска. Видео пробует OpenCV; "
                    "для надёжности: установите ffmpeg, добавьте в PATH или задайте "
                    "PHOTO_AI_SORTER_FFMPEG=путь\\к\\ffmpeg.exe"
                ),
                text_color=("goldenrod", "#c9a227"),
            )

    def _save_gui_settings(self) -> None:
        try:
            dup: dict = {}
            if self._dup_pane is not None:
                dup = self._dup_pane.dup_settings_dict()
            save_gui_settings(
                {
                    "input_dir": self._in_var.get().strip(),
                    "output_dir": self._out_var.get().strip(),
                    "api_base": self._api_var.get().strip(),
                    "model_manual": self._model_manual_var.get().strip(),
                    "active_model_profile": self._active_model_profile_var.get().strip(),
                    "model_profiles": profiles_to_settings(self._model_profiles),
                    "media_mode": self._media_mode_var.get().strip(),
                    "sort_workers": self._sort_workers_var.get().strip(),
                    "tag_mode": self._tag_mode_var.get(),
                    "free_tag_mode": self._tag_mode_var.get() == "free",
                    "prompt_extra": self._prompt_extra_var.get().strip(),
                    "review_first_sort": bool(self._review_first_var.get()),
                    "user_context": self._context.get("1.0", "end").strip(),
                    "cache_settings": {
                        "clear_on_start": self._cache_clear_on_start,
                        "dup_force_recompute_default": self._dup_force_recompute_default,
                    },
                    "duplicate_finder": dup,
                }
            )
            save_secret_settings({"lm_studio_api_key": self._api_key_resolved()})
        except OSError:
            pass

    def _save_lm_secret_clicked(self) -> None:
        try:
            save_secret_settings({"lm_studio_api_key": self._api_key_resolved()})
            self._append_log("LM Studio API key сохранён локально в данных приложения (не в репозитории).")
        except OSError as e:
            messagebox.showerror(t("cache.error"), str(e))

    def _update_start_state(self) -> None:
        ok = bool(self._in_var.get().strip() and self._out_var.get().strip())
        if not self._running:
            self._btn_start.configure(state="normal" if ok else "disabled")
        self._set_probe_busy(self._probe_busy)

    def _apply_model_profile(self) -> None:
        p = self._active_profile()
        if p.api_base:
            self._api_var.set(p.api_base)
        if p.model:
            self._model_manual_var.set(p.model)
        self._sort_workers_var.set(str(p.workers))
        if p.prompt_extra:
            self._prompt_extra_var.set(p.prompt_extra)
        self._append_log(t("lm.profile.applied", name=p.name, model=p.model or DEFAULT_MODEL))

    def _save_current_model_profile(self) -> None:
        name = self._active_model_profile_var.get().strip() or "classifier"
        self._model_profiles[name] = ModelProfile(
            name=name,
            role=self._model_profiles.get(name, self._active_profile()).role,
            api_base=self._api_var.get().strip() or DEFAULT_API_BASE,
            model=self._model_resolved(),
            workers=max(1, min(4, int(self._sort_workers_var.get().strip() or "3"))),
            prompt_extra=self._prompt_extra_var.get().strip(),
        )
        self._profile_combo.configure(values=sorted(self._model_profiles.keys()))
        self._save_gui_settings()
        self._append_log(t("lm.profile.saved", name=name))

    def _on_benchmark_models(self) -> None:
        base = self._api_var.get().strip() or DEFAULT_API_BASE
        models = list(self._model_combo.cget("values") or [])
        models = [m for m in models if isinstance(m, str) and not m.startswith("—")]
        if not models:
            self._append_log(t("lm.benchmark.no_models"))
            return

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            try:
                rows = benchmark_models(
                    base,
                    models,
                    api_key=self._api_key_resolved(),
                    limit=8,
                    on_progress=lambda msg: self.after(0, lambda m=msg: self._append_log(m)),
                )
            except Exception as e:
                self.after(0, lambda: self._append_log(t("lm.benchmark.error", err=e)))
                self.after(0, lambda: self._set_probe_busy(False))
                return

            def apply() -> None:
                if not rows:
                    self._append_log(t("lm.benchmark.empty"))
                else:
                    best = rows[0]
                    self._model_manual_var.set(str(best["model"]))
                    self._append_log(
                        t(
                            "lm.benchmark.best",
                            model=best["model"],
                            latency=best["latency_sec"],
                            score=best["score"],
                        )
                    )
                    self._save_current_model_profile()
                self._set_probe_busy(False)

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()
        self._append_log(t("lm.benchmark.start", n=len(models)))

    def _open_prompt_composer(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title(t("prompt.composer.title"))
        win.geometry("720x560")
        win.transient(self)
        ctk.CTkLabel(win, text=t("prompt.composer.context"), anchor="w").pack(anchor="w", padx=12, pady=(12, 4))
        ctx = ctk.CTkTextbox(win, height=160)
        ctx.pack(fill="x", padx=12, pady=(0, 8))
        ctx.insert("1.0", self._context.get("1.0", "end").strip())
        ctk.CTkLabel(win, text=t("prompt.composer.extra"), anchor="w").pack(anchor="w", padx=12, pady=(4, 4))
        extra = ctk.CTkTextbox(win, height=220)
        extra.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        extra.insert("1.0", self._prompt_extra_var.get().strip())

        def apply() -> None:
            self._context.delete("1.0", "end")
            self._context.insert("1.0", ctx.get("1.0", "end").strip())
            self._prompt_extra_var.set(extra.get("1.0", "end").strip())
            self._save_current_model_profile()
            win.destroy()

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text=t("buttons.close"), command=win.destroy).pack(side="right")
        ctk.CTkButton(row, text=t("buttons.apply"), command=apply).pack(side="right", padx=(0, 8))

    def _open_category_aliases(self) -> None:
        aliases = load_category_aliases()
        win = ctk.CTkToplevel(self)
        win.title(t("prompt.aliases.title"))
        win.geometry("620x520")
        win.transient(self)
        ctk.CTkLabel(win, text=t("prompt.aliases.hint"), anchor="w", justify="left").pack(fill="x", padx=12, pady=(12, 6))
        tb = ctk.CTkTextbox(win, height=390)
        tb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        tb.insert("1.0", json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True))

        def save() -> None:
            try:
                raw = json.loads(tb.get("1.0", "end").strip() or "{}")
                save_category_aliases(normalize_aliases(raw))
            except (OSError, json.JSONDecodeError) as e:
                messagebox.showerror(t("cache.error"), str(e))
                return
            self._append_log(t("prompt.aliases.saved"))
            win.destroy()

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(row, text=t("buttons.close"), command=win.destroy).pack(side="right")
        ctk.CTkButton(row, text=t("buttons.save"), command=save).pack(side="right", padx=(0, 8))

    def _pick_in(self) -> None:
        d = filedialog.askdirectory(title="Папка с файлами для сортировки")
        if d:
            self._in_var.set(d)

    def _pick_out(self) -> None:
        d = filedialog.askdirectory(title="Папка для результата")
        if d:
            self._out_var.set(d)

    def _append_log(self, line: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", line + "\n")
        content = self._log.get("1.0", "end")
        lines = content.splitlines()
        if len(lines) > LOG_MAX_LINES:
            overflow = len(lines) - LOG_MAX_LINES
            self._log.delete("1.0", f"{overflow + 1}.0")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _flush_channel_error_summary(self) -> None:
        if self._channel_err_suppressed > 0:
            self._append_log(
                f"[channel] suppress similar channel errors: {self._channel_err_suppressed}"
            )
            self._channel_err_suppressed = 0
        self._channel_err_streak = 0

    def _set_controls_running(self, running: bool) -> None:
        self._running = running
        self._btn_start.configure(state="disabled" if running else "normal")
        self._btn_pause.configure(state="normal" if running else "disabled")
        self._btn_stop.configure(state="normal" if running else "disabled")
        if not running:
            self._btn_pause.configure(text=t("buttons.pause"))
        self._set_probe_busy(self._probe_busy)

    def _on_refresh_models(self) -> None:
        base = self._api_var.get().strip() or DEFAULT_API_BASE

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            try:
                models = list_models(base, api_key=self._api_key_resolved())
            except Exception as e:
                self.after(0, lambda: self._append_log(f"Список моделей: ошибка {e!s}"))
                self.after(0, lambda: self._set_probe_busy(False))
                return

            def apply() -> None:
                if models:
                    self._model_combo.configure(values=models)
                    self._model_var.set(models[0])
                else:
                    self._model_combo.configure(values=["— сервер вернул пустой список —"])
                    self._model_var.set(self._model_combo.cget("values")[0])
                self._vision_status.configure(text=t("lm.vision.unknown"), text_color=("gray30", "gray70"))
                self._append_log(f"Загружено моделей: {len(models)}")
                self._set_probe_busy(False)

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()
        self._append_log("Запрос списка моделей...")

    def _on_check_vision(self) -> None:
        base = self._api_var.get().strip() or DEFAULT_API_BASE
        model = self._model_resolved()

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            try:
                meta = None
                try:
                    meta = find_model_object(base, model, api_key=self._api_key_resolved()) if model else None
                except Exception:
                    meta = None
                hint = vision_hint_from_model_dict(meta) if isinstance(meta, dict) else None
                ok, detail = vision_self_test(base, model, api_key=self._api_key_resolved())

                def apply() -> None:
                    if hint is True:
                        hint_txt = t("vision.meta.yes")
                    elif hint is False:
                        hint_txt = t("vision.meta.no")
                    else:
                        hint_txt = t("vision.meta.none")
                    if ok:
                        self._vision_status.configure(
                            text=f"Vision: OK ({hint_txt})",
                            text_color=("green", "#6abf69"),
                        )
                        self._append_log(f"Vision OK [{hint_txt}]: {detail}")
                    else:
                        self._vision_status.configure(
                            text=t("lm.vision.error", hint=hint_txt),
                            text_color=("darkred", "#ff7b7b"),
                        )
                        self._append_log(f"Vision FAIL [{hint_txt}]: {detail}")
                    self._set_probe_busy(False)

                self.after(0, apply)
            except Exception as e:
                def fail() -> None:
                    self._vision_status.configure(text=t("lm.vision.error_short"), text_color=("darkred", "#ff7b7b"))
                    self._append_log(f"Vision: {e!s}")
                    self._set_probe_busy(False)

                self.after(0, fail)

        threading.Thread(target=work, daemon=True).start()
        self._append_log(f"Проверка vision (встроенная тест-карта 512x512), модель «{model}»...")

    def _on_self_test(self) -> None:
        base = self._api_var.get().strip() or DEFAULT_API_BASE
        model = self._model_resolved()

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            ok, report = full_api_self_test(base, model, api_key=self._api_key_resolved())

            def apply() -> None:
                self._append_log("——— Самотест ———")
                self._append_log(report)
                self._append_log("——— конец ———")
                if ok:
                    self._vision_status.configure(
                        text=t("lm.vision.selftest_ok"),
                        text_color=("green", "#6abf69"),
                    )
                else:
                    self._vision_status.configure(
                        text=t("lm.vision.selftest_error"),
                        text_color=("darkred", "#ff7b7b"),
                    )
                self._set_probe_busy(False)

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()
        self._append_log("Запуск самотеста API...")

    def _on_start(self) -> None:
        in_path = Path(self._in_var.get().strip())
        out_path = Path(self._out_var.get().strip())
        if not in_path.is_dir():
            self._append_log("Ошибка: укажите существующую папку для сканирования.")
            return
        if in_path.resolve() == out_path.resolve():
            self._append_log("Ошибка: папка результата не должна совпадать с папкой сканирования.")
            return
        if not out_path.exists():
            try:
                out_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self._append_log(f"Ошибка папки назначения: {e}")
                return
        if _path_is_relative_to(out_path, in_path):
            self._append_log(
                "Внимание: папка результата находится внутри источника; уже лежащие там файлы будут исключены из скана."
            )

        user_context = self._context.get("1.0", "end").strip()
        api_base = self._api_var.get().strip() or DEFAULT_API_BASE
        model = self._model_resolved()

        self._set_controls_running(True)
        self._done_files = 0
        self._total_files = 0
        self._progress.set(0)
        self._prog_label.configure(text="0 / …")
        self._eta_label.configure(text=t("eta.estimate_after_first"))
        try:
            media_mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            media_mode = MediaScanMode.PHOTOS_ONLY
        mode_labels = {
            MediaScanMode.PHOTOS_ONLY: "только фото",
            MediaScanMode.PHOTOS_AND_VIDEO: "фото + видео",
            MediaScanMode.VIDEO_ONLY: "только видео",
        }
        self._append_log(f"--- Старт (режим: {mode_labels.get(media_mode, media_mode.value)}) ---")
        tag_mode = self._tag_mode_var.get()
        if tag_mode == "strict":
            self._append_log(t("sort.log_mode_strict"))
        elif tag_mode == "auto":
            self._append_log(t("sort.log_mode_auto"))
        else:
            self._append_log(t("sort.log_mode_free"))
            self._append_log(t("sort.warn_large_library_free"))
        self._save_gui_settings()

        workers = max(1, min(4, int(self._sort_workers_var.get().strip() or "3")))
        profile = self._active_profile()
        aliases = load_category_aliases()
        self._worker = SortWorker(
            self._db,
            self._msg_queue,
            api_base=api_base,
            model=model,
            api_key=self._api_key_resolved(),
            workers=workers,
            free_tag_mode=tag_mode == "free",
            auto_tag_mode=tag_mode == "auto",
            structured_output=True,
            review_first=bool(self._review_first_var.get()),
            category_aliases=aliases,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_timeout_sec=profile.timeout_sec,
        )
        prompt_extra = self._prompt_extra_var.get().strip()
        self._worker.prompt_extra = prompt_extra
        self._worker.reset_stop()
        self._worker.set_paused(False)

        self._worker.start_in_thread(in_path, out_path, user_context, media_mode=media_mode)
        self._current_run_id = self._worker.run_id

    def _on_pause(self) -> None:
        if not self._worker:
            return
        paused = not self._worker.is_paused()
        self._worker.set_paused(paused)
        self._btn_pause.configure(text=t("buttons.resume") if paused else t("buttons.pause"))
        self._append_log("Пауза" if paused else "Продолжение")

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._append_log("Стоп запрошен (после текущего запроса)...")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_msg(self, msg: dict) -> None:
        run_id = msg.get("run_id")
        if self._running and self._current_run_id and run_id and run_id != self._current_run_id:
            return
        msg_type = msg.get("type")
        if msg_type == "scan_done":
            self._total_files = int(msg.get("total", 0))
            self._prog_label.configure(text=f"{self._done_files} / {self._total_files}")
            self._eta_label.configure(text="Осталось: —")
            if self._total_files >= 50_000 and self._tag_mode_var.get() != "strict":
                self._append_log(t("sort.warn_many_files_free", n=self._total_files))
        elif msg_type == "current":
            p = msg.get("path", "")
            self._append_log(f"Файл: {p}")
        elif msg_type == "log":
            text = str(msg.get("text", ""))
            low = text.lower()
            is_channel = "channel error" in low or "channel closed" in low
            if is_channel:
                self._channel_err_streak += 1
                if self._channel_err_streak <= 2:
                    self._append_log(text)
                else:
                    self._channel_err_suppressed += 1
            else:
                if self._channel_err_streak > 0:
                    self._flush_channel_error_summary()
                self._append_log(text)
        elif msg_type == "progress":
            self._done_files = int(msg.get("done", 0))
            tot = int(msg.get("total", self._total_files or 1))
            self._total_files = tot
            self._prog_label.configure(text=f"{self._done_files} / {tot}")
            if tot > 0:
                self._progress.set(min(1.0, self._done_files / float(tot)))
            eta_sec = float(msg.get("eta_sec", 0) or 0)
            if self._done_files > 0 and eta_sec > 0:
                self._eta_label.configure(text=t("eta.approx", eta=_format_eta(eta_sec)))
            elif self._done_files > 0 and tot > self._done_files:
                self._eta_label.configure(text=t("eta.counting"))
            else:
                self._eta_label.configure(text=t("eta.left"))
        elif msg_type == "finished":
            if self._channel_err_streak > 0:
                self._flush_channel_error_summary()
            reason = msg.get("reason", "")
            self._append_log(f"--- Готово ({reason}) ---")
            self._eta_label.configure(text=t("eta.left"))
            self._set_controls_running(False)
            self._worker = None
            self._current_run_id = None
            self._update_start_state()
            self._save_gui_settings()
        elif msg_type == "state_changed":
            state = str(msg.get("state", ""))
            if state == TaskState.PAUSED.value:
                self._btn_pause.configure(text=t("buttons.resume"))
            elif state == TaskState.RUNNING.value:
                self._btn_pause.configure(text=t("buttons.pause"))
        elif msg_type == "metric":
            name = str(msg.get("name", "metric"))
            payload = msg.get("payload", {})
            self._append_log(f"[metrics:{name}] {payload}")
        elif msg_type == "health":
            payload = msg.get("payload", {})
            self._health_label.configure(
                text=t(
                    "health.api",
                    calls=payload.get("api_calls", 0),
                    avg=payload.get("avg_api_sec", 0),
                    errors=payload.get("api_errors", 0),
                    review=payload.get("needs_review", 0),
                )
            )

    def on_close(self) -> None:
        if self._worker:
            self._worker.request_stop()
        if self._dup_pane is not None:
            self._dup_pane.on_app_close()
        else:
            self._sig_db.close()
        self._save_gui_settings()
        self._db.close()
        self.destroy()


def main() -> None:
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

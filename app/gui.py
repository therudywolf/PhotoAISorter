# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CustomTkinter main window."""

from __future__ import annotations

import queue
import threading
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import webbrowser
from pathlib import Path

import customtkinter as ctk

from app.cache_service import CacheService
from app.category_aliases import load_category_aliases
from app.context_tags import (
    build_custom_categories,
    build_user_context_from_tags,
    get_active_set,
    load_tag_store,
)
from app.tag_config import TagMode, ResolvedTagConfig, resolve_tag_config
from app.constants import (
    CANONICAL_CATEGORIES,
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    GENERAL_CATEGORIES,
    LOG_MAX_LINES,
    MediaScanMode,
    PIPELINE_VERSION,
)
from app.db import Database, make_sort_session_key
from app.gui_duplicates import DuplicatesPane
from app.model_profiles import ModelProfile, merge_profiles, profiles_to_settings
from app.settings_store import load_gui_settings, load_secret_settings, save_gui_settings, save_secret_settings
from app.signature_db import SignatureDatabase
from app.task_state import TaskState
from app.ui_texts import t
from app.lm_studio import (
    full_api_self_test,
    list_models,
    loaded_model_instances,
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


_MEDIA_MODE_LABELS = {
    MediaScanMode.PHOTOS_ONLY.value: "Фото",
    MediaScanMode.PHOTOS_AND_VIDEO.value: "Фото + видео",
    MediaScanMode.VIDEO_ONLY.value: "Видео + GIF",
}
_MEDIA_MODE_VALUES = {v: k for k, v in _MEDIA_MODE_LABELS.items()}

_TAG_MODE_LABELS = {
    "preset_sfw": "SFW",
    "preset_nsfw": "NSFW",
    "preset_furry_sfw": "Furry SFW",
    "preset_furry_nsfw": "Furry NSFW",
    "auto": "Авто",
    "free": "Свободно",
    "custom": "Свой список",
}
_TAG_MODE_VALUES = {v: k for k, v in _TAG_MODE_LABELS.items()}


def _parse_tag_mode_str(tag_mode_str: str) -> tuple:
    """Convert internal tag mode string to (TagMode, SearchProfile) pair."""
    from app.constants import SearchProfile
    if tag_mode_str == "preset_sfw":
        return (TagMode.PRESET, SearchProfile.SFW)
    elif tag_mode_str == "preset_nsfw":
        return (TagMode.PRESET, SearchProfile.NSFW)
    elif tag_mode_str == "preset_furry_sfw":
        return (TagMode.PRESET, SearchProfile.FURRY_SFW)
    elif tag_mode_str == "preset_furry_nsfw":
        return (TagMode.PRESET, SearchProfile.FURRY_NSFW)
    elif tag_mode_str == "auto":
        return (TagMode.AUTO, SearchProfile.SFW)
    elif tag_mode_str == "free":
        return (TagMode.FREE, SearchProfile.SFW)
    elif tag_mode_str == "custom":
        return (TagMode.CUSTOM, SearchProfile.SFW)
    else:
        return (TagMode.PRESET, SearchProfile.SFW)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Photo AI Sorter")
        self.geometry("920x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._db = Database()
        self._interrupted_sort_sessions = self._db.mark_running_sort_sessions_interrupted()
        self._sig_db = SignatureDatabase()
        self._cache_service = CacheService(self._db, self._sig_db)
        self._msg_queue: queue.Queue = queue.Queue()
        self._worker: SortWorker | None = None
        self._workers: dict[str, SortWorker] = {}
        self._run_labels: dict[str, str] = {}
        self._run_progress: dict[str, tuple[int, int]] = {}
        self._run_counter = 0
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
        self._media_mode_label_var = ctk.StringVar(
            value=_MEDIA_MODE_LABELS.get(mm, _MEDIA_MODE_LABELS[MediaScanMode.PHOTOS_ONLY.value])
        )
        _saved_mode = str(saved.get("tag_mode", "") or "").strip()
        _mode_migration = {
            "strict": "preset_furry_nsfw",
            "general": "preset_furry_nsfw",
        }
        _saved_mode = _mode_migration.get(_saved_mode, _saved_mode)
        if _saved_mode not in _TAG_MODE_LABELS:
            _saved_free = bool(saved.get("free_tag_mode", False))
            _saved_mode = "free" if _saved_free else "preset_sfw"
        self._tag_mode_var = ctk.StringVar(value=_saved_mode)
        self._tag_mode_label_var = ctk.StringVar(
            value=_TAG_MODE_LABELS.get(_saved_mode, _TAG_MODE_LABELS["preset_sfw"])
        )
        self._prompt_extra_var = ctk.StringVar(value=str(saved.get("prompt_extra", "") or ""))
        self._review_first_var = ctk.BooleanVar(value=bool(saved.get("review_first_sort", False)))

        self._build()
        if self._interrupted_sort_sessions:
            self._append_log(f"Найдено прерванных сессий сортировки: {self._interrupted_sort_sessions}.")
        if self._cache_clear_on_start:
            self._cache_service.clear_all()
            self._append_log("Кеш очищен при запуске (по настройке).")
        self._refresh_context_display()
        self._in_var.trace_add("write", lambda *_: self._update_start_state())
        self._out_var.trace_add("write", lambda *_: self._update_start_state())
        self._media_mode_var.trace_add("write", lambda *_: self.after(0, self._on_media_mode_value_changed))
        self._update_start_state()
        self._update_ffmpeg_hint()
        self._refresh_resume_sort_button()
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

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            header,
            text="Photo AI Sorter",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="About / License",
            width=140,
            fg_color=("gray75", "gray30"),
            command=self._show_about_dialog,
        ).pack(side="right")

        self._tabs = ctk.CTkTabview(self)
        self._tabs.pack(fill="both", expand=True, padx=0, pady=0)
        tab_sort = self._tabs.add(t("tabs.sort"))
        tab_dup = self._tabs.add(t("tabs.duplicates"))
        sort_scroll = ctk.CTkScrollableFrame(tab_sort, fg_color="transparent")
        sort_scroll.pack(fill="both", expand=True)
        P = sort_scroll

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
        ctk.CTkSegmentedButton(
            mode_btns,
            values=[_MEDIA_MODE_LABELS[m.value] for m in MediaScanMode],
            variable=self._media_mode_label_var,
            command=self._on_media_mode_label_change,
        ).pack(side="left", fill="x", expand=True)


        row_ff = ctk.CTkFrame(folders, fg_color="transparent")
        row_ff.pack(fill="x", padx=8, pady=(0, 4))
        self._ffmpeg_hint = ctk.CTkLabel(
            row_ff,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray35", "gray60"),
        )
        self._ffmpeg_hint.pack(side="left", fill="x", expand=True)

        _section_label(folders, t("folders.tag_mode.section"))
        row_tag_mode = ctk.CTkFrame(folders, fg_color="transparent")
        row_tag_mode.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkSegmentedButton(
            row_tag_mode,
            values=[_TAG_MODE_LABELS[k] for k in ("preset_sfw", "preset_nsfw", "preset_furry_sfw", "preset_furry_nsfw", "auto", "free", "custom")],
            variable=self._tag_mode_label_var,
            command=self._on_tag_mode_label_change,
        ).pack(fill="x", expand=True)
        row_tag_btns = ctk.CTkFrame(folders, fg_color="transparent")
        row_tag_btns.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkButton(
            row_tag_btns,
            text=t("folders.tag_mode.show_list"),
            width=150,
            command=self._show_canonical_tags_dialog,
        ).pack(side="left", padx=(0, 8))
        self._tag_mode_hint = ctk.CTkLabel(
            folders,
            text="",
            anchor="w",
            justify="left",
            wraplength=820,
            text_color=("gray35", "gray65"),
            font=ctk.CTkFont(size=11),
        )
        self._tag_mode_hint.pack(fill="x", padx=8, pady=(0, 8))
        self._tag_mode_var.trace_add("write", lambda *_: self.after(0, self._on_tag_mode_value_changed))
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
        ).pack(side="left", fill="x", expand=True)

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
            text="Профили…",
            width=120,
            command=self._open_profile_manager,
        )
        self._btn_profile_save.pack(side="left")

        row_vis = ctk.CTkFrame(lm, fg_color="transparent")
        row_vis.pack(fill="x", padx=8, pady=(0, 8))
        self._vision_status = ctk.CTkLabel(
            row_vis,
            text=t("lm.vision.unknown"),
            anchor="w",
            text_color=("gray30", "gray70"),
        )
        self._vision_status.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_selftest = ctk.CTkButton(row_vis, text="Проверить API", width=140, command=self._on_self_test)
        self._btn_selftest.pack(side="left", padx=(0, 8))
        self._btn_loaded_models = ctk.CTkButton(
            row_vis,
            text="Модели в памяти",
            width=145,
            fg_color=("gray75", "gray30"),
            command=self._on_loaded_models,
        )
        self._btn_loaded_models.pack(side="left")

        ctx_row = ctk.CTkFrame(P, fg_color="transparent")
        ctx_row.pack(fill="x", padx=12, pady=(0, 2))
        ctk.CTkLabel(ctx_row, text="Контекст для ИИ (USER_CONTEXT):", anchor="w").pack(side="left")
        ctk.CTkButton(
            ctx_row, text="Теги...", width=80,
            fg_color=("gray75", "gray30"),
            command=self._open_context_tags,
        ).pack(side="right")
        self._context = ctk.CTkTextbox(P, height=70, state="normal")
        self._context.pack(fill="x", padx=12, pady=(0, 6))
        self._context.insert("1.0", "")
        self._refresh_context_display()

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

        btns = ctk.CTkFrame(P, fg_color="transparent")
        btns.pack(fill="x", **pad)
        self._btn_start = ctk.CTkButton(btns, text=t("buttons.start"), width=130, command=self._on_start, state="disabled")
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_resume_sort = ctk.CTkButton(
            btns,
            text="Продолжить сессию",
            width=170,
            command=self._on_resume_last_sort,
            state="disabled",
        )
        self._btn_resume_sort.pack(side="left", padx=(0, 8))
        self._btn_pause = ctk.CTkButton(btns, text=t("buttons.pause"), command=self._on_pause, state="disabled")
        self._btn_pause.pack(side="left", padx=(0, 8))
        self._btn_stop = ctk.CTkButton(btns, text=t("buttons.stop"), command=self._on_stop, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))

        sort_options = ctk.CTkFrame(P, fg_color="transparent")
        sort_options.pack(fill="x", padx=12, pady=(0, 6))
        self._chk_review_first = ctk.CTkCheckBox(
            sort_options,
            text=t("sort.review_first"),
            variable=self._review_first_var,
        )
        self._chk_review_first.pack(side="left", padx=(0, 12))
        ctk.CTkLabel(
            sort_options,
            text=t("sort.review_first.hint"),
            anchor="w",
            text_color=("gray35", "gray65"),
            font=ctk.CTkFont(size=11),
        ).pack(side="left", fill="x", expand=True)
        self._btn_clear_cache = ctk.CTkButton(
            sort_options,
            text=t("buttons.clear_cache"),
            width=120,
            fg_color=("gray70", "gray35"),
            hover_color=("gray60", "gray45"),
            command=self._on_clear_hash_cache,
        )
        self._btn_clear_cache.pack(side="right")

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

        dup_scroll = ctk.CTkScrollableFrame(tab_dup, fg_color="transparent")
        dup_scroll.pack(fill="both", expand=True)
        self._dup_pane = DuplicatesPane(dup_scroll, self)
        self._dup_pane.pack(fill="both", expand=True, padx=4, pady=4)

    def _show_about_dialog(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("About / License")
        win.geometry("620x430")
        win.transient(self)
        text = (
            "Photo AI Sorter\n\n"
            "Copyright (C) 2026 Photo AI Sorter contributors.\n\n"
            "License: GNU Affero General Public License v3.0 only.\n\n"
            "This program is free software: you can redistribute it and/or modify it under "
            "the terms of the GNU Affero General Public License as published by the Free "
            "Software Foundation, version 3.\n\n"
            "This program is distributed WITHOUT ANY WARRANTY; without even the implied "
            "warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n"
            "Source code: https://github.com/therudywolf/PhotoAISorter\n"
            "Full license text is included in the LICENSE file."
        )
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(size=13), wrap="word")
        tb.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        tb.insert("1.0", text)
        tb.configure(state="disabled")
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(
            row,
            text="Open Source",
            width=130,
            command=lambda: webbrowser.open("https://github.com/therudywolf/PhotoAISorter"),
        ).pack(side="left")
        ctk.CTkButton(row, text=t("buttons.close"), width=120, command=win.destroy).pack(side="right")

    def _on_media_mode_label_change(self, label: str) -> None:
        value = _MEDIA_MODE_VALUES.get(label)
        if value:
            self._media_mode_var.set(value)

    def _on_media_mode_value_changed(self) -> None:
        label = _MEDIA_MODE_LABELS.get(
            self._media_mode_var.get(),
            _MEDIA_MODE_LABELS[MediaScanMode.PHOTOS_ONLY.value],
        )
        if self._media_mode_label_var.get() != label:
            self._media_mode_label_var.set(label)
        self._update_ffmpeg_hint()

    def _on_tag_mode_label_change(self, label: str) -> None:
        value = _TAG_MODE_VALUES.get(label)
        if value:
            self._tag_mode_var.set(value)

    def _on_tag_mode_value_changed(self) -> None:
        label = _TAG_MODE_LABELS.get(self._tag_mode_var.get(), _TAG_MODE_LABELS["preset_sfw"])
        if self._tag_mode_label_var.get() != label:
            self._tag_mode_label_var.set(label)
        self._update_tag_mode_hint()

    def _update_tag_mode_hint(self) -> None:
        mode = self._tag_mode_var.get()
        if mode == "free":
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_free"))
        elif mode == "auto":
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_auto"))
        elif mode == "custom":
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_custom"))
        else:
            self._tag_mode_hint.configure(text=t("folders.tag_mode.hint_preset"))

    def _show_canonical_tags_dialog(self) -> None:
        win = ctk.CTkToplevel(self)
        mode = self._tag_mode_var.get()
        if mode == "custom":
            win.title("Свой список категорий")
            _store = load_tag_store()
            _aset = get_active_set(_store)
            categories = build_custom_categories(_aset) if _aset else ()
        elif mode in ("auto", "free"):
            win.title(t("folders.tags.dialog_title_reference"))
            categories = GENERAL_CATEGORIES
        else:
            win.title(t("folders.tags.dialog_title_preset"))
            _mode, _profile = _parse_tag_mode_str(mode)
            from app.constants import categories_for_profile
            categories = categories_for_profile(_profile)
        win.geometry("480x520")
        win.transient(self)
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(size=12))
        tb.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        tb.insert("1.0", _format_tag_list_for_display(categories))
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
        self._btn_selftest.configure(state=st)
        self._btn_clear_cache.configure(state="disabled" if busy or self._running else "normal")
        if hasattr(self, "_btn_loaded_models"):
            self._btn_loaded_models.configure(state="disabled" if busy else "normal")
        if hasattr(self, "_btn_profile_save"):
            self._btn_profile_save.configure(state="disabled" if busy or self._running else "normal")
        if hasattr(self, "_chk_review_first"):
            self._chk_review_first.configure(state="disabled" if self._running else "normal")
        if hasattr(self, "_btn_resume_sort"):
            if busy or self._running or self._latest_sort_session() is None:
                self._btn_resume_sort.configure(state="disabled")
            else:
                self._btn_resume_sort.configure(state="normal")

    def _on_clear_hash_cache(self) -> None:
        if self._running:
            messagebox.showwarning(t("cache.warn_busy"), t("cache.warn_stop_first"))
            return
        if self._probe_busy:
            messagebox.showwarning(t("cache.warn_busy"), t("cache.warn_probe_wait"))
            return

        win = ctk.CTkToplevel(self)
        win.title(t("cache.ask.title"))
        win.geometry("420x220")
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Выберите, что очистить:", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=16, pady=(16, 12))

        def do_sort() -> None:
            win.destroy()
            try:
                n = self._cache_service.clear_sort_cache()
                self._append_log(t("cache.log.cleared_sort", count=n))
            except OSError as e:
                messagebox.showerror(t("cache.error"), str(e))

        def do_dup() -> None:
            win.destroy()
            try:
                n = self._cache_service.clear_duplicate_cache()
                self._append_log(t("cache.log.cleared_dup", count=n))
            except OSError as e:
                messagebox.showerror(t("cache.error"), str(e))

        def do_both() -> None:
            win.destroy()
            try:
                ns = self._cache_service.clear_sort_cache()
                nd = self._cache_service.clear_duplicate_cache()
                self._append_log(t("cache.log.cleared_both", sort_count=ns, dup_count=nd))
            except OSError as e:
                messagebox.showerror(t("cache.error"), str(e))

        btns = ctk.CTkFrame(win, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(btns, text="Кеш сортировки", width=180, command=do_sort).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(btns, text="Кеш дубликатов", width=180, command=do_dup).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(btns, text="Оба кеша", width=180, fg_color="darkred", command=do_both).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(win, text="Отмена", width=120, fg_color=("gray70", "gray35"), command=win.destroy).pack(pady=(0, 12))

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
                    "tag_mode": self._tag_mode_var.get(),
                    "free_tag_mode": self._tag_mode_var.get() == "free",
                    "prompt_extra": self._prompt_extra_var.get().strip(),
                    "review_first_sort": bool(self._review_first_var.get()),
                    "user_context": build_user_context_from_tags(get_active_set(load_tag_store())),
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

    def _update_start_state(self) -> None:
        ok = bool(self._in_var.get().strip() and self._out_var.get().strip())
        if not self._running:
            self._btn_start.configure(state="normal" if ok else "disabled")
            if hasattr(self, "_btn_resume_sort"):
                self._refresh_resume_sort_button()
        self._set_probe_busy(self._probe_busy)

    def _latest_sort_session(self):
        return self._db.latest_incomplete_sort_session()

    def _sort_session_key_for_current(self, source_dir: Path, dest_dir: Path, media_mode: MediaScanMode, tag_mode: str) -> str:
        return make_sort_session_key(
            str(source_dir.resolve()),
            str(dest_dir.resolve()),
            media_mode.value,
            tag_mode,
            bool(self._review_first_var.get()),
            PIPELINE_VERSION,
        )

    def _refresh_resume_sort_button(self) -> None:
        if not hasattr(self, "_btn_resume_sort"):
            return
        row = self._latest_sort_session()
        if row is None or self._running:
            self._btn_resume_sort.configure(state="disabled")
            return
        self._btn_resume_sort.configure(state="normal")

    def _sort_session_summary(self, row) -> str:
        done = int(row["done_files"] or 0)
        total = int(row["total_files"] or 0)
        status = str(row["status"] or "unknown")
        src = str(row["source_dir"] or "")
        return f"{done}/{total} файлов, статус {status}\n{src}"

    def _apply_sort_session_to_fields(self, row) -> None:
        self._in_var.set(str(row["source_dir"] or ""))
        self._out_var.set(str(row["dest_dir"] or ""))
        mm = str(row["media_mode"] or MediaScanMode.PHOTOS_ONLY.value)
        if mm in {m.value for m in MediaScanMode}:
            self._media_mode_var.set(mm)
        tag_mode = str(row["tag_mode"] or "preset_sfw")
        _mode_migration = {"strict": "preset_furry_nsfw", "general": "preset_furry_nsfw"}
        tag_mode = _mode_migration.get(tag_mode, tag_mode)
        if tag_mode in _TAG_MODE_LABELS:
            self._tag_mode_var.set(tag_mode)
        self._review_first_var.set(bool(int(row["review_first"] or 0)))

    def _ask_sort_resume_mode(self, session_key: str) -> tuple[bool, bool]:
        row = self._db.get_sort_session(session_key)
        if row is None:
            return False, False
        total = int(row["total_files"] or 0)
        done = int(row["done_files"] or 0)
        status = str(row["status"] or "")
        if total <= 0 or (status == "completed" and done >= total):
            return False, False
        res = messagebox.askyesnocancel(
            "Найден сохранённый прогресс",
            (
                "Для этих параметров сортировки уже есть сессия:\n\n"
                f"{self._sort_session_summary(row)}\n\n"
                "Да — продолжить\n"
                "Нет — начать новую сессию (общий SHA-кеш всё равно сохранится)\n"
                "Отмена — не запускать"
            ),
        )
        if res is None:
            return True, False
        if res:
            return False, True
        self._db.clear_sort_session(session_key)
        return False, False

    def _on_resume_last_sort(self) -> None:
        row = self._latest_sort_session()
        if row is None:
            self._append_log("Нет сохранённых сессий сортировки для продолжения.")
            self._refresh_resume_sort_button()
            return
        self._apply_sort_session_to_fields(row)
        self._append_log(f"Продолжение сессии сортировки: {self._sort_session_summary(row)}")
        self._on_start(force_resume=True)

    def _apply_model_profile(self) -> None:
        p = self._active_profile()
        if p.api_base:
            self._api_var.set(p.api_base)
        if p.model:
            self._model_manual_var.set(p.model)
        if p.prompt_extra:
            self._prompt_extra_var.set(p.prompt_extra)
        self._append_log(t("lm.profile.applied", name=p.name, model=p.model or DEFAULT_MODEL))

    def _save_current_model_profile(self) -> None:
        name = self._active_model_profile_var.get().strip() or "classifier"
        current = self._model_profiles.get(name, self._active_profile())
        self._model_profiles[name] = ModelProfile(
            name=name,
            role=current.role,
            api_base=self._api_var.get().strip() or DEFAULT_API_BASE,
            model=self._model_resolved(),
            temperature=current.temperature,
            max_tokens=current.max_tokens,
            timeout_sec=current.timeout_sec,
            workers=3,
            api_workers=1,
            prompt_extra=self._prompt_extra_var.get().strip(),
        )
        self._profile_combo.configure(values=sorted(self._model_profiles.keys()))
        self._save_gui_settings()
        self._append_log(t("lm.profile.saved", name=name))

    def _open_profile_manager(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Профили моделей")
        win.geometry("680x460")
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Сохранённые профили", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=12, pady=(12, 6))

        tb = ctk.CTkTextbox(win, height=260, font=ctk.CTkFont(family="Consolas", size=12))
        tb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        def refresh_list() -> None:
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            if not self._model_profiles:
                tb.insert("1.0", "Нет сохранённых профилей.")
            else:
                active = self._active_model_profile_var.get().strip()
                for name in sorted(self._model_profiles):
                    p = self._model_profiles[name]
                    marker = "  ◀ активный" if name == active else ""
                    tb.insert("end", f"[{name}]{marker}\n")
                    tb.insert("end", f"  Модель:      {p.model or DEFAULT_MODEL}\n")
                    tb.insert("end", f"  API base:    {p.api_base or DEFAULT_API_BASE}\n")
                    tb.insert("end", f"  Temperature: {p.temperature}  Max tokens: {p.max_tokens}  Timeout: {p.timeout_sec}s\n")
                    if p.prompt_extra:
                        short = p.prompt_extra[:80] + ("..." if len(p.prompt_extra) > 80 else "")
                        tb.insert("end", f"  Промпт:      {short}\n")
                    tb.insert("end", "\n")
            tb.configure(state="disabled")

        refresh_list()

        row_save = ctk.CTkFrame(win, fg_color="transparent")
        row_save.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(row_save, text="Сохранить текущие настройки как:", anchor="w").pack(side="left", padx=(0, 8))
        name_var = ctk.StringVar(value=self._active_model_profile_var.get().strip() or "classifier")
        ctk.CTkEntry(row_save, textvariable=name_var, width=200).pack(side="left", padx=(0, 8))

        def save() -> None:
            self._active_model_profile_var.set(name_var.get().strip() or "classifier")
            self._save_current_model_profile()
            refresh_list()

        ctk.CTkButton(row_save, text="Сохранить", width=120, command=save).pack(side="left")

        ctk.CTkButton(win, text="Закрыть", width=120, command=win.destroy).pack(pady=(0, 12))

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

    def _format_loaded_models(self, rows: list[dict[str, object]]) -> str:
        if not rows:
            return "LM Studio: загруженных моделей не найдено."
        parts: list[str] = [f"LM Studio: загружено инстансов: {len(rows)}"]
        by_key: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_key.setdefault(str(row.get("model_key") or ""), []).append(row)
        for key, items in sorted(by_key.items()):
            dup = " duplicate" if len(items) > 1 else ""
            parts.append(f"{key}: {len(items)}{dup}")
            for item in items:
                ttl = item.get("remaining_ttl_seconds")
                ttl_txt = f", ttl={int(ttl)}s" if isinstance(ttl, (int, float)) else ""
                parts.append(
                    "  "
                    f"{item.get('instance_id')} "
                    f"(ctx={item.get('context_length')}, parallel={item.get('parallel')}{ttl_txt})"
                )
        return "\n".join(parts)

    def _refresh_context_display(self) -> None:
        store = load_tag_store()
        active = get_active_set(store)
        text = build_user_context_from_tags(active)
        self._context.configure(state="normal")
        self._context.delete("1.0", "end")
        if active and text:
            self._context.insert("1.0", f"[{active.name}]\n{text}")
        elif active:
            self._context.insert("1.0", f"[{active.name}] — тегов: {len(active.tags)} (без описаний)")
        else:
            self._context.insert("1.0", "(нет активного набора — откройте «Теги...»)")
        self._context.configure(state="disabled")

    def _open_context_tags(self) -> None:
        from app.gui_context_tags import TagSetsDialog
        TagSetsDialog(self, on_save=self._refresh_context_display)

    def _on_loaded_models(self) -> None:
        base = self._api_var.get().strip() or DEFAULT_API_BASE

        def work() -> None:
            self.after(0, lambda: self._set_probe_busy(True))
            try:
                rows = loaded_model_instances(base, api_key=self._api_key_resolved())
                text = self._format_loaded_models(rows)
            except Exception as e:
                text = f"LM Studio loaded models: ошибка {e!s}"
            self.after(0, lambda: self._append_log(text))
            self.after(0, lambda: self._set_probe_busy(False))

        threading.Thread(target=work, daemon=True).start()
        self._append_log("Проверяю загруженные модели LM Studio...")

    def _on_start(self, *, force_resume: bool = False) -> None:
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

        _tag_store = load_tag_store()
        api_base = self._api_var.get().strip() or DEFAULT_API_BASE
        model = self._model_resolved()

        try:
            media_mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            media_mode = MediaScanMode.PHOTOS_ONLY
        tag_mode_str = self._tag_mode_var.get()
        tag_mode, search_profile = _parse_tag_mode_str(tag_mode_str)

        from app.constants import SearchProfile
        tag_cfg = resolve_tag_config(tag_mode, profile=search_profile, tag_store=_tag_store)

        if tag_mode == TagMode.CUSTOM and not tag_cfg.categories:
            self._append_log("Ошибка: нет активного набора тегов или он пуст. Откройте «Теги...» и настройте.")
            return

        user_context = tag_cfg.user_context
        session_key = self._sort_session_key_for_current(in_path, out_path, media_mode, tag_mode_str)
        resume_session = bool(force_resume)
        if not force_resume:
            cancelled, resume_session = self._ask_sort_resume_mode(session_key)
            if cancelled:
                return

        self._set_controls_running(True)
        self._done_files = 0
        self._total_files = 0
        self._workers.clear()
        self._run_labels.clear()
        self._run_progress.clear()
        self._run_counter = 0
        self._progress.set(0)
        self._prog_label.configure(text="0 / …")
        self._eta_label.configure(text=t("eta.estimate_after_first"))
        mode_labels = {
            MediaScanMode.PHOTOS_ONLY: "только фото",
            MediaScanMode.PHOTOS_AND_VIDEO: "фото + видео",
            MediaScanMode.VIDEO_ONLY: "только видео",
        }
        self._append_log(f"--- Старт (режим: {mode_labels.get(media_mode, media_mode.value)}) ---")
        if resume_session:
            self._append_log("Сессия: продолжение сохранённого прогресса.")
        else:
            self._append_log("Сессия: новая или без найденного незавершённого прогресса.")
        if tag_mode == TagMode.PRESET:
            self._append_log(t("sort.log_mode_preset", profile=search_profile.value))
        elif tag_mode == TagMode.AUTO:
            self._append_log(t("sort.log_mode_auto"))
        elif tag_mode == TagMode.CUSTOM:
            self._append_log("Режим папок: пользовательский список категорий.")
        else:
            self._append_log(t("sort.log_mode_free"))
            self._append_log(t("sort.warn_large_library_free"))
        self._save_gui_settings()

        workers = 3
        api_workers = 1
        profile = self._active_profile()
        aliases = load_category_aliases()

        self._worker = SortWorker(
            self._db,
            self._msg_queue,
            api_base=api_base,
            model=model,
            api_key=self._api_key_resolved(),
            workers=workers,
            api_workers=api_workers,
            tag_config=tag_cfg,
            structured_output=True,
            review_first=bool(self._review_first_var.get()),
            category_aliases=aliases,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_timeout_sec=profile.timeout_sec,
            session_key=session_key,
            resume_session=resume_session,
        )
        prompt_extra = self._prompt_extra_var.get().strip()
        self._worker.prompt_extra = prompt_extra
        self._worker.reset_stop()
        self._worker.set_paused(False)

        self._worker.start_in_thread(
            in_path,
            out_path,
            user_context,
            media_mode=media_mode,
            session_key=session_key,
            resume_session=resume_session,
        )
        self._current_run_id = self._worker.run_id
        if self._worker.run_id:
            self._workers[self._worker.run_id] = self._worker
            self._run_labels[self._worker.run_id] = "main"

    def _on_pause(self) -> None:
        workers = [w for w in self._workers.values() if w.is_alive()]
        if not workers:
            return
        paused = not all(w.is_paused() for w in workers)
        for worker in workers:
            worker.set_paused(paused)
        self._btn_pause.configure(text=t("buttons.resume") if paused else t("buttons.pause"))
        self._append_log(("Пауза" if paused else "Продолжение") + f": {len(workers)} sort worker(s)")

    def _on_stop(self) -> None:
        workers = [w for w in self._workers.values() if w.is_alive()]
        if workers:
            for worker in workers:
                worker.request_stop()
            self._append_log(
                f"Стоп запрошен для {len(workers)} sort worker(s): прогресс сохранится после текущего файла/запроса..."
            )

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _run_log_prefix(self, run_id: object) -> str:
        rid = str(run_id or "")
        label = self._run_labels.get(rid, "")
        if not label:
            return ""
        if label != "main" or len(self._run_labels) > 1:
            return f"[{label}] "
        return ""

    def _refresh_aggregate_progress(self) -> None:
        if not self._run_progress:
            return
        done = sum(v[0] for v in self._run_progress.values())
        total = sum(v[1] for v in self._run_progress.values())
        self._done_files = done
        self._total_files = total
        self._prog_label.configure(text=f"{done} / {total}")
        if total > 0:
            self._progress.set(min(1.0, done / float(total)))

    def _handle_msg(self, msg: dict) -> None:
        run_id = msg.get("run_id")
        prefix = self._run_log_prefix(run_id)
        msg_type = msg.get("type")
        if msg_type == "scan_done":
            total = int(msg.get("total", 0))
            if run_id:
                self._run_progress[str(run_id)] = (0, total)
                self._refresh_aggregate_progress()
            else:
                self._total_files = total
                self._prog_label.configure(text=f"{self._done_files} / {self._total_files}")
            self._eta_label.configure(text="Осталось: —")
            if self._total_files >= 50_000 and self._tag_mode_var.get() in {"auto", "free"}:
                self._append_log(t("sort.warn_many_files_free", n=self._total_files))
        elif msg_type == "current":
            p = msg.get("path", "")
            self._append_log(f"{prefix}Файл: {p}")
        elif msg_type == "log":
            text = prefix + str(msg.get("text", ""))
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
            done = int(msg.get("done", 0))
            tot = int(msg.get("total", self._total_files or 1))
            if run_id:
                self._run_progress[str(run_id)] = (done, tot)
                self._refresh_aggregate_progress()
            else:
                self._done_files = done
                self._total_files = tot
                self._prog_label.configure(text=f"{self._done_files} / {tot}")
                if tot > 0:
                    self._progress.set(min(1.0, self._done_files / float(tot)))
            eta_sec = float(msg.get("eta_sec", 0) or 0)
            if len(self._run_progress) > 1:
                self._eta_label.configure(text=t("eta.counting"))
            elif self._done_files > 0 and eta_sec > 0:
                self._eta_label.configure(text=t("eta.approx", eta=_format_eta(eta_sec)))
            elif self._done_files > 0 and tot > self._done_files:
                self._eta_label.configure(text=t("eta.counting"))
            else:
                self._eta_label.configure(text=t("eta.left"))
        elif msg_type == "finished":
            if self._channel_err_streak > 0:
                self._flush_channel_error_summary()
            reason = msg.get("reason", "")
            self._append_log(f"{prefix}--- Готово ({reason}) ---")
            if run_id:
                self._workers.pop(str(run_id), None)
            remaining = [w for w in self._workers.values() if w.is_alive()]
            if not remaining:
                self._eta_label.configure(text=t("eta.left"))
                self._set_controls_running(False)
                self._worker = None
                self._current_run_id = None
                self._update_start_state()
                self._refresh_resume_sort_button()
                self._save_gui_settings()
            else:
                self._worker = remaining[-1]
                self._current_run_id = self._worker.run_id
        elif msg_type == "session":
            status = str(msg.get("status", ""))
            done = int(msg.get("done", 0) or 0)
            total = int(msg.get("total", 0) or 0)
            if status and status != "running":
                self._append_log(f"{prefix}Сессия сортировки сохранена: {status}, {done}/{total}.")
        elif msg_type == "state_changed":
            state = str(msg.get("state", ""))
            if state == TaskState.PAUSED.value:
                self._btn_pause.configure(text=t("buttons.resume"))
            elif state == TaskState.RUNNING.value:
                self._btn_pause.configure(text=t("buttons.pause"))
        elif msg_type == "metric":
            name = str(msg.get("name", "metric"))
            payload = msg.get("payload", {})
            self._append_log(f"{prefix}[metrics:{name}] {payload}")
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
        workers = [w for w in self._workers.values() if w.is_alive()]
        if self._worker and self._worker not in workers:
            workers.append(self._worker)
        for worker in workers:
            worker.request_stop()
        for worker in workers:
            worker.join(5.0)
        if self._dup_pane is not None:
            self._dup_pane.on_app_close()
        else:
            self._sig_db.close()
        self._save_gui_settings()
        try:
            self._db.close()
        except Exception:
            pass
        self.destroy()


def main() -> None:
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

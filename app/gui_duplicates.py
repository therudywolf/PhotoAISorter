"""Duplicates finder tab: simple strictness UI, staged progress, resume support."""

from __future__ import annotations

import os
import queue
import threading
import tkinter.messagebox as messagebox
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.constants import DEFAULT_API_BASE, DEFAULT_MODEL, MediaScanMode, VIDEO_EXTENSIONS
from app.duplicate_finder import (
    DuplicateFinderOptions,
    load_records_from_db,
    merge_options_from_dict,
    options_from_preset,
    options_to_dict,
    regroup_from_cached_records,
)
from app.dup_export import export_duplicate_run
from app.dup_ui_constants import CTK_ACCENT_PRIMARY, CTK_ACCENT_PRIMARY_HOVER
from app.dup_thumbs import thumb_ctk as _thumb_ctk
from app.gui_dup_groups_window import DuplicateGroupsViewer
from app.duplicate_worker import DuplicateFinderWorker
from app.settings_store import duplicate_journal_path
from app.signature_db import SignatureDatabase, make_session_key
from app.lm_studio import list_models
from app.ui_texts import t as ui_t
from app.video_frames import resolve_ffmpeg_executable

if TYPE_CHECKING:
    from app.gui import App

ACCURACY_PRESET_KEYS: tuple[str, ...] = ("fast", "balanced", "strict", "deep")

# Large duplicate groups: cap thumbnails per card and paginate the rest in a modal.
DUP_CARD_MODAL_PAGE = 200
PERSIST_DEBOUNCE_MS = 350
# При большем числе файлов удаление выполняется в фоновом потоке, UI обновляется через очередь.
DELETE_PROGRESS_THREAD_THRESHOLD = 500

def _dup_stage_label(stage: str) -> str:
    return {
        "scan_signatures": ui_t("dup.stage.scan"),
        "grouping": ui_t("dup.stage.grouping"),
        "llm_verify": ui_t("dup.stage.llm"),
        "done": ui_t("dup.stage.done"),
        "start": ui_t("dup.stage.start"),
    }.get(stage, ui_t("dup.stage.generic", name=stage))


FILTER_TYPE_KEY_TO_LABEL = {
    "all": ui_t("dup.filter.type.all"),
    "exact": ui_t("dup.filter.type.exact"),
    "similar": ui_t("dup.filter.type.similar"),
}
FILTER_SKIP_KEY_TO_LABEL = {
    "all": ui_t("dup.filter.skip.all"),
    "skip": ui_t("dup.filter.skip.skipped"),
    "active": ui_t("dup.filter.skip.active"),
}
SORT_KEY_TO_LABEL = {
    "size_desc": ui_t("dup.sort.size_desc"),
    "size_asc": ui_t("dup.sort.size_asc"),
    "name": ui_t("dup.sort.name"),
    "type": ui_t("dup.sort.type"),
}


def _dup_section_label(parent: Any, text: str) -> None:
    ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(
        anchor="w", padx=8, pady=(10, 4)
    )

def _path_same(a: str, b: str) -> bool:
    """Сравнение путей для «оставить» vs список (Windows / разный регистр / слэши)."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class DuplicatesPane(ctk.CTkFrame):
    def __init__(self, master: Any, app: "App") -> None:
        super().__init__(master, fg_color="transparent")
        self._app = app
        self._sig_db: SignatureDatabase = app._sig_db
        self._dup_queue: queue.Queue = queue.Queue()
        self._dup_worker: DuplicateFinderWorker | None = None
        self._running = False
        self._current_run_id: str | None = None

        self._dir_var = ctk.StringVar(value="")
        self._preset_var = ctk.StringVar(value="balanced")
        self._media_mode_var = ctk.StringVar(value=MediaScanMode.PHOTOS_ONLY.value)
        self._workers_var = ctk.StringVar(value="3")
        self._keep_var = ctk.StringVar(value="largest_pixels")
        self._force_recompute_var = ctk.BooleanVar(value=bool(getattr(app, "_dup_force_recompute_default", False)))
        self._llm_var = ctk.BooleanVar(value=False)
        self._llm_user_overridden = False

        self._dup_api_var = ctk.StringVar(value=(app._api_var.get().strip() or DEFAULT_API_BASE))
        self._dup_model_var = ctk.StringVar(value=app._model_var.get() or ui_t("lm.models.placeholder"))
        self._dup_model_manual_var = ctk.StringVar(value=str(app._model_manual_var.get() or "").strip())
        self._dup_probe_busy = False

        self._groups: list[dict[str, Any]] = []
        self._group_keep_check_vars: list[dict[str, ctk.BooleanVar]] = []
        self._group_path_lists: list[list[str]] = []
        self._group_skip_vars: list[ctk.BooleanVar] = []
        self._group_approved_vars: list[ctk.BooleanVar] = []
        self._group_delete_entire_vars: list[ctk.BooleanVar] = []
        self._group_ids: list[str] = []
        self._last_records_meta: list[dict[str, Any]] = []
        self._group_records: list[dict[str, Any]] = []
        self._filtered_indices: list[int] = []
        self._session_key: str | None = None
        self._filter_type_key = ctk.StringVar(value="all")
        self._filter_skip_key = ctk.StringVar(value="all")
        self._sort_key = ctk.StringVar(value="size_desc")
        self._filter_min_size_var = ctk.StringVar(value="2")
        self._profiles: dict[str, dict[str, Any]] = {}
        self._advanced_visible = False
        self._last_scan_summary: dict[str, int] = {}
        self._frm_advanced: ctk.CTkFrame | None = None
        self._btn_advanced_toggle: ctk.CTkButton | None = None
        self._llm_hint_label: ctk.CTkLabel | None = None
        self._last_export_dir: Path | None = None
        self._persist_after_id: str | int | None = None
        self._viewer_window: DuplicateGroupsViewer | None = None

        self._build()
        self._load_dup_settings_into_fields()
        self._media_mode_var.trace_add("write", lambda *_: self.after(0, self._update_dup_ffmpeg_hint))
        self.after(150, self._poll_dup_queue)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        controls_host = ctk.CTkFrame(self, fg_color="transparent")
        controls_host.pack(fill="x", **pad)
        self._controls_scroll = ctk.CTkScrollableFrame(controls_host, height=320, fg_color="transparent")
        self._controls_scroll.pack(fill="x", expand=False)
        top = ctk.CTkFrame(self._controls_scroll, fg_color=("gray90", "gray16"), corner_radius=8)
        top.pack(fill="x", padx=2, pady=2)

        _dup_section_label(top, ui_t("dup.section.source"))
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkEntry(row, textvariable=self._dir_var, placeholder_text=ui_t("dup.folder.placeholder")).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ctk.CTkButton(row, text=ui_t("dup.folder.pick"), width=100, command=self._pick_dir).pack(side="right")
        row_m = ctk.CTkFrame(top, fg_color="transparent")
        row_m.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row_m, text=ui_t("dup.media"), width=70, anchor="w").pack(side="left")
        for val, label in (
            (MediaScanMode.PHOTOS_ONLY.value, ui_t("dup.media.photos")),
            (MediaScanMode.PHOTOS_AND_VIDEO.value, ui_t("dup.media.photos_video")),
            (MediaScanMode.VIDEO_ONLY.value, ui_t("dup.media.video_only")),
        ):
            ctk.CTkRadioButton(row_m, text=label, variable=self._media_mode_var, value=val).pack(side="left", padx=(0, 12))

        _dup_section_label(top, ui_t("dup.section.accuracy"))
        row_p = ctk.CTkFrame(top, fg_color="transparent")
        row_p.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row_p, text=ui_t("dup.preset"), width=120, anchor="w").pack(side="left")
        _acc_labels = [ui_t(f"dup.accuracy.{k}") for k in ACCURACY_PRESET_KEYS]
        self._preset_combo = ctk.CTkComboBox(
            row_p,
            values=_acc_labels,
            width=420,
            state="readonly",
            command=lambda _v: self._on_preset_change(),
        )
        self._preset_combo.pack(side="left", padx=(0, 8))
        self._preset_combo.set(ui_t("dup.accuracy.balanced"))

        row_method = ctk.CTkFrame(top, fg_color="transparent")
        row_method.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkCheckBox(row_method, text=ui_t("dup.llm.checkbox"), variable=self._llm_var, command=self._on_llm_toggle).pack(
            side="left", anchor="w"
        )
        self._llm_hint_label = ctk.CTkLabel(
            top,
            text=ui_t("dup.llm.video_hint"),
            text_color=("gray35", "gray65"),
            anchor="w",
            wraplength=760,
            font=ctk.CTkFont(size=11),
        )
        self._llm_hint_label.pack(fill="x", padx=12, pady=(0, 4))

        llm_dup = ctk.CTkFrame(top, fg_color=("gray92", "gray18"), corner_radius=6)
        llm_dup.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(llm_dup, text=ui_t("dup.llm.block_title"), font=ctk.CTkFont(size=12, weight="bold"), anchor="w").pack(
            anchor="w", padx=8, pady=(6, 2)
        )
        row_dup_api = ctk.CTkFrame(llm_dup, fg_color="transparent")
        row_dup_api.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_dup_api, text="Базовый URL:", width=120, anchor="w").pack(side="left")
        ctk.CTkEntry(row_dup_api, textvariable=self._dup_api_var, placeholder_text="http://host:port").pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        self._btn_dup_refresh = ctk.CTkButton(
            row_dup_api, text=ui_t("lm.refresh_models"), width=200, command=self._on_dup_refresh_models
        )
        self._btn_dup_refresh.pack(side="right")
        row_dup_m = ctk.CTkFrame(llm_dup, fg_color="transparent")
        row_dup_m.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_dup_m, text="Модель (список):", width=120, anchor="w").pack(side="left")
        self._dup_model_combo = ctk.CTkComboBox(
            row_dup_m,
            values=[ui_t("lm.models.placeholder")],
            variable=self._dup_model_var,
            width=420,
            state="readonly",
            command=lambda _v: self._on_dup_model_combo_change(),
        )
        self._dup_model_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        row_dup_man = ctk.CTkFrame(llm_dup, fg_color="transparent")
        row_dup_man.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(row_dup_man, text="Вручную (приоритет):", width=120, anchor="w").pack(side="left")
        ctk.CTkEntry(
            row_dup_man,
            textvariable=self._dup_model_manual_var,
            placeholder_text="Если пусто — берётся из списка выше",
        ).pack(side="left", fill="x", expand=True)

        self._method_label = ctk.CTkLabel(
            top,
            text="",
            anchor="w",
            justify="left",
            wraplength=760,
            text_color=("gray35", "gray65"),
        )
        self._method_label.pack(fill="x", padx=8, pady=(0, 4))

        row_k = ctk.CTkFrame(top, fg_color="transparent")
        row_k.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row_k, text=ui_t("dup.keep_policy"), anchor="w").pack(side="left")
        for val, lbl in (
            ("largest_pixels", ui_t("dup.keep.largest_px")),
            ("newest_mtime", ui_t("dup.keep.newest")),
            ("largest_file", ui_t("dup.keep.largest_file")),
        ):
            ctk.CTkRadioButton(row_k, text=lbl, variable=self._keep_var, value=val).pack(side="left", padx=(0, 10))

        _dup_section_label(top, ui_t("dup.section.scan"))
        row_b = ctk.CTkFrame(top, fg_color="transparent")
        row_b.pack(fill="x", padx=8, pady=(2, 4))
        self._btn_scan = ctk.CTkButton(row_b, text=ui_t("dup.scan"), command=self._on_scan)
        self._btn_scan.pack(side="left", padx=(0, 8))
        self._btn_pause = ctk.CTkButton(row_b, text=ui_t("buttons.pause"), command=self._on_pause, state="disabled")
        self._btn_pause.pack(side="left", padx=(0, 8))
        self._btn_stop = ctk.CTkButton(row_b, text=ui_t("buttons.stop"), command=self._on_stop, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))
        self._btn_regroup = ctk.CTkButton(row_b, text=ui_t("dup.regroup"), command=self._on_regroup)
        self._btn_regroup.pack(side="left", padx=(0, 8))

        self._btn_advanced_toggle = ctk.CTkButton(
            top,
            text=ui_t("dup.advanced.toggle_show"),
            fg_color=("gray75", "gray30"),
            command=self._toggle_advanced_section,
        )
        self._btn_advanced_toggle.pack(anchor="w", padx=8, pady=(4, 2))

        self._frm_advanced = ctk.CTkFrame(top, fg_color=("gray93", "gray19"), corner_radius=6)
        row_adv1 = ctk.CTkFrame(self._frm_advanced, fg_color="transparent")
        row_adv1.pack(fill="x", padx=8, pady=6)
        ctk.CTkCheckBox(row_adv1, text=ui_t("dup.force_recompute"), variable=self._force_recompute_var).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(row_adv1, text=ui_t("dup.workers"), width=72, anchor="w").pack(side="left", padx=(8, 4))
        ctk.CTkComboBox(row_adv1, values=["1", "2", "3", "4"], variable=self._workers_var, width=70, state="readonly").pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            row_adv1,
            text=ui_t("dup.workers.hint"),
            text_color=("gray38", "gray62"),
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=480,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        row_adv2 = ctk.CTkFrame(self._frm_advanced, fg_color="transparent")
        row_adv2.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(row_adv2, text=ui_t("dup.bulk.skip_small"), width=150, command=self._bulk_skip_small).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row_adv2, text=ui_t("dup.bulk.unskip_all"), width=150, command=self._bulk_unskip_all).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row_adv2, text=ui_t("dup.bulk.suggested_keep"), width=200, command=self._apply_suggested_deletes).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(row_adv2, text=ui_t("dup.profile.save"), width=120, command=self._save_profile).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row_adv2, text=ui_t("dup.profile.load"), width=120, command=self._load_profile).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row_adv2, text=ui_t("dup.undo_log"), width=120, command=self._show_undo_log).pack(side="left")

        _dup_section_label(top, ui_t("dup.section.results"))
        self._dup_ffmpeg_hint = ctk.CTkLabel(
            top,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray30", "gray65"),
        )
        self._dup_ffmpeg_hint.pack(fill="x", padx=12, pady=(0, 4))
        self._btn_review = ctk.CTkButton(
            top, text=ui_t("dup.review.open"), width=280, fg_color=("gray75", "gray30"), command=self._open_duplicate_review
        )
        self._btn_review.pack(anchor="w", padx=8, pady=(0, 6))
        ctk.CTkLabel(
            top,
            text=ui_t("dup.viewer.hint"),
            text_color=("gray35", "gray65"),
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=760,
        ).pack(fill="x", padx=12, pady=(0, 4))
        row_filters = ctk.CTkFrame(top, fg_color="transparent")
        row_filters.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(row_filters, text=ui_t("dup.filter.type"), width=95, anchor="w").pack(side="left")
        self._combo_filter_type = ctk.CTkComboBox(
            row_filters,
            values=list(FILTER_TYPE_KEY_TO_LABEL.values()),
            width=200,
            state="readonly",
            command=self._on_pick_filter_type,
        )
        self._combo_filter_type.pack(side="left", padx=(4, 8))
        ctk.CTkLabel(row_filters, text=ui_t("dup.filter.skip"), width=80, anchor="w").pack(side="left")
        self._combo_filter_skip = ctk.CTkComboBox(
            row_filters,
            values=list(FILTER_SKIP_KEY_TO_LABEL.values()),
            width=200,
            state="readonly",
            command=self._on_pick_filter_skip,
        )
        self._combo_filter_skip.pack(side="left", padx=(4, 8))
        ctk.CTkLabel(row_filters, text=ui_t("dup.filter.min_size"), width=150, anchor="w").pack(side="left")
        ctk.CTkComboBox(
            row_filters,
            values=["2", "3", "5", "10"],
            variable=self._filter_min_size_var,
            width=56,
            state="readonly",
            command=lambda _v: self._refresh_virtualized_groups(),
        ).pack(side="left", padx=(4, 8))
        row_filters2 = ctk.CTkFrame(top, fg_color="transparent")
        row_filters2.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row_filters2, text=ui_t("dup.filter.sort"), width=95, anchor="w").pack(side="left")
        self._combo_sort = ctk.CTkComboBox(
            row_filters2,
            values=list(SORT_KEY_TO_LABEL.values()),
            width=280,
            state="readonly",
            command=self._on_pick_sort,
        )
        self._combo_sort.pack(side="left", padx=(4, 8))

        row_prog = ctk.CTkFrame(top, fg_color="transparent")
        row_prog.pack(fill="x", padx=8, pady=(0, 4))
        self._lbl_review_progress = ctk.CTkLabel(row_prog, text="", anchor="w", text_color=("gray28", "gray72"))
        self._lbl_review_progress.pack(side="left", fill="x", expand=True)

        self._lbl_hotkeys = ctk.CTkLabel(
            top,
            text=ui_t("dup.hotkeys"),
            anchor="w",
            justify="left",
            wraplength=760,
            text_color=("gray32", "gray68"),
            font=ctk.CTkFont(size=11),
        )
        self._lbl_hotkeys.pack(fill="x", padx=12, pady=(0, 6))

        _dup_section_label(top, ui_t("dup.section.actions"))
        row_act = ctk.CTkFrame(top, fg_color="transparent")
        row_act.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(
            row_act,
            text=ui_t("dup.delete.approve_only"),
            text_color=("gray30", "gray70"),
            anchor="w",
        ).pack(side="left", padx=(0, 12))
        self._lbl_delete_count = ctk.CTkLabel(
            row_act,
            text=ui_t("dup.delete.count", n=0),
            text_color=("gray15", "gray85"),
            font=ctk.CTkFont(weight="bold"),
        )
        self._lbl_delete_count.pack(side="left", padx=(0, 16))
        row_act2 = ctk.CTkFrame(top, fg_color="transparent")
        row_act2.pack(fill="x", padx=8, pady=(0, 8))
        self._btn_apply_suggested = ctk.CTkButton(
            row_act2,
            text=ui_t("dup.delete.apply_suggested"),
            command=self._apply_suggested_deletes_wrap,
        )
        self._btn_apply_suggested.pack(side="left", padx=(0, 8))
        self._btn_preview = ctk.CTkButton(row_act2, text=ui_t("dup.delete.preview"), command=self._preview_delete)
        self._btn_preview.pack(side="left", padx=(0, 8))
        self._btn_trash = ctk.CTkButton(
            row_act2, text=ui_t("dup.delete.trash"), fg_color="darkred", command=lambda: self._do_delete(trash=True)
        )
        self._btn_trash.pack(side="left", padx=(0, 8))
        self._btn_perm = ctk.CTkButton(
            row_act2, text=ui_t("dup.delete.permanent"), fg_color="#553333", command=lambda: self._do_delete(trash=False)
        )
        self._btn_perm.pack(side="left")

        self._stage_label = ctk.CTkLabel(self, text=ui_t("dup.stage.generic", name="—"))
        self._stage_label.pack(anchor="w", padx=12, pady=(2, 0))
        self._prog = ctk.CTkProgressBar(self)
        self._prog.pack(fill="x", padx=12, pady=(4, 2))
        self._prog.set(0)
        self._prog_label = ctk.CTkLabel(self, text=ui_t("dup.stage.done"))
        self._prog_label.pack(anchor="w", padx=12)

        viewer_box = ctk.CTkFrame(self, fg_color=("gray88", "gray18"), corner_radius=12)
        viewer_box.pack(fill="both", expand=True, padx=12, pady=(10, 12))
        self._btn_open_viewer = ctk.CTkButton(
            viewer_box,
            text=ui_t("dup.viewer.open"),
            height=52,
            corner_radius=10,
            fg_color=CTK_ACCENT_PRIMARY,
            hover_color=CTK_ACCENT_PRIMARY_HOVER,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._open_groups_viewer,
            state="disabled",
        )
        self._btn_open_viewer.pack(pady=(32, 24))

        row_more = ctk.CTkFrame(self, fg_color="transparent")
        row_more.pack(fill="x", padx=12, pady=(0, 8))
        self._lbl_groups_summary = ctk.CTkLabel(
            row_more,
            text="",
            anchor="w",
            justify="left",
            wraplength=760,
            text_color=("gray28", "gray72"),
        )
        self._lbl_groups_summary.pack(side="left", fill="x", expand=True)

        self._sync_filter_sort_combos()
        self._on_preset_change()
        self._update_dup_llm_buttons_state()
        self._update_dup_ffmpeg_hint()
        self.bind("<KeyPress-s>", self._on_hotkey_s)
        self.bind("<KeyPress-a>", self._on_hotkey_a)
        self.bind("<KeyPress-d>", self._on_hotkey_d)
        self.bind("<KeyPress-j>", self._on_hotkey_j)
        self.bind("<KeyPress-k>", self._on_hotkey_k)
        self.bind("<FocusIn>", lambda _e: self.focus_set())
        self.bind("<Configure>", self._on_resize)
        self._lbl_groups_summary.configure(text=ui_t("dup.groups.summary.empty"))
        self._frm_advanced.pack_forget()

    def _make_modal(self, top: ctk.CTkToplevel) -> None:
        top.update_idletasks()
        sw = max(640, int(top.winfo_screenwidth() * 0.92))
        sh = max(520, int(top.winfo_screenheight() * 0.88))
        w = min(top.winfo_reqwidth(), sw)
        h = min(top.winfo_reqheight(), sh)
        x = max(20, (top.winfo_screenwidth() - w) // 2)
        y = max(20, (top.winfo_screenheight() - h) // 2)
        top.geometry(f"{w}x{h}+{x}+{y}")
        top.transient(self.winfo_toplevel())
        top.lift()
        top.attributes("-topmost", True)
        top.after(150, lambda: top.attributes("-topmost", False))
        try:
            top.grab_set()
        except Exception:
            pass
        try:
            top.focus_force()
        except Exception:
            pass

    def _on_resize(self, _event=None) -> None:
        width = max(720, self.winfo_width())
        wrap = max(460, width - 180)
        summary_wrap = max(380, width - 320)
        ctrl_h = max(230, min(380, int(self.winfo_height() * 0.36)))
        try:
            self._controls_scroll.configure(height=ctrl_h)
        except Exception:
            pass
        self._method_label.configure(wraplength=wrap)
        self._lbl_hotkeys.configure(wraplength=wrap)
        self._lbl_groups_summary.configure(wraplength=summary_wrap)
        if self._llm_hint_label is not None:
            self._llm_hint_label.configure(wraplength=wrap)

    def _toggle_advanced_section(self) -> None:
        if self._frm_advanced is None or self._btn_advanced_toggle is None:
            return
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self._frm_advanced.pack(fill="x", padx=8, pady=(0, 6), after=self._btn_advanced_toggle)
            self._btn_advanced_toggle.configure(text=ui_t("dup.advanced.toggle_hide"))
        else:
            self._frm_advanced.pack_forget()
            self._btn_advanced_toggle.configure(text=ui_t("dup.advanced.toggle_show"))

    def _sync_filter_sort_combos(self) -> None:
        self._combo_filter_type.set(FILTER_TYPE_KEY_TO_LABEL.get(self._filter_type_key.get(), ui_t("dup.filter.type.all")))
        self._combo_filter_skip.set(FILTER_SKIP_KEY_TO_LABEL.get(self._filter_skip_key.get(), ui_t("dup.filter.skip.all")))
        self._combo_sort.set(SORT_KEY_TO_LABEL.get(self._sort_key.get(), ui_t("dup.sort.size_desc")))

    def _on_pick_filter_type(self, choice: str) -> None:
        rev = {v: k for k, v in FILTER_TYPE_KEY_TO_LABEL.items()}
        self._filter_type_key.set(rev.get(choice, "all"))
        self._refresh_virtualized_groups()

    def _on_pick_filter_skip(self, choice: str) -> None:
        rev = {v: k for k, v in FILTER_SKIP_KEY_TO_LABEL.items()}
        self._filter_skip_key.set(rev.get(choice, "all"))
        self._refresh_virtualized_groups()

    def _on_pick_sort(self, choice: str) -> None:
        rev = {v: k for k, v in SORT_KEY_TO_LABEL.items()}
        self._sort_key.set(rev.get(choice, "size_desc"))
        self._refresh_virtualized_groups()

    def _update_dup_ffmpeg_hint(self) -> None:
        try:
            mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            mode = MediaScanMode.PHOTOS_ONLY
        if mode == MediaScanMode.PHOTOS_ONLY:
            self._dup_ffmpeg_hint.configure(text="")
            return
        ff = resolve_ffmpeg_executable()
        if ff:
            short = ff if len(ff) < 64 else "..." + ff[-60:]
            self._dup_ffmpeg_hint.configure(text=ui_t("dup.ffmpeg.ok", path=short), text_color=("gray30", "gray65"))
        else:
            self._dup_ffmpeg_hint.configure(text=ui_t("dup.ffmpeg.missing"), text_color=("goldenrod", "#c9a227"))

    def _open_groups_viewer(self) -> None:
        if not self._group_records:
            messagebox.showinfo("Дубликаты", "Сначала выполните сканирование.")
            return
        if self._viewer_window is not None:
            try:
                if self._viewer_window.winfo_exists():
                    self._viewer_window.lift()
                    self._viewer_window.focus_force()
                    self._viewer_window.refresh_safe()
                    return
            except Exception:
                self._viewer_window = None
        self._viewer_window = DuplicateGroupsViewer(self.winfo_toplevel(), self)

    def _notify_viewer_filters_changed(self) -> None:
        w = self._viewer_window
        if w is not None:
            try:
                if w.winfo_exists():
                    w.after(0, w.refresh_safe)
            except Exception:
                pass

    def _refresh_delete_count_and_review_progress(self) -> None:
        n = len(self._collect_delete_paths())
        self._lbl_delete_count.configure(text=ui_t("dup.delete.count", n=n))
        total = len(self._filtered_indices)
        approved = 0
        for gi in self._filtered_indices:
            if gi < len(self._group_approved_vars) and self._group_approved_vars[gi].get():
                approved += 1
        self._lbl_review_progress.configure(text=ui_t("dup.approve.progress", approved=approved, total=total))

    def _apply_suggested_deletes_wrap(self) -> None:
        self._approve_all_groups()
        self._apply_suggested_deletes()
        self._refresh_delete_count_and_review_progress()
        self.after(100, self._preview_delete)

    def _after_keep_change(self) -> None:
        for i, km in enumerate(self._group_keep_check_vars):
            if i < len(self._group_skip_vars) and self._group_skip_vars[i].get():
                continue
            plist = self._group_path_lists[i]
            if plist and not any(km[p].get() for p in plist):
                km[plist[0]].set(True)
        self._persist_review_state()
        self._refresh_delete_count_and_review_progress()

    def _preset_key_from_combo(self) -> str:
        labels = [ui_t(f"dup.accuracy.{k}") for k in ACCURACY_PRESET_KEYS]
        cur = self._preset_combo.get()
        try:
            idx = labels.index(cur)
            return ACCURACY_PRESET_KEYS[idx]
        except ValueError:
            return "balanced"

    def _pick_dir(self) -> None:
        from tkinter import filedialog

        d = filedialog.askdirectory(title=ui_t("dup.pick_dir.title"))
        if d:
            self._dir_var.set(d)

    def _parse_options(self) -> DuplicateFinderOptions | None:
        key = self._preset_key_from_combo()
        base = options_from_preset(key)
        try:
            workers = max(1, min(4, int(self._workers_var.get().strip())))
        except ValueError:
            messagebox.showerror("Параметры", "Воркеры должны быть числом от 1 до 4")
            return None
        base.parallel_workers = workers
        base.keep_policy = self._keep_var.get()  # type: ignore[assignment]
        base.strictness = key  # type: ignore[assignment]
        base.use_llm_pairs = bool(self._llm_var.get())
        return base

    def _on_preset_change(self) -> None:
        key = self._preset_key_from_combo()
        base = options_from_preset(key)
        # Keep user choice if already changed manually after initial load.
        if not hasattr(self, "_llm_initialized"):
            self._llm_var.set(base.use_llm_pairs)
            self._llm_initialized = True
        elif not self._llm_user_overridden:
            self._llm_var.set(base.use_llm_pairs)
        blurb_map = {
            "fast": "dup.method.blurb_fast",
            "balanced": "dup.method.blurb_balanced",
            "strict": "dup.method.blurb_strict",
            "deep": "dup.method.blurb_deep",
        }
        text = ui_t(blurb_map.get(key, "dup.method.blurb_balanced"))
        if self._llm_var.get():
            text = f"{text}\n\n{ui_t('dup.method.llm_addon')}"
        self._method_label.configure(text=text)

    def _on_llm_toggle(self) -> None:
        self._llm_user_overridden = True
        self._on_preset_change()

    def _dup_model_resolved(self) -> str:
        manual = self._dup_model_manual_var.get().strip()
        if manual:
            return manual
        v = self._dup_model_var.get().strip()
        if v.startswith("—") or v == "":
            return DEFAULT_MODEL
        return v

    def _on_dup_model_combo_change(self) -> None:
        if self._dup_model_manual_var.get().strip():
            self._dup_model_manual_var.set("")
            self._app._append_log("Дубликаты: ручная модель очищена — используется выбранная из списка.")

    def _set_dup_probe_busy(self, busy: bool) -> None:
        self._dup_probe_busy = busy
        self._update_dup_llm_buttons_state()

    def _update_dup_llm_buttons_state(self) -> None:
        st = "disabled" if self._running or self._dup_probe_busy else "normal"
        self._btn_dup_refresh.configure(state=st)

    def _on_dup_refresh_models(self) -> None:
        base = self._dup_api_var.get().strip() or DEFAULT_API_BASE

        def work() -> None:
            self.after(0, lambda: self._set_dup_probe_busy(True))
            try:
                models = list_models(base, api_key=self._app._api_key_resolved())
            except Exception as e:
                self.after(0, lambda: self._app._append_log(f"Дубликаты — список моделей: ошибка {e!s}"))
                self.after(0, lambda: self._set_dup_probe_busy(False))
                return

            def apply() -> None:
                if models:
                    self._dup_model_combo.configure(values=models)
                    self._dup_model_var.set(models[0])
                else:
                    self._dup_model_combo.configure(values=["— сервер вернул пустой список —"])
                    self._dup_model_var.set(self._dup_model_combo.cget("values")[0])
                self._app._append_log(f"Дубликаты: загружено моделей: {len(models)}")
                self._set_dup_probe_busy(False)

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()
        self._app._append_log("Дубликаты: запрос списка моделей…")

    def dup_settings_dict(self) -> dict[str, Any]:
        o = self._parse_options() or options_from_preset("balanced")
        d = options_to_dict(o)
        d["preset"] = self._preset_key_from_combo()
        d["folder"] = self._dir_var.get().strip()
        d["media_mode"] = self._media_mode_var.get().strip()
        d["force_recompute"] = self._force_recompute_var.get()
        d["llm_enabled"] = bool(self._llm_var.get())
        d["review_sort"] = self._sort_key.get()
        d["review_filter_type"] = self._filter_type_key.get()
        d["review_filter_skip"] = self._filter_skip_key.get()
        d["review_filter_min_size"] = self._filter_min_size_var.get()
        d["profiles"] = self._profiles
        d["dup_api_base"] = self._dup_api_var.get().strip() or DEFAULT_API_BASE
        d["dup_model"] = self._dup_model_var.get().strip()
        d["dup_model_manual"] = self._dup_model_manual_var.get().strip()
        return d

    def _load_dup_settings_into_fields(self) -> None:
        raw = getattr(self._app, "_loaded_settings", None)
        if not isinstance(raw, dict):
            return
        dup = raw.get("duplicate_finder")
        if not isinstance(dup, dict):
            return
        folder = str(dup.get("folder", "") or "")
        if folder:
            self._dir_var.set(folder)
        mm = str(dup.get("media_mode", "") or "")
        if mm in {m.value for m in MediaScanMode}:
            self._media_mode_var.set(mm)
        preset = str(dup.get("preset", "balanced") or "balanced")
        keys = list(ACCURACY_PRESET_KEYS)
        labels = [ui_t(f"dup.accuracy.{k}") for k in ACCURACY_PRESET_KEYS]
        if preset in keys:
            self._preset_combo.set(labels[keys.index(preset)])
        base = options_from_preset(preset)
        merged = merge_options_from_dict(dup, base)
        self._workers_var.set(str(max(1, min(4, int(merged.parallel_workers)))))
        self._keep_var.set(str(merged.keep_policy))
        self._force_recompute_var.set(bool(dup.get("force_recompute", False)))
        self._llm_var.set(bool(dup.get("llm_enabled", merged.use_llm_pairs)))
        self._llm_user_overridden = False
        if str(dup.get("review_sort", "")) in {"size_desc", "size_asc", "name", "type"}:
            self._sort_key.set(str(dup.get("review_sort")))
        if str(dup.get("review_filter_type", "")) in {"all", "exact", "similar"}:
            self._filter_type_key.set(str(dup.get("review_filter_type")))
        if str(dup.get("review_filter_skip", "")) in {"all", "skip", "active"}:
            self._filter_skip_key.set(str(dup.get("review_filter_skip")))
        if str(dup.get("review_filter_min_size", "")) in {"2", "3", "5", "10"}:
            self._filter_min_size_var.set(str(dup.get("review_filter_min_size")))
        self._sync_filter_sort_combos()
        profiles = dup.get("profiles")
        if isinstance(profiles, dict):
            self._profiles = {str(k): v for k, v in profiles.items() if isinstance(v, dict)}
        dab = str(dup.get("dup_api_base", "") or "").strip()
        if dab:
            self._dup_api_var.set(dab)
        dm = str(dup.get("dup_model", "") or "").strip()
        if dm:
            self._dup_model_var.set(dm)
        self._dup_model_manual_var.set(str(dup.get("dup_model_manual", "") or "").strip())
        self._llm_initialized = True
        self._on_preset_change()

    def _ask_resume_mode(self, root: Path, strictness: str, media_mode: MediaScanMode) -> tuple[bool, bool]:
        key = make_session_key(str(root.resolve()), media_mode.value, strictness)
        session = self._sig_db.get_session(key)
        if not session:
            return False, False
        done = int(session["done_files"])
        total = int(session["total_files"])
        status = str(session["status"])
        if total <= 0:
            return False, False
        if done >= total and status == "completed":
            return False, False

        res = messagebox.askyesnocancel(
            "Найден незавершённый прогресс",
            "Для этой папки уже есть сохранённый прогресс.\n\nДа — продолжить\nНет — начать заново\nОтмена — не запускать",
        )
        if res is None:
            return True, False
        if res:
            return False, True
        return False, False

    def _on_scan(self) -> None:
        if self._running:
            return
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            messagebox.showwarning("Папка", "Укажите существующую папку.")
            return
        opts = self._parse_options()
        if opts is None:
            return
        try:
            mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            mode = MediaScanMode.PHOTOS_ONLY

        cancelled, resume = self._ask_resume_mode(root, opts.strictness, mode)
        if cancelled:
            return
        force_recompute = bool(self._force_recompute_var.get()) or (not resume)
        self._session_key = make_session_key(str(root.resolve()), mode.value, opts.strictness)

        self._running = True
        self._btn_scan.configure(state="disabled")
        self._btn_pause.configure(state="normal", text="Пауза")
        self._btn_stop.configure(state="normal")
        self._prog.set(0)
        self._stage_label.configure(text=ui_t("dup.stage.start"))
        self._prog_label.configure(text=ui_t("dup.stage.scan"))
        self._clear_groups_ui()
        self._app._save_gui_settings()
        self._update_dup_llm_buttons_state()

        dup_api = self._dup_api_var.get().strip() or DEFAULT_API_BASE
        dup_model = self._dup_model_resolved()
        self._dup_worker = DuplicateFinderWorker(
            self._sig_db,
            self._dup_queue,
            api_base=dup_api,
            model=dup_model,
            api_key=self._app._api_key_resolved(),
        )
        self._app._append_log(
            f"Дубликаты: API={dup_api} | модель={dup_model} | LLM={'ON' if opts.use_llm_pairs else 'OFF'}"
        )
        self._dup_worker.reset_stop()
        self._dup_worker.set_paused(False)
        self._dup_worker.start_in_thread(root, opts, media_mode=mode, force_recompute=force_recompute, resume=resume)
        self._current_run_id = self._dup_worker.run_id

    def _on_pause(self) -> None:
        if not self._dup_worker:
            return
        paused = not self._dup_worker.is_paused()
        self._dup_worker.set_paused(paused)
        self._btn_pause.configure(text=ui_t("buttons.resume") if paused else ui_t("buttons.pause"))
        self._app._append_log("Дубликаты: пауза" if paused else "Дубликаты: продолжение")

    def _on_stop(self) -> None:
        if self._dup_worker:
            self._dup_worker.request_stop()

    def _on_regroup(self) -> None:
        if self._running:
            return
        if not self._last_records_meta:
            messagebox.showinfo("Кеш", "Сначала выполните сканирование.")
            return
        opts = self._parse_options()
        if opts is None:
            return
        paths = [Path(m["path"]) for m in self._last_records_meta]
        records = load_records_from_db(paths, self._sig_db, opts)
        if len(records) < 2:
            messagebox.showinfo("Кеш", "Недостаточно данных в кеше, выполните сканирование.")
            return
        try:
            mode = MediaScanMode(self._media_mode_var.get())
        except ValueError:
            mode = MediaScanMode.PHOTOS_ONLY
        self._session_key = make_session_key(str(Path(self._dir_var.get().strip()).resolve()), mode.value, opts.strictness)
        llm_decisions = None
        if opts.use_llm_pairs and self._session_key:
            llm_decisions = self._sig_db.list_llm_pair_decisions(self._session_key)
        groups = regroup_from_cached_records(records, opts, llm_pair_decisions=llm_decisions)
        serial = [{"paths": [str(x) for x in g.paths], "suggested_keep": str(g.suggested_keep), "is_exact": g.is_exact} for g in groups]
        self._ingest_groups(serial)

    def _poll_dup_queue(self) -> None:
        try:
            while True:
                self._handle_dup_msg(self._dup_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(150, self._poll_dup_queue)

    def _handle_dup_msg(self, msg: dict[str, Any]) -> None:
        run_id = msg.get("run_id")
        if self._running and self._current_run_id and run_id and run_id != self._current_run_id:
            return
        if msg.get("type") == "dup_scan_done":
            self._current_run_id = str(run_id or self._current_run_id or "")
        msg_type = msg.get("type")
        if msg_type == "dup_scan_done":
            t0 = int(msg.get("total", 0))
            self._prog_label.configure(text=ui_t("dup.progress.files", done=0, total=t0))
        elif msg_type == "dup_progress":
            done = int(msg.get("done", 0))
            total = int(msg.get("total", 1))
            self._prog_label.configure(text=ui_t("dup.progress.files", done=done, total=total))
            if total > 0:
                self._prog.set(min(1.0, done / float(total)))
        elif msg_type == "dup_stage":
            st = str(msg.get("stage", ""))
            self._stage_label.configure(text=_dup_stage_label(st))
        elif msg_type == "dup_stage_progress":
            done = int(msg.get("done", 0))
            total = int(msg.get("total", 1))
            st = str(msg.get("stage", ""))
            if total > 0:
                self._prog.set(min(1.0, done / float(total)))
            self._prog_label.configure(
                text=ui_t("dup.progress.stage_detail", stage=_dup_stage_label(st), done=done, total=total)
            )
        elif msg_type == "dup_groups_ready":
            self._last_records_meta = list(msg.get("records") or [])
            self._ingest_groups(list(msg.get("groups") or []))
        elif msg_type == "dup_finished":
            self._running = False
            self._current_run_id = None
            self._btn_scan.configure(state="normal")
            self._btn_pause.configure(state="disabled", text=ui_t("buttons.pause"))
            self._btn_stop.configure(state="disabled")
            self._stage_label.configure(text=ui_t("dup.stage.done"))
            reason = str(msg.get("reason", ""))
            if self._last_scan_summary and reason == "completed" and self._group_records:
                ndel = len(self._collect_delete_paths())
                self._prog_label.configure(
                    text=ui_t(
                        "dup.finished.summary",
                        groups=self._last_scan_summary.get("groups", 0),
                        files=self._last_scan_summary.get("files", 0),
                        to_delete=ndel,
                    )
                )
            else:
                self._prog_label.configure(text=ui_t("dup.finished.with_reason", done=ui_t("dup.stage.done"), reason=reason))
            self._update_dup_llm_buttons_state()
        elif msg_type == "state_changed":
            state = str(msg.get("state", ""))
            if state == "paused":
                self._btn_pause.configure(text=ui_t("buttons.resume"))
            elif state == "running":
                self._btn_pause.configure(text=ui_t("buttons.pause"))
        elif msg_type == "log":
            self._app._append_log(str(msg.get("text", "")))
        elif msg_type == "metric":
            name = str(msg.get("name", "metric"))
            payload = msg.get("payload", {})
            self._app._append_log(f"[metrics:{name}] {payload}")

    def _clear_groups_ui(self) -> None:
        self._groups = []
        self._group_keep_check_vars = []
        self._group_path_lists = []
        self._group_skip_vars = []
        self._group_approved_vars = []
        self._group_delete_entire_vars = []
        self._group_ids = []
        self._group_records = []
        self._filtered_indices = []
        aid = getattr(self, "_persist_after_id", None)
        if aid is not None:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
            self._persist_after_id = None
        self._lbl_groups_summary.configure(text="")
        self._refresh_delete_count_and_review_progress()

    def _ingest_groups(self, groups: list[dict[str, Any]]) -> None:
        self._clear_groups_ui()
        self._groups = groups
        self._group_records = []
        for gi, g in enumerate(groups):
            paths = [Path(p) for p in g.get("paths", [])]
            pstr = [str(p) for p in paths]
            sk = str(g.get("suggested_keep", pstr[0] if pstr else ""))
            keep_map: dict[str, ctk.BooleanVar] = {}
            for p in pstr:
                keep_map[p] = ctk.BooleanVar(value=bool(sk and _path_same(p, sk)))
            if pstr and not any(keep_map[p].get() for p in pstr):
                keep_map[pstr[0]].set(True)
            skip_var = ctk.BooleanVar(value=False)
            approved_var = ctk.BooleanVar(value=False)
            delete_entire_var = ctk.BooleanVar(value=False)
            group_id = f"g-{gi}-{abs(hash('|'.join(pstr))) % 100000000}"
            self._group_keep_check_vars.append(keep_map)
            self._group_skip_vars.append(skip_var)
            self._group_approved_vars.append(approved_var)
            self._group_delete_entire_vars.append(delete_entire_var)
            self._group_path_lists.append(pstr)
            self._group_ids.append(group_id)
            self._group_records.append(
                {
                    "group_id": group_id,
                    "paths": pstr,
                    "suggested_keep": sk,
                    "is_exact": bool(g.get("is_exact")),
                }
            )
        self._restore_review_state()
        files_in = sum(len(g.get("paths", [])) for g in self._group_records)
        self._last_scan_summary = {"groups": len(groups), "files": files_in}
        self._last_export_dir = None
        if self._session_key and self._group_records:
            try:
                groups_payload = [
                    {"paths": list(g["paths"]), "suggested_keep": g["suggested_keep"], "is_exact": g["is_exact"]}
                    for g in self._group_records
                ]
                self._last_export_dir = export_duplicate_run(
                    self._session_key,
                    root_path=self._dir_var.get().strip(),
                    media_mode=self._media_mode_var.get().strip(),
                    strictness=self._preset_key_from_combo(),
                    groups=groups_payload,
                    records=list(self._last_records_meta),
                )
            except OSError:
                pass
        self._refresh_virtualized_groups()
        self._refresh_delete_count_and_review_progress()

    def _apply_suggested_deletes(self) -> None:
        for gi, g in enumerate(self._group_records):
            if gi >= len(self._group_keep_check_vars):
                break
            if gi >= len(self._group_approved_vars) or not self._group_approved_vars[gi].get():
                continue
            if self._group_skip_vars[gi].get():
                continue
            sk = str(g.get("suggested_keep", ""))
            km = self._group_keep_check_vars[gi]
            for p in self._group_path_lists[gi]:
                if p in km:
                    km[p].set(bool(sk and _path_same(p, sk)))
        self.flush_persist_review_state()
        self._refresh_delete_count_and_review_progress()

    def _keep_paths_for_group(self, gi: int) -> set[str]:
        plist = self._group_path_lists[gi]
        km = self._group_keep_check_vars[gi]
        chosen = {p for p in plist if km[p].get()}
        if not chosen and plist:
            chosen = {plist[0]}
        return chosen

    def _collect_delete_paths(self) -> list[Path]:
        seen: set[str] = set()
        out: list[Path] = []
        for i in range(len(self._group_keep_check_vars)):
            if i >= len(self._group_approved_vars) or not self._group_approved_vars[i].get():
                continue
            if i < len(self._group_skip_vars) and self._group_skip_vars[i].get():
                continue
            plist = self._group_path_lists[i]
            if i < len(self._group_delete_entire_vars) and self._group_delete_entire_vars[i].get():
                for p in plist:
                    pp = Path(p)
                    try:
                        key = str(pp.resolve())
                    except OSError:
                        key = os.path.normcase(os.path.normpath(str(pp)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(pp)
                continue
            keep_set = self._keep_paths_for_group(i)
            for p in plist:
                if p in keep_set:
                    continue
                pp = Path(p)
                try:
                    key = str(pp.resolve())
                except OSError:
                    key = os.path.normcase(os.path.normpath(str(pp)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(pp)
        return out

    def _delete_scope_summary(self) -> tuple[int, int, int, int]:
        """(файлов к удалению, групп с удалением, всего групп в скане, групп после фильтров)."""
        paths = self._collect_delete_paths()
        n_groups_del = 0
        for i in range(len(self._group_keep_check_vars)):
            if i >= len(self._group_approved_vars) or not self._group_approved_vars[i].get():
                continue
            if i < len(self._group_skip_vars) and self._group_skip_vars[i].get():
                continue
            plist = self._group_path_lists[i]
            if i < len(self._group_delete_entire_vars) and self._group_delete_entire_vars[i].get():
                if plist:
                    n_groups_del += 1
                continue
            keep_set = self._keep_paths_for_group(i)
            if any(p not in keep_set for p in plist):
                n_groups_del += 1
        return len(paths), n_groups_del, len(self._group_records), len(self._filtered_indices)

    def _update_groups_summary(self) -> None:
        total = len(self._group_records)
        filt = len(self._filtered_indices)
        if total == 0:
            self._lbl_groups_summary.configure(text=ui_t("dup.groups.summary.empty"))
            return
        self._lbl_groups_summary.configure(
            text=ui_t("dup.groups.summary", total=total, filt=filt),
        )

    def _skipped_groups_count(self) -> int:
        return sum(1 for v in self._group_skip_vars if v.get())

    def _preview_delete(self) -> None:
        paths = self._collect_delete_paths()
        if not paths:
            messagebox.showinfo(
                "Предпросмотр",
                "Нет файлов к удалению. Утвердите группы (галочка «Утвердить»), настройте «оставить» или «удалить всю группу».",
            )
            return
        n_files, n_gr_del, n_gr_tot, n_filt = self._delete_scope_summary()
        top = ctk.CTkToplevel(self)
        top.title("Предпросмотр удаления")
        top.geometry("720x520")
        self._make_modal(top)
        tb = ctk.CTkTextbox(top)
        tb.pack(fill="both", expand=True, padx=8, pady=8)
        skipped = self._skipped_groups_count()
        missing = sum(1 for p in paths if not p.exists())
        header = (
            f"Сводка\n"
            f"Файлов к удалению: {n_files}\n"
            f"Групп с удалением: {n_gr_del} из {n_gr_tot} (всего в скане)\n"
            f"Групп после фильтров списка: {n_filt}\n"
            f"{ui_t('dup.delete.approve_only')}\n"
            f"Пропущено групп (skip): {skipped}\n"
            f"Нет на диске: {missing}\n"
            f"{'-' * 48}\n"
        )
        show = paths[:800]
        tail = "" if len(paths) <= 800 else f"\n… и ещё {len(paths) - 800} путь(ей)"
        tb.insert("1.0", header + "\n".join(str(p) for p in show) + tail)
        tb.configure(state="disabled")

    def _confirm_permanent_delete(self, n_files: int) -> bool:
        result = {"ok": False}
        top = ctk.CTkToplevel(self)
        top.title(ui_t("dup.perm.title"))
        top.geometry("480x230")
        self._make_modal(top)
        var = ctk.BooleanVar(value=False)
        ctk.CTkLabel(top, text=f"Будет безвозвратно удалено файлов: {n_files}").pack(padx=16, pady=(16, 8))
        ctk.CTkCheckBox(top, text=ui_t("dup.perm.checkbox"), variable=var).pack(anchor="w", padx=16)
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(pady=20)

        def on_ok() -> None:
            if var.get():
                result["ok"] = True
            top.destroy()

        def on_cancel() -> None:
            top.destroy()

        ctk.CTkButton(row, text=ui_t("dup.perm.confirm"), command=on_ok).pack(side="left", padx=8)
        ctk.CTkButton(row, text=ui_t("dup.perm.cancel"), command=on_cancel).pack(side="left", padx=8)
        self.wait_window(top)
        return bool(result["ok"])

    def _do_delete(self, *, trash: bool) -> None:
        paths = self._collect_delete_paths()
        if not paths:
            messagebox.showinfo("Удаление", "Нет отмеченных файлов.")
            return
        mode = "в корзину" if trash else "безвозвратно"
        skipped = self._skipped_groups_count()
        missing = sum(1 for p in paths if not p.exists())
        n_files, n_gr_del, n_gr_tot, n_filt = self._delete_scope_summary()
        if not messagebox.askyesno(
            ui_t("dup.delete.confirm.title"),
            f"Удалить {n_files} файл(ов) {mode}?\n\n"
            f"Групп с удалением: {n_gr_del} из {n_gr_tot} (всего в скане).\n"
            f"В списке после фильтров: {n_filt} групп.\n"
            f"{ui_t('dup.delete.approve_only')}\n"
            f"Пропущено групп (skip): {skipped}\n"
            f"Нет на диске: {missing}\n",
        ):
            return
        if not trash and not self._confirm_permanent_delete(n_files):
            return
        self._run_delete_with_progress(paths, trash=trash)

    def _short_delete_filename(self, p: Path) -> str:
        s = p.name
        if len(s) > 52:
            return s[:24] + "…" + s[-24:]
        return s

    def _finalize_delete_outcome(
        self,
        paths: list[Path],
        errors: list[str],
        *,
        trash: bool,
        deleted_paths: list[Path] | None = None,
        cancelled: bool = False,
    ) -> None:
        if cancelled:
            deleted = deleted_paths or []
            if deleted:
                messagebox.showinfo(
                    ui_t("dup.delete.progress_title"),
                    ui_t("dup.delete.cancelled_partial", k=len(deleted), total=len(paths)),
                )
                self._append_journal_entry(deleted, trash=trash)
            else:
                messagebox.showinfo(ui_t("dup.delete.progress_title"), ui_t("dup.delete.cancelled_none"))
        elif errors:
            messagebox.showerror("Ошибки", "\n".join(errors[:20]))
        else:
            reported = deleted_paths if deleted_paths is not None else paths
            messagebox.showinfo("Готово", f"Удалено: {len(reported)}")
            self._append_journal_entry(reported, trash=trash)
        self._clear_groups_ui()
        self._groups = []
        self._last_records_meta = []

    def _run_delete_with_progress(self, paths: list[Path], *, trash: bool) -> None:
        total = len(paths)
        if total == 0:
            return

        if trash:
            try:
                import send2trash
            except ImportError:
                messagebox.showerror("send2trash", "Установите пакет send2trash.")
                return
        else:
            send2trash = None  # type: ignore[assignment]

        prog_win = ctk.CTkToplevel(self.winfo_toplevel())
        prog_win.title(ui_t("dup.delete.progress_title"))
        prog_win.configure(fg_color=("gray92", "gray17"))
        prog_win.resizable(False, False)
        ctk.CTkLabel(prog_win, text=ui_t("dup.delete.progress_label"), font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(18, 6)
        )
        lbl_count = ctk.CTkLabel(prog_win, text=ui_t("dup.delete.progress_count", done=0, total=total))
        lbl_count.pack(pady=(0, 4))
        lbl_file = ctk.CTkLabel(prog_win, text="", text_color=("gray40", "gray65"), wraplength=500)
        lbl_file.pack(pady=(0, 8))
        bar = ctk.CTkProgressBar(prog_win, width=440)
        bar.pack(padx=24, pady=(0, 8))
        bar.set(0)
        cancel_event = threading.Event()
        state_sync: dict[str, Any] = {"i": 0, "errors": [], "deleted": [], "cancelled": False}

        def request_cancel() -> None:
            state_sync["cancelled"] = True
            cancel_event.set()

        ctk.CTkButton(prog_win, text=ui_t("dup.delete.progress_cancel"), width=120, command=request_cancel).pack(
            pady=(0, 18)
        )
        self._make_modal(prog_win)

        batch = 5

        if total > DELETE_PROGRESS_THREAD_THRESHOLD:
            q: queue.Queue[tuple[Any, ...]] = queue.Queue()

            def worker() -> None:
                errs: list[str] = []
                deleted: list[Path] = []
                for j, p in enumerate(paths):
                    if cancel_event.is_set():
                        q.put(("finished", "cancelled", errs, deleted))
                        return
                    try:
                        if p.is_file():
                            if trash:
                                send2trash.send2trash(str(p))
                            else:
                                p.unlink()
                            deleted.append(p)
                        q.put(("progress", j + 1, p))
                    except OSError as e:
                        errs.append(f"{p}: {e!s}")
                    except Exception as e:
                        if trash:
                            errs.append(f"{p}: {e!s}")
                q.put(("finished", "done", errs, deleted))

            t = threading.Thread(target=worker, daemon=True, name="dup-delete-batch")
            t.start()
            finished = {"done": False}

            def poll() -> None:
                if finished["done"]:
                    return
                try:
                    if not prog_win.winfo_exists():
                        return
                except Exception:
                    return
                try:
                    while True:
                        msg = q.get_nowait()
                        kind = msg[0]
                        if kind == "progress":
                            _, done, path = msg
                            end = int(done)
                            if total:
                                bar.set(end / total)
                            lbl_count.configure(text=ui_t("dup.delete.progress_count", done=end, total=total))
                            lbl_file.configure(text=ui_t("dup.delete.progress_file", name=self._short_delete_filename(path)))
                        elif kind == "finished":
                            _, status, errs, deleted = msg
                            finished["done"] = True
                            try:
                                prog_win.destroy()
                            except Exception:
                                pass
                            errs_list = list(errs)
                            deleted_list = list(deleted)
                            st = str(status)

                            def finalize_thread(
                                paths_t: list[Path] = paths,
                                err_t: list[str] = errs_list,
                                del_t: list[Path] = deleted_list,
                                cancelled_flag: bool = st == "cancelled",
                            ) -> None:
                                self._finalize_delete_outcome(
                                    paths_t, err_t, trash=trash, deleted_paths=del_t, cancelled=cancelled_flag
                                )

                            self.after(0, finalize_thread)
                            return
                except queue.Empty:
                    pass
                if not finished["done"]:
                    self.after(16, poll)

            self.after(0, poll)
            return

        def step() -> None:
            try:
                if not prog_win.winfo_exists():
                    return
            except Exception:
                return
            if state_sync["cancelled"]:
                try:
                    prog_win.destroy()
                except Exception:
                    pass
                self.after(
                    0,
                    lambda: self._finalize_delete_outcome(
                        paths,
                        state_sync["errors"],
                        trash=trash,
                        deleted_paths=list(state_sync["deleted"]),
                        cancelled=True,
                    ),
                )
                return
            n = len(paths)
            end = min(state_sync["i"] + batch, n)
            for j in range(state_sync["i"], end):
                p = paths[j]
                try:
                    if p.is_file():
                        if trash:
                            send2trash.send2trash(str(p))
                        else:
                            p.unlink()
                        state_sync["deleted"].append(p)
                except OSError as e:
                    state_sync["errors"].append(f"{p}: {e!s}")
                except Exception as e:
                    if trash:
                        state_sync["errors"].append(f"{p}: {e!s}")
            state_sync["i"] = end
            if n:
                bar.set(end / n)
            lbl_count.configure(text=ui_t("dup.delete.progress_count", done=end, total=n))
            if end > 0:
                lbl_file.configure(text=ui_t("dup.delete.progress_file", name=self._short_delete_filename(paths[end - 1])))
            if end >= n:
                try:
                    prog_win.destroy()
                except Exception:
                    pass
                errs = list(state_sync["errors"])
                deleted = list(state_sync["deleted"])
                self.after(
                    0,
                    lambda: self._finalize_delete_outcome(
                        paths, errs, trash=trash, deleted_paths=deleted, cancelled=False
                    ),
                )
            else:
                self.after(1, step)

        self.after(0, step)

    def _append_journal_entry(self, paths: list[Path], *, trash: bool) -> None:
        entry = {
            "ts": int(time.time()),
            "mode": "trash" if trash else "permanent",
            "count": len(paths),
            "paths": [str(p) for p in paths],
        }
        p = duplicate_journal_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _show_undo_log(self) -> None:
        p = duplicate_journal_path()
        if not p.is_file():
            messagebox.showinfo(ui_t("dup.journal.title"), ui_t("dup.journal.empty"))
            return
        lines = p.read_text(encoding="utf-8").splitlines()[-200:]
        top = ctk.CTkToplevel(self)
        top.title(ui_t("dup.journal.title"))
        top.geometry("760x520")
        self._make_modal(top)
        tb = ctk.CTkTextbox(top)
        tb.pack(fill="both", expand=True, padx=8, pady=8)
        tb.insert("1.0", "\n".join(lines) if lines else "Журнал пуст.")
        tb.configure(state="disabled")

    def _refresh_virtualized_groups(self) -> None:
        if not self._group_records:
            self._filtered_indices = []
            self._update_groups_summary()
            self._refresh_delete_count_and_review_progress()
            self._notify_viewer_filters_changed()
            try:
                self._btn_open_viewer.configure(state="disabled")
            except Exception:
                pass
            return
        self._filtered_indices = self._build_filtered_indices()
        self._update_groups_summary()
        self._refresh_delete_count_and_review_progress()
        self._notify_viewer_filters_changed()
        try:
            self._btn_open_viewer.configure(state="normal")
        except Exception:
            pass

    def _build_filtered_indices(self) -> list[int]:
        f_type = self._filter_type_key.get()
        f_skip = self._filter_skip_key.get()
        try:
            min_size = max(2, int(self._filter_min_size_var.get()))
        except ValueError:
            min_size = 2
        idxs = []
        for i, g in enumerate(self._group_records):
            size_ok = len(g["paths"]) >= min_size
            if not size_ok:
                continue
            if f_type == "exact" and not g.get("is_exact"):
                continue
            if f_type == "similar" and g.get("is_exact"):
                continue
            skip_val = self._group_skip_vars[i].get() if i < len(self._group_skip_vars) else False
            if f_skip == "skip" and not skip_val:
                continue
            if f_skip == "active" and skip_val:
                continue
            idxs.append(i)
        s = self._sort_key.get()
        if s == "size_desc":
            idxs.sort(key=lambda i: len(self._group_records[i]["paths"]), reverse=True)
        elif s == "size_asc":
            idxs.sort(key=lambda i: len(self._group_records[i]["paths"]))
        elif s == "name":
            idxs.sort(key=lambda i: (self._group_records[i]["paths"][0].lower() if self._group_records[i]["paths"] else ""))
        elif s == "type":
            idxs.sort(key=lambda i: (not self._group_records[i].get("is_exact"), -len(self._group_records[i]["paths"])))
        return idxs

    def _open_keep_list_modal(self, gi: int) -> None:
        if gi < 0 or gi >= len(self._group_path_lists):
            return
        plist = self._group_path_lists[gi]
        km = self._group_keep_check_vars[gi]
        skipped = self._group_skip_vars[gi].get()
        del_ent = gi < len(self._group_delete_entire_vars) and self._group_delete_entire_vars[gi].get()
        try:
            ord_n = self._filtered_indices.index(gi) + 1
        except ValueError:
            ord_n = gi + 1
        total = len(plist)
        total_pages = max(1, (total + DUP_CARD_MODAL_PAGE - 1) // DUP_CARD_MODAL_PAGE)
        top = ctk.CTkToplevel(self)
        top.title(ui_t("dup.card.modal.title", n=ord_n))
        top.geometry("720x560")
        self._make_modal(top)
        page_holder: list[int] = [0]
        info_lbl = ctk.CTkLabel(top, text="", anchor="w")
        info_lbl.pack(fill="x", padx=10, pady=(8, 4))
        scroll = ctk.CTkScrollableFrame(top, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=8, pady=4)
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=8)
        btn_prev = ctk.CTkButton(btn_row, text=ui_t("dup.card.modal.prev"), width=100)
        btn_prev.pack(side="left", padx=(0, 8))
        btn_next = ctk.CTkButton(btn_row, text=ui_t("dup.card.modal.next"), width=100)
        btn_next.pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text=ui_t("dup.card.modal.close"), command=top.destroy).pack(side="right")

        def rebuild() -> None:
            for w in scroll.winfo_children():
                w.destroy()
            pg = page_holder[0]
            start = pg * DUP_CARD_MODAL_PAGE
            end = min(total, start + DUP_CARD_MODAL_PAGE)
            info_lbl.configure(
                text=ui_t(
                    "dup.card.modal.page_info",
                    page=pg + 1,
                    pages=total_pages,
                    a=start + 1,
                    b=end,
                    total=total,
                )
            )
            for idx in range(start, end):
                p = plist[idx]
                row = ctk.CTkFrame(scroll, fg_color=("gray92", "gray17"))
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=p, anchor="w", wraplength=480).pack(side="left", fill="x", expand=True, padx=4)
                ctk.CTkCheckBox(
                    row,
                    text=ui_t("dup.card.keep"),
                    variable=km[p],
                    command=self._after_keep_change,
                    state=("disabled" if skipped or del_ent else "normal"),
                ).pack(side="right")
            btn_prev.configure(state=("disabled" if pg <= 0 else "normal"))
            btn_next.configure(state=("disabled" if pg >= total_pages - 1 else "normal"))

        def go_prev() -> None:
            page_holder[0] = max(0, page_holder[0] - 1)
            rebuild()

        def go_next() -> None:
            page_holder[0] = min(total_pages - 1, page_holder[0] + 1)
            rebuild()

        btn_prev.configure(command=go_prev)
        btn_next.configure(command=go_next)
        rebuild()

    def _bulk_skip_small(self) -> None:
        try:
            min_size = max(2, int(self._filter_min_size_var.get()))
        except ValueError:
            min_size = 3
        for i, g in enumerate(self._group_records):
            if len(g["paths"]) < min_size:
                self._group_skip_vars[i].set(True)
        self.flush_persist_review_state()
        self._refresh_delete_count_and_review_progress()
        self._refresh_virtualized_groups()

    def _bulk_unskip_all(self) -> None:
        for v in self._group_skip_vars:
            v.set(False)
        self.flush_persist_review_state()
        self._refresh_delete_count_and_review_progress()
        self._refresh_virtualized_groups()

    def _persist_review_state_impl(self) -> None:
        if not self._session_key:
            return
        payload = self._sig_db.session_payload(self._session_key)
        review_state = {
            "groups": {
                gid: {
                    "skip": self._group_skip_vars[i].get(),
                    "keep_paths": [p for p in self._group_path_lists[i] if self._group_keep_check_vars[i][p].get()],
                    "approved": self._group_approved_vars[i].get() if i < len(self._group_approved_vars) else False,
                    "delete_entire": self._group_delete_entire_vars[i].get() if i < len(self._group_delete_entire_vars) else False,
                }
                for i, gid in enumerate(self._group_ids)
            }
        }
        payload["review_state"] = review_state
        self._sig_db.update_session_payload(self._session_key, payload)

    def _persist_review_state_callback(self) -> None:
        self._persist_after_id = None
        self._persist_review_state_impl()

    def _persist_review_state(self) -> None:
        if not self._session_key:
            return
        if self._persist_after_id is not None:
            try:
                self.after_cancel(self._persist_after_id)
            except Exception:
                pass
        self._persist_after_id = self.after(PERSIST_DEBOUNCE_MS, self._persist_review_state_callback)

    def flush_persist_review_state(self) -> None:
        if self._persist_after_id is not None:
            try:
                self.after_cancel(self._persist_after_id)
            except Exception:
                pass
            self._persist_after_id = None
        self._persist_review_state_impl()

    def _restore_review_state(self) -> None:
        if not self._session_key:
            return
        payload = self._sig_db.session_payload(self._session_key)
        rs = payload.get("review_state", {})
        groups_state = rs.get("groups", {}) if isinstance(rs, dict) else {}
        if not isinstance(groups_state, dict):
            return
        for i, gid in enumerate(self._group_ids):
            st = groups_state.get(gid)
            if not isinstance(st, dict):
                continue
            self._group_skip_vars[i].set(bool(st.get("skip", False)))
            if i < len(self._group_approved_vars):
                self._group_approved_vars[i].set(bool(st.get("approved", st.get("reviewed", False))))
            if i < len(self._group_delete_entire_vars):
                self._group_delete_entire_vars[i].set(bool(st.get("delete_entire", False)))
            plist = self._group_path_lists[i]
            km = self._group_keep_check_vars[i]
            kp_list = st.get("keep_paths")
            if isinstance(kp_list, list) and kp_list:
                keep_set = {str(x) for x in kp_list}
                for p in plist:
                    if p in km:
                        km[p].set(any(_path_same(p, k) for k in keep_set))
            else:
                kp = str(st.get("keep_path", "") or "")
                for p in plist:
                    if p in km:
                        km[p].set(bool(kp and _path_same(p, kp)))
            if plist and not any(km[p].get() for p in plist):
                km[plist[0]].set(True)

    def _on_hotkey_s(self, _event=None) -> None:
        if self.focus_get() is None:
            return
        if not self._filtered_indices:
            return
        idx = self._filtered_indices[0]
        self._group_skip_vars[idx].set(not self._group_skip_vars[idx].get())
        self._persist_review_state()
        self._refresh_delete_count_and_review_progress()
        self.after(0, self._refresh_virtualized_groups)

    def _on_hotkey_a(self, _event=None) -> None:
        if self.focus_get() is None:
            return
        self._apply_suggested_deletes()

    def _on_hotkey_d(self, _event=None) -> None:
        if self.focus_get() is None:
            return
        self._preview_delete()

    def _on_hotkey_j(self, _event=None) -> None:
        if self.focus_get() is None:
            return
        self._open_groups_viewer()

    def _on_hotkey_k(self, _event=None) -> None:
        # Navigation up is implicit in native scroll; keep key for parity.
        return

    def _approve_all_groups(self) -> None:
        for v in self._group_approved_vars:
            v.set(True)
        self._persist_review_state()
        self._refresh_delete_count_and_review_progress()
        self.after(0, self._refresh_virtualized_groups)

    def _compare_group(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._group_records):
            return
        paths = [Path(p) for p in self._group_records[idx]["paths"]]
        if len(paths) < 2:
            messagebox.showinfo(ui_t("dup.compare.title"), ui_t("dup.compare.one_file"))
            return
        plist = self._group_path_lists[idx]
        km = self._group_keep_check_vars[idx]
        kept = [Path(p) for p in plist if km[p].get()]
        rest = [Path(p) for p in plist if not km[p].get()]
        ordered: list[Path] = kept + rest
        if not ordered:
            ordered = list(paths)
        top = ctk.CTkToplevel(self)
        top.title(ui_t("dup.compare.title"))
        top.geometry("1024x520")
        self._make_modal(top)
        action_row = ctk.CTkFrame(top, fg_color="transparent")
        action_row.pack(fill="x", padx=10, pady=(8, 6))
        ctk.CTkLabel(
            action_row,
            text="Визуальное сравнение файлов группы",
            text_color=("gray35", "gray65"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            action_row,
            text=ui_t("dup.compare.approve_all"),
            width=160,
            fg_color=CTK_ACCENT_PRIMARY,
            hover_color=CTK_ACCENT_PRIMARY_HOVER,
            command=self._approve_all_groups,
        ).pack(side="right")
        scroll = ctk.CTkScrollableFrame(top, orientation="horizontal", height=400, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        max_side = min(320, max(160, 900 // max(1, len(ordered))))
        for p in ordered:
            cell = ctk.CTkFrame(scroll, fg_color=("gray90", "gray16"))
            cell.pack(side="left", fill="y", padx=6, pady=4)
            img = _thumb_ctk(p, size=(max_side, max_side))
            if img is not None:
                lbl = ctk.CTkLabel(cell, text="", image=img)
                lbl.image = img
                lbl.pack(pady=(8, 4))
            else:
                suf = p.suffix.lower()
                txt = ui_t("dup.thumb.video_no_frame") if suf in VIDEO_EXTENSIONS else ui_t("dup.thumb.no_preview")
                ctk.CTkLabel(cell, text=txt).pack(pady=(8, 4))
            ctk.CTkLabel(cell, text=p.name, wraplength=max_side + 40).pack(pady=(0, 8))

    def _open_duplicate_review(self) -> None:
        from tkinter import filedialog

        from app.gui_dup_review import open_duplicate_review_window

        if self._last_export_dir:
            p = self._last_export_dir / "result.json"
            if p.is_file():
                open_duplicate_review_window(self.winfo_toplevel(), p)
                return
        f = filedialog.askopenfilename(
            title=ui_t("dup.review.pick_file"),
            filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")],
        )
        if f:
            open_duplicate_review_window(self.winfo_toplevel(), Path(f))

    def _save_profile(self) -> None:
        key = self._preset_key_from_combo()
        profile_name = f"profile_{key}"
        self._profiles[profile_name] = self.dup_settings_dict()
        self._app._save_gui_settings()
        messagebox.showinfo(ui_t("dup.profile.dialog_save"), ui_t("dup.profile.saved", name=profile_name))

    def _load_profile(self) -> None:
        if not self._profiles:
            messagebox.showinfo(ui_t("dup.profile.dialog_load"), ui_t("dup.profile.none"))
            return
        name = sorted(self._profiles.keys())[0]
        prof = self._profiles.get(name, {})
        if not isinstance(prof, dict):
            return
        mm = str(prof.get("media_mode", ""))
        if mm in {m.value for m in MediaScanMode}:
            self._media_mode_var.set(mm)
        self._workers_var.set(str(max(1, min(4, int(prof.get("parallel_workers", 3) or 3)))))
        self._force_recompute_var.set(bool(prof.get("force_recompute", False)))
        self._llm_var.set(bool(prof.get("llm_enabled", False)))
        preset = str(prof.get("preset", "") or "")
        keys = list(ACCURACY_PRESET_KEYS)
        labels = [ui_t(f"dup.accuracy.{k}") for k in ACCURACY_PRESET_KEYS]
        if preset in keys:
            self._preset_combo.set(labels[keys.index(preset)])
        if str(prof.get("review_filter_type", "")) in {"all", "exact", "similar"}:
            self._filter_type_key.set(str(prof.get("review_filter_type")))
        if str(prof.get("review_filter_skip", "")) in {"all", "skip", "active"}:
            self._filter_skip_key.set(str(prof.get("review_filter_skip")))
        if str(prof.get("review_sort", "")) in {"size_desc", "size_asc", "name", "type"}:
            self._sort_key.set(str(prof.get("review_sort")))
        if str(prof.get("review_filter_min_size", "")) in {"2", "3", "5", "10"}:
            self._filter_min_size_var.set(str(prof.get("review_filter_min_size")))
        self._sync_filter_sort_combos()
        self._on_preset_change()
        messagebox.showinfo(ui_t("dup.profile.dialog_load"), ui_t("dup.profile.loaded", name=name))

    def on_app_close(self) -> None:
        if self._dup_worker:
            self._dup_worker.request_stop()
        self.flush_persist_review_state()
        vw = self._viewer_window
        if vw is not None:
            try:
                vw.destroy()
            except Exception:
                pass
        self._sig_db.close()

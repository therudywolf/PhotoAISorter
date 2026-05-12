# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Отдельное окно: крупный просмотр групп дубликатов, утверждение и удаление всей группы."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.constants import VIDEO_EXTENSIONS
from app.dup_thumbs import thumb_ctk
from app.dup_ui_constants import CTK_ACCENT_PRIMARY, CTK_ACCENT_PRIMARY_HOVER
from app.ui_texts import t as ui_t

if TYPE_CHECKING:
    from app.gui_duplicates import DuplicatesPane

VIEWER_THUMB_SIZE = (240, 240)
_CACHE_LIMIT = 256

# Локальные bind на окно и виджеты (без bind_all / unbind_all по приложению).
_VIEWER_KEY_SEQUENCES = (
    "<Left>",
    "<Right>",
    "<Up>",
    "<Down>",
    "<KP_Left>",
    "<KP_Right>",
    "<KP_Up>",
    "<KP_Down>",
    "<space>",
)


class DuplicateGroupsViewer(ctk.CTkToplevel):
    def __init__(self, master: Any, pane: "DuplicatesPane") -> None:
        super().__init__(master)
        self._pane = pane
        self._fp = 0
        self._thumb_gen = 0
        self._thumb_queue: list[tuple[int, ctk.CTkLabel, Path, int]] = []
        self._thumb_cache: OrderedDict[str, ctk.CTkImage] = OrderedDict()
        self._thumb_lock = threading.Lock()
        self._thumb_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dup-viewer-th")
        self._scroll_body: ctk.CTkScrollableFrame | None = None

        pane._viewer_window = self
        self.title(ui_t("dup.viewer.title"))
        self.configure(fg_color=("gray92", "gray14"))
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            self.geometry(f"{min(1200, sw - 40)}x{min(900, sh - 80)}+20+20")
        except Exception:
            self.geometry("1100x820")
        self.minsize(880, 560)

        self._build()
        self._wire_viewer_static_keys()
        self._clamp_fp()
        self._redraw()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<KeyPress-bracketleft>", lambda _e: self._nav_prev())
        self.bind("<KeyPress-bracketright>", lambda _e: self._nav_next())
        self.after(80, self._focus_viewer)

    def _focus_viewer(self) -> None:
        try:
            if self.winfo_exists():
                self.lift()
                self.focus_force()
                self.focus_set()
        except Exception:
            pass

    def _focus_in_number_entry(self) -> bool:
        fw = self.focus_get()
        if fw is None:
            return False
        if fw == self._entry_go:
            return True
        inner = getattr(self._entry_go, "_entry", None)
        if inner is not None and fw == inner:
            return True
        return False

    def _widget_in_checkbox_tree(self, widget: Any) -> bool:
        w: Any = widget
        for _ in range(40):
            if w is None:
                break
            try:
                if isinstance(w, ctk.CTkCheckBox):
                    return True
            except Exception:
                pass
            w = getattr(w, "master", None)
        return False

    def _focus_in_checkbox_tree(self) -> bool:
        fw = self.focus_get()
        if fw is None:
            return False
        return self._widget_in_checkbox_tree(fw)

    def _bind_keys_to_widget(self, w: Any) -> None:
        for seq in _VIEWER_KEY_SEQUENCES:
            w.bind(seq, self._on_viewer_key)

    def _bind_keys_recursive(self, w: Any) -> None:
        self._bind_keys_to_widget(w)
        for c in w.winfo_children():
            self._bind_keys_recursive(c)

    def _wire_viewer_static_keys(self) -> None:
        self._bind_keys_to_widget(self)
        self._bind_keys_recursive(self._nav_frame)
        self._bind_keys_to_widget(self._lbl_viewer_hotkeys)
        self._bind_keys_to_widget(self._sep_viewer_nav)

    def _bind_scroll_area_keys(self, inner: ctk.CTkScrollableFrame | None) -> None:
        if self._scroll_body is None:
            return
        self._bind_keys_recursive(self._scroll_body)
        pc = getattr(self._scroll_body, "_parent_canvas", None)
        if pc is not None:
            self._bind_keys_to_widget(pc)
        if inner is not None:
            ipc = getattr(inner, "_parent_canvas", None)
            if ipc is not None:
                self._bind_keys_to_widget(ipc)

    def _on_viewer_key(self, event: Any) -> str | None:
        try:
            if not self.winfo_exists():
                return None
            if event.widget.winfo_toplevel() != self:
                return None
        except Exception:
            return None

        if self._focus_in_number_entry():
            return None

        ks = getattr(event, "keysym", "") or ""

        if ks in ("Left", "Up", "KP_Left", "KP_Up"):
            self._nav_prev()
            return "break"
        if ks in ("Right", "Down", "KP_Right", "KP_Down"):
            self._nav_next()
            return "break"
        if ks in ("space", "Space"):
            if self._widget_in_checkbox_tree(event.widget) or self._focus_in_checkbox_tree():
                return None
            self._toggle_approve_current()
            return "break"
        return None

    def _on_close(self) -> None:
        self._thumb_gen += 1
        try:
            self._thumb_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._pane._viewer_window = None
        self.destroy()

    def _build(self) -> None:
        self._nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav = self._nav_frame
        nav.pack(fill="x", padx=16, pady=(14, 6))
        self._lbl_viewer_hotkeys = ctk.CTkLabel(
            self,
            text=ui_t("dup.viewer.hotkeys"),
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray38", "gray62"),
            wraplength=900,
        )
        self._lbl_viewer_hotkeys.pack(fill="x", padx=18, pady=(0, 4))
        self._sep_viewer_nav = ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray28"))
        self._sep_viewer_nav.pack(fill="x", padx=16, pady=(0, 8))

        self._btn_prev = ctk.CTkButton(
            nav,
            text=ui_t("dup.nav.prev"),
            width=108,
            height=36,
            corner_radius=8,
            fg_color=("gray75", "gray32"),
            command=self._nav_prev,
        )
        self._btn_prev.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(nav, text=ui_t("dup.nav.label")).pack(side="left", padx=(0, 6))
        self._entry_go = ctk.CTkEntry(nav, width=80, height=36, corner_radius=8, placeholder_text="1")
        self._entry_go.pack(side="left", padx=(0, 8))
        ctk.CTkButton(nav, text=ui_t("dup.nav.go"), width=92, height=36, corner_radius=8, command=self._nav_go).pack(
            side="left", padx=(0, 10)
        )
        self._btn_next = ctk.CTkButton(
            nav,
            text=ui_t("dup.nav.next"),
            width=108,
            height=36,
            corner_radius=8,
            fg_color=CTK_ACCENT_PRIMARY,
            hover_color=CTK_ACCENT_PRIMARY_HOVER,
            command=self._nav_next,
        )
        self._btn_next.pack(side="left", padx=(0, 12))
        self._lbl_pos = ctk.CTkLabel(nav, text="", anchor="w", text_color=("gray22", "gray78"))
        self._lbl_pos.pack(side="left", fill="x", expand=True)
        self._entry_go.bind("<Return>", lambda _e: self._nav_go())

        self._scroll_body = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self._scroll_body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _toggle_approve_current(self) -> None:
        filt = self._pane._filtered_indices
        if not filt:
            return
        gi = filt[self._fp]
        v = self._pane._group_approved_vars[gi]
        v.set(not v.get())
        self._pane._persist_review_state()
        self._pane._refresh_delete_count_and_review_progress()

    def _clamp_fp(self) -> None:
        filt = self._pane._filtered_indices
        if not filt:
            self._fp = 0
            return
        self._fp = max(0, min(self._fp, len(filt) - 1))

    def refresh_safe(self) -> None:
        self._clamp_fp()
        self._redraw()

    def _nav_prev(self) -> None:
        if self._fp <= 0:
            return
        self._fp -= 1
        self._redraw()

    def _nav_next(self) -> None:
        filt = self._pane._filtered_indices
        if not filt or self._fp >= len(filt) - 1:
            return
        self._fp += 1
        self._redraw()

    def _nav_go(self) -> None:
        filt = self._pane._filtered_indices
        if not filt:
            return
        raw = self._entry_go.get().strip()
        try:
            n = int(raw)
        except ValueError:
            return
        if n < 1 or n > len(filt):
            return
        self._fp = n - 1
        self._redraw()

    def _update_nav_labels(self) -> None:
        filt = self._pane._filtered_indices
        total = len(filt)
        if total == 0:
            self._lbl_pos.configure(text="")
            self._btn_prev.configure(state="disabled")
            self._btn_next.configure(state="disabled")
            return
        cur = self._fp + 1
        self._lbl_pos.configure(text=ui_t("dup.nav.hint", cur=cur, total=total))
        self._btn_prev.configure(state="normal" if self._fp > 0 else "disabled")
        self._btn_next.configure(state="normal" if self._fp < total - 1 else "disabled")
        self._entry_go.delete(0, "end")
        self._entry_go.insert(0, str(cur))

    def _touch_cache(self, key: str, img: ctk.CTkImage) -> None:
        with self._thumb_lock:
            self._thumb_cache[key] = img
            self._thumb_cache.move_to_end(key)
            while len(self._thumb_cache) > _CACHE_LIMIT:
                self._thumb_cache.popitem(last=False)

    def _queue_thumb(self, lbl: ctk.CTkLabel, path: Path) -> None:
        self._thumb_queue.append((self._thumb_gen, lbl, path, 0))
        self.after(1, self._drain_thumb_queue)

    def _drain_thumb_queue(self) -> None:
        if not self._thumb_queue:
            return
        gen = self._thumb_gen
        batch = 5
        while self._thumb_queue and batch > 0:
            item_gen, img_label, path, _ = self._thumb_queue.pop(0)
            if item_gen != gen or not img_label.winfo_exists():
                continue
            key = str(path)
            with self._thumb_lock:
                img = self._thumb_cache.get(key)
                if img is not None:
                    self._thumb_cache.move_to_end(key)
            if img is not None:
                img_label.configure(text="", image=img)
                img_label.image = img
            else:
                loading = ui_t("dup.thumb.ffmpeg_try") if path.suffix.lower() in VIDEO_EXTENSIONS else ui_t("dup.thumb.loading")
                img_label.configure(text=loading)

                def task(p: Path = path, k: str = key, lbl: ctk.CTkLabel = img_label) -> None:
                    def vlog(_m: str) -> None:
                        pass

                    created = thumb_ctk(p, size=VIEWER_THUMB_SIZE, on_video_log=vlog)

                    def apply() -> None:
                        if item_gen != self._thumb_gen or not lbl.winfo_exists():
                            return
                        if created is not None:
                            self._touch_cache(k, created)
                            lbl.configure(text="", image=created)
                            lbl.image = created
                        else:
                            lbl.configure(text=ui_t("dup.thumb.no_preview"))

                    self.after(0, apply)

                self._thumb_pool.submit(task)
            batch -= 1
        if self._thumb_queue:
            self.after(16, self._drain_thumb_queue)

    def _redraw(self) -> None:
        self._thumb_gen += 1
        self._thumb_queue.clear()
        if self._scroll_body is None:
            return
        for w in self._scroll_body.winfo_children():
            w.destroy()

        filt = self._pane._filtered_indices
        self._update_nav_labels()
        if not filt:
            ctk.CTkLabel(self._scroll_body, text=ui_t("dup.viewer.empty"), font=ctk.CTkFont(size=14)).pack(pady=48)
            self._bind_scroll_area_keys(None)
            return

        gi = filt[self._fp]
        g = self._pane._group_records[gi]
        paths = [Path(p) for p in g["paths"]]
        n_paths = len(paths)
        km = self._pane._group_keep_check_vars[gi]
        skip_var = self._pane._group_skip_vars[gi]
        appr_var = self._pane._group_approved_vars[gi]
        del_all_var = self._pane._group_delete_entire_vars[gi]

        title = ui_t("dup.card.group_title", n=self._fp + 1, count=n_paths)
        if g.get("is_exact"):
            title += ui_t("dup.card.exact_suffix")
        ctk.CTkLabel(self._scroll_body, text=title, font=ctk.CTkFont(size=19, weight="bold")).pack(anchor="w", padx=4, pady=(6, 10))

        row = ctk.CTkFrame(self._scroll_body, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))
        ctk.CTkCheckBox(
            row,
            text=ui_t("dup.card.approve"),
            variable=appr_var,
            command=lambda: (self._pane._persist_review_state(), self._pane._refresh_delete_count_and_review_progress()),
        ).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(
            row,
            text=ui_t("dup.card.skip"),
            variable=skip_var,
            command=lambda: (self._pane._persist_review_state(), self._pane._refresh_delete_count_and_review_progress(), self._on_skip_filter()),
        ).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(
            row,
            text=ui_t("dup.card.delete_entire"),
            variable=del_all_var,
            command=lambda: (self._pane._persist_review_state(), self._pane._refresh_delete_count_and_review_progress(), self._redraw()),
        ).pack(side="left", padx=(0, 12))
        ctk.CTkButton(
            row,
            text=ui_t("dup.card.compare"),
            width=168,
            height=34,
            corner_radius=8,
            fg_color=("gray72", "gray34"),
            command=lambda: self._pane._compare_group(gi),
        ).pack(side="left", padx=(8, 0))
        if n_paths > 48:
            ctk.CTkButton(
                row,
                text=ui_t("dup.card.open_full_list"),
                width=188,
                height=34,
                corner_radius=8,
                fg_color=("gray72", "gray34"),
                command=lambda: self._pane._open_keep_list_modal(gi),
            ).pack(side="left", padx=(8, 0))

        skipped = skip_var.get()
        del_entire = del_all_var.get()
        keep_ok = not skipped and not del_entire

        inner = ctk.CTkScrollableFrame(
            self._scroll_body,
            orientation="horizontal",
            height=VIEWER_THUMB_SIZE[1] + 128,
            fg_color=("gray90", "gray18"),
            corner_radius=12,
        )
        inner.pack(fill="x", pady=(4, 8))

        for p in paths:
            ps = str(p)
            cell = ctk.CTkFrame(inner, fg_color=("gray88", "gray20"), corner_radius=10)
            cell.pack(side="left", padx=10, pady=12)
            img_label = ctk.CTkLabel(
                cell,
                text=ui_t("dup.thumb.loading"),
                width=VIEWER_THUMB_SIZE[0] + 8,
                height=VIEWER_THUMB_SIZE[1] + 8,
            )
            img_label.pack(pady=(8, 4))
            if not skipped:
                self._queue_thumb(img_label, p)
            else:
                img_label.configure(text="—")
            ctk.CTkLabel(cell, text=p.name, wraplength=VIEWER_THUMB_SIZE[0] + 40).pack()
            ctk.CTkCheckBox(
                cell,
                text=ui_t("dup.card.keep"),
                variable=km[ps],
                command=self._pane._after_keep_change,
                state=("normal" if keep_ok else "disabled"),
            ).pack(pady=(0, 8))

        self._bind_scroll_area_keys(inner)

    def _on_skip_filter(self) -> None:
        if self._pane._filter_skip_key.get() in ("skip", "active"):
            self._pane._refresh_virtualized_groups()
            self.refresh_safe()

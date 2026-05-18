# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Dialog to manage CLIP exemplar photos per tag (drag-and-drop on Windows)."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter.filedialog as filedialog
from pathlib import Path

import customtkinter as ctk

from app.constants import STILL_IMAGE_EXTENSIONS
from app.context_tags import build_custom_categories, get_active_set, load_tag_store
from app.fast_classify.exemplar_files import add_exemplar_files, remove_exemplar_file
from app.fast_classify.exemplars import ensure_refs_layout, list_exemplar_paths, refs_dir
from app.fast_classify.registry import clear_classifier_cache


class ExemplarsDialog(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTkBaseClass, *, on_log: None | callable = None) -> None:
        super().__init__(parent)
        self.title("Эталоны для быстрой CLIP")
        self.geometry("640x520")
        self.resizable(True, True)
        self.transient(parent)
        self._on_log = on_log
        self._tag_var = ctk.StringVar(value="")
        self._build()
        self._reload_tags()
        self.after(150, self._setup_drag_drop)
        self.after(100, self.lift)

    def _build(self) -> None:
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(top, text="Тег:", width=40).pack(side="left")
        self._tag_combo = ctk.CTkComboBox(
            top,
            values=["(нет тегов)"],
            variable=self._tag_var,
            width=220,
            state="readonly",
            command=lambda _v: self._refresh_file_list(),
        )
        self._tag_combo.pack(side="left", padx=(4, 8))
        ctk.CTkButton(top, text="Открыть папку", width=120, command=self._open_tag_folder).pack(
            side="left"
        )

        self._hint = ctk.CTkLabel(
            self,
            text=(
                "Эталоны — примеры фото для каждого тега (5–20 шт.). "
                "Перетащите файлы в зону ниже или нажмите «Добавить фото…»."
            ),
            wraplength=600,
            justify="left",
            text_color=("gray35", "gray65"),
            font=ctk.CTkFont(size=11),
        )
        self._hint.pack(anchor="w", padx=12, pady=(0, 6))

        self._drop_frame = ctk.CTkFrame(self, height=88, corner_radius=8, fg_color=("gray88", "gray22"))
        self._drop_frame.pack(fill="x", padx=12, pady=(0, 8))
        self._drop_frame.pack_propagate(False)
        self._drop_label = ctk.CTkLabel(
            self._drop_frame,
            text="Перетащите сюда JPEG / PNG / WEBP\n(Windows) или используйте кнопку ниже",
            justify="center",
        )
        self._drop_label.pack(expand=True)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkButton(btn_row, text="Добавить фото…", command=self._add_files_dialog).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(
            btn_row,
            text="Удалить выбранные",
            fg_color=("gray75", "gray35"),
            command=self._remove_selected,
        ).pack(side="left", padx=(0, 8))
        self._count_label = ctk.CTkLabel(btn_row, text="", anchor="e")
        self._count_label.pack(side="right", fill="x", expand=True)

        self._list = ctk.CTkScrollableFrame(self, height=240)
        self._list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self._file_vars: list[tuple[str, ctk.BooleanVar]] = []

        ctk.CTkButton(self, text="Закрыть", width=120, command=self.destroy).pack(pady=(0, 12))

    def _reload_tags(self) -> None:
        store = load_tag_store()
        active = get_active_set(store)
        tags = [t for t in build_custom_categories(active) if t != "uncategorized"] if active else []
        root = ensure_refs_layout(extra_tags=tags)
        if self._on_log:
            self._on_log(f"Эталоны хранятся в {root} (папка data/ проекта, не в git)")
        if not tags:
            self._tag_combo.configure(values=["(нет тегов)"])
            self._tag_var.set("(нет тегов)")
            return
        self._tag_combo.configure(values=tags)
        if self._tag_var.get() not in tags:
            self._tag_var.set(tags[0])
        self._refresh_file_list()

    def _current_tag(self) -> str | None:
        t = self._tag_var.get().strip()
        if not t or t.startswith("("):
            return None
        return t

    def _refresh_file_list(self) -> None:
        for w in self._list.winfo_children():
            w.destroy()
        self._file_vars.clear()
        tag = self._current_tag()
        if not tag:
            self._count_label.configure(text="")
            return
        paths = list_exemplar_paths(tag, limit=48)
        for p in paths:
            var = ctk.BooleanVar(value=False)
            self._file_vars.append((p.name, var))
            row = ctk.CTkFrame(self._list, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkCheckBox(row, text=p.name, variable=var).pack(side="left", anchor="w")
        self._count_label.configure(text=f"{len(paths)} / 48 файлов")

    def _setup_drag_drop(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import windnd  # type: ignore[import-untyped]

            hwnd = self.winfo_id()
            windnd.hook_dropfiles(
                hwnd,
                func=lambda files: self.after(0, lambda: self._on_drop_files(files)),
            )
            self._drop_label.configure(
                text="Перетащите сюда JPEG / PNG / WEBP\n(отпустите для добавления в выбранный тег)"
            )
        except Exception:
            pass

    def _on_drop_files(self, files: list[bytes] | tuple[bytes, ...]) -> None:
        paths: list[Path] = []
        for raw in files:
            try:
                p = Path(os.fsdecode(raw)).resolve()
            except (TypeError, ValueError):
                continue
            if p.is_file():
                paths.append(p)
        if paths:
            self._import_paths(paths)

    def _add_files_dialog(self) -> None:
        tag = self._current_tag()
        if not tag:
            return
        paths = filedialog.askopenfilenames(
            parent=self,
            title=f"Эталоны для «{tag}»",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"),
                ("Все файлы", "*.*"),
            ],
        )
        if paths:
            self._import_paths([Path(p) for p in paths])

    def _import_paths(self, paths: list[Path]) -> None:
        tag = self._current_tag()
        if not tag:
            return
        valid = [p for p in paths if p.suffix.lower() in STILL_IMAGE_EXTENSIONS]
        if not valid:
            return
        n = add_exemplar_files(tag, valid)
        if n:
            clear_classifier_cache()
            if self._on_log:
                self._on_log(f"Эталоны «{tag}»: добавлено {n} фото")
        self._refresh_file_list()

    def _remove_selected(self) -> None:
        tag = self._current_tag()
        if not tag:
            return
        removed = 0
        for name, var in self._file_vars:
            if var.get():
                if remove_exemplar_file(tag, name):
                    removed += 1
        if removed:
            clear_classifier_cache()
            if self._on_log:
                self._on_log(f"Эталоны «{tag}»: удалено {removed}")
        self._refresh_file_list()

    def _open_tag_folder(self) -> None:
        tag = self._current_tag()
        if not tag:
            return
        folder = refs_dir() / tag
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except OSError as e:
            if self._on_log:
                self._on_log(f"Не удалось открыть папку: {e}")

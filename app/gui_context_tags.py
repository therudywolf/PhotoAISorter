"""Dialog for managing context tags and custom category lists."""

from __future__ import annotations

import tkinter.messagebox as messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from app.context_tags import (
    ContextTag,
    ContextTagStore,
    CustomCategoryList,
    load_context_tags,
    save_context_tags,
)

if TYPE_CHECKING:
    pass


class ContextTagsDialog(ctk.CTkToplevel):
    """Manager window for user-defined context tags."""

    def __init__(self, parent: ctk.CTkBaseClass, on_save: None | callable = None):
        super().__init__(parent)
        self.title("Контекстные теги")
        self.geometry("680x560")
        self.resizable(True, True)
        self._on_save_callback = on_save
        self._store = load_context_tags()
        self._build()
        self.after(100, self.lift)

    def _build(self) -> None:
        notebook = ctk.CTkTabview(self, anchor="nw")
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        notebook.add("Теги")
        notebook.add("Списки категорий")

        self._build_tags_tab(notebook.tab("Теги"))
        self._build_lists_tab(notebook.tab("Списки категорий"))

    def _build_tags_tab(self, parent: ctk.CTkFrame) -> None:
        hint = ctk.CTkLabel(
            parent,
            text=(
                "Контекстные теги — именованные описания для распознавания ваших персональных объектов.\n"
                "Например: ключ 'my_dog', описание 'Чёрный лабрадор, кличка Рекс'.\n"
                "Включённые теги автоматически передаются модели при классификации."
            ),
            wraplength=620,
            justify="left",
            text_color=("gray38", "gray62"),
        )
        hint.pack(anchor="w", padx=8, pady=(4, 8))

        self._tags_frame = ctk.CTkScrollableFrame(parent, height=300)
        self._tags_frame.pack(fill="both", expand=True, padx=4, pady=4)

        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(4, 8))
        ctk.CTkButton(btn_row, text="+ Добавить тег", width=140, command=self._add_tag).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Сохранить", width=100, command=self._save).pack(side="right", padx=4)

        self._refresh_tags_list()

    def _refresh_tags_list(self) -> None:
        for w in self._tags_frame.winfo_children():
            w.destroy()
        for idx, tag in enumerate(self._store.tags):
            self._render_tag_row(idx, tag)

    def _render_tag_row(self, idx: int, tag: ContextTag) -> None:
        row = ctk.CTkFrame(self._tags_frame, fg_color=("gray88", "gray20"), corner_radius=6)
        row.pack(fill="x", padx=2, pady=2)

        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x", padx=6, pady=(4, 0))

        enabled_var = ctk.BooleanVar(value=tag.enabled)
        enabled_var.trace_add("write", lambda *_, i=idx, v=enabled_var: self._toggle_enabled(i, v.get()))
        ctk.CTkCheckBox(top, text="", variable=enabled_var, width=24).pack(side="left")

        ctk.CTkLabel(top, text="Ключ:", width=40).pack(side="left", padx=(4, 2))
        key_entry = ctk.CTkEntry(top, width=140)
        key_entry.pack(side="left", padx=(0, 8))
        key_entry.insert(0, tag.key)
        key_entry.bind("<FocusOut>", lambda e, i=idx, w=key_entry: self._update_key(i, w.get()))

        ctk.CTkLabel(top, text="Название:", width=70).pack(side="left", padx=(4, 2))
        label_entry = ctk.CTkEntry(top, width=140)
        label_entry.pack(side="left", padx=(0, 8))
        label_entry.insert(0, tag.label)
        label_entry.bind("<FocusOut>", lambda e, i=idx, w=label_entry: self._update_label(i, w.get()))

        ctk.CTkButton(
            top, text="✕", width=28, height=28,
            fg_color=("gray75", "gray35"),
            command=lambda i=idx: self._delete_tag(i),
        ).pack(side="right")

        desc_frame = ctk.CTkFrame(row, fg_color="transparent")
        desc_frame.pack(fill="x", padx=6, pady=(2, 4))
        ctk.CTkLabel(desc_frame, text="Описание:", width=70, anchor="w").pack(side="left")
        desc_entry = ctk.CTkEntry(desc_frame, width=450)
        desc_entry.pack(side="left", fill="x", expand=True, padx=(2, 4))
        desc_entry.insert(0, tag.description)
        desc_entry.bind("<FocusOut>", lambda e, i=idx, w=desc_entry: self._update_desc(i, w.get()))

    def _toggle_enabled(self, idx: int, val: bool) -> None:
        if idx < len(self._store.tags):
            self._store.tags[idx].enabled = val

    def _update_key(self, idx: int, val: str) -> None:
        if idx < len(self._store.tags):
            self._store.tags[idx].key = val.strip().lower().replace(" ", "_")

    def _update_label(self, idx: int, val: str) -> None:
        if idx < len(self._store.tags):
            self._store.tags[idx].label = val.strip()

    def _update_desc(self, idx: int, val: str) -> None:
        if idx < len(self._store.tags):
            self._store.tags[idx].description = val.strip()

    def _add_tag(self) -> None:
        new_tag = ContextTag(key="new_tag", label="Новый тег", description="")
        self._store.tags.append(new_tag)
        self._refresh_tags_list()

    def _delete_tag(self, idx: int) -> None:
        if idx < len(self._store.tags):
            self._store.tags.pop(idx)
            self._refresh_tags_list()

    # --- Custom category lists tab ---

    def _build_lists_tab(self, parent: ctk.CTkFrame) -> None:
        hint = ctk.CTkLabel(
            parent,
            text=(
                "Пользовательские списки категорий. Создайте свой набор тегов/папок для сортировки.\n"
                "Каждая строка — один тег (имя папки). Пустые строки игнорируются."
            ),
            wraplength=620,
            justify="left",
            text_color=("gray38", "gray62"),
        )
        hint.pack(anchor="w", padx=8, pady=(4, 8))

        top_row = ctk.CTkFrame(parent, fg_color="transparent")
        top_row.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkLabel(top_row, text="Список:", width=50).pack(side="left")
        self._list_selector_var = ctk.StringVar(value="")
        names = [cl.name for cl in self._store.custom_lists] or ["(нет)"]
        self._list_selector = ctk.CTkComboBox(
            top_row, values=names, variable=self._list_selector_var,
            width=200, state="readonly",
            command=self._on_list_selected,
        )
        self._list_selector.pack(side="left", padx=4)
        if self._store.custom_lists:
            self._list_selector_var.set(self._store.custom_lists[0].name)

        ctk.CTkButton(top_row, text="+ Новый", width=80, command=self._add_list).pack(side="left", padx=4)
        ctk.CTkButton(top_row, text="Удалить", width=80,
                      fg_color=("gray75", "gray35"), command=self._delete_list).pack(side="left", padx=4)
        ctk.CTkButton(top_row, text="Сохранить", width=100, command=self._save).pack(side="right", padx=4)

        self._list_text = ctk.CTkTextbox(parent, height=280)
        self._list_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._load_current_list_text()

    def _on_list_selected(self, _value: str = "") -> None:
        self._save_current_list_text()
        self._load_current_list_text()

    def _load_current_list_text(self) -> None:
        self._list_text.delete("1.0", "end")
        name = self._list_selector_var.get()
        for cl in self._store.custom_lists:
            if cl.name == name:
                self._list_text.insert("1.0", "\n".join(cl.categories))
                break

    def _save_current_list_text(self) -> None:
        name = self._list_selector_var.get()
        text = self._list_text.get("1.0", "end").strip()
        categories = [line.strip() for line in text.splitlines() if line.strip()]
        for cl in self._store.custom_lists:
            if cl.name == name:
                cl.categories = categories
                break

    def _add_list(self) -> None:
        dialog = ctk.CTkInputDialog(text="Название нового списка:", title="Новый список категорий")
        name = (dialog.get_input() or "").strip()
        if not name:
            return
        for cl in self._store.custom_lists:
            if cl.name == name:
                messagebox.showwarning("Уже существует", f"Список '{name}' уже существует.")
                return
        self._save_current_list_text()
        self._store.custom_lists.append(CustomCategoryList(name=name, categories=[]))
        names = [cl.name for cl in self._store.custom_lists]
        self._list_selector.configure(values=names)
        self._list_selector_var.set(name)
        self._load_current_list_text()

    def _delete_list(self) -> None:
        name = self._list_selector_var.get()
        if not name:
            return
        self._store.custom_lists = [cl for cl in self._store.custom_lists if cl.name != name]
        names = [cl.name for cl in self._store.custom_lists] or ["(нет)"]
        self._list_selector.configure(values=names)
        self._list_selector_var.set(names[0] if self._store.custom_lists else "(нет)")
        self._load_current_list_text()

    def _save(self) -> None:
        self._save_current_list_text()
        self._store.tags = [t for t in self._store.tags if t.key.strip()]
        save_context_tags(self._store)
        if self._on_save_callback:
            self._on_save_callback()
        messagebox.showinfo("Сохранено", "Контекстные теги и списки категорий сохранены.")

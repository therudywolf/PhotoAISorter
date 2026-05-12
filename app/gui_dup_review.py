# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Lightweight window to browse exported duplicate-finder results (disk JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import customtkinter as ctk

from app.ui_texts import t as ui_t


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def open_duplicate_review_window(master: ctk.Misc, result_json: Path) -> None:
    data = _load_result(result_json)
    groups = list(data.get("groups") or [])
    records = list(data.get("records") or [])
    meta_path = result_json.parent / "meta.json"
    meta_txt = ""
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                meta_txt = f"{meta.get('root_path', '')} | {meta.get('media_mode', '')} | {meta.get('strictness', '')}"
        except (OSError, json.JSONDecodeError):
            pass

    top = ctk.CTkToplevel(master)
    top.title(ui_t("dup.review.title"))
    top.geometry("900x640")
    top.transient(master.winfo_toplevel())
    top.lift()
    try:
        top.grab_set()
    except Exception:
        pass

    head = ctk.CTkLabel(top, text=ui_t("dup.review.subtitle", groups=len(groups), records=len(records)), anchor="w")
    head.pack(fill="x", padx=10, pady=(8, 4))
    if meta_txt:
        ctk.CTkLabel(top, text=meta_txt, anchor="w", text_color=("gray35", "gray65"), font=ctk.CTkFont(size=11)).pack(
            fill="x", padx=10, pady=(0, 6)
        )

    scroll = ctk.CTkScrollableFrame(top, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=8, pady=8)

    for i, g in enumerate(groups):
        if not isinstance(g, dict):
            continue
        paths = g.get("paths") or []
        sk = g.get("suggested_keep", "")
        exact = g.get("is_exact", False)
        fr = ctk.CTkFrame(scroll, fg_color=("gray88", "gray20"), corner_radius=6)
        fr.pack(fill="x", pady=4)
        tag = ui_t("dup.card.exact_suffix") if exact else ""
        title = ui_t("dup.review.group_line", n=i + 1, nfiles=len(paths), tag=tag)
        ctk.CTkLabel(fr, text=title, font=ctk.CTkFont(weight="bold"), anchor="w").pack(anchor="w", padx=8, pady=(6, 2))
        if sk:
            ctk.CTkLabel(fr, text=ui_t("dup.review.suggested", path=str(sk)), anchor="w", text_color=("gray30", "gray70")).pack(
                anchor="w", padx=8, pady=(0, 4)
            )
        for p in paths[:200]:
            ctk.CTkLabel(fr, text=str(p), anchor="w", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=12, pady=1)
        if len(paths) > 200:
            ctk.CTkLabel(fr, text=ui_t("dup.review.truncated", n=len(paths) - 200), text_color=("gray40", "gray60")).pack(
                anchor="w", padx=12, pady=4
            )

    if not groups and records:
        ctk.CTkLabel(scroll, text=ui_t("dup.review.records_only", n=len(records)), anchor="w").pack(anchor="w", padx=8, pady=8)

    ctk.CTkButton(top, text=ui_t("dup.review.close"), command=top.destroy).pack(pady=8)

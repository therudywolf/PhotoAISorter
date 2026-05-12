# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Review-first sort manifest writer."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class SortReviewManifest:
    def __init__(
        self,
        dest_dir: Path,
        *,
        source_dir: Path,
        media_mode: str,
        tag_mode: str,
        model: str,
    ) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = dest_dir / "_review_runs" / f"sort-{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "manifest.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self._lock = threading.Lock()
        self._count = 0
        meta = {
            "source_dir": str(source_dir),
            "dest_dir": str(dest_dir),
            "media_mode": media_mode,
            "tag_mode": tag_mode,
            "model": model,
            "created_at": time.time(),
            "format": "jsonl",
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def count(self) -> int:
        return self._count

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._count += 1

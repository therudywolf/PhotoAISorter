"""Unified cache operations for sort and duplicate databases."""

from __future__ import annotations

from dataclasses import dataclass

from app.db import Database
from app.signature_db import SignatureDatabase


@dataclass(frozen=True)
class CacheSummary:
    sort_records: int
    duplicate_signatures: int


class CacheService:
    def __init__(self, sort_db: Database, sig_db: SignatureDatabase) -> None:
        self._sort_db = sort_db
        self._sig_db = sig_db

    def summary(self) -> CacheSummary:
        return CacheSummary(
            sort_records=self._sort_db.count_records(),
            duplicate_signatures=self._sig_db.count_signatures(),
        )

    def clear_sort_cache(self) -> int:
        return self._sort_db.clear_all_records()

    def clear_duplicate_cache(self) -> int:
        return self._sig_db.clear_all()

    def clear_all(self) -> tuple[int, int]:
        return self.clear_sort_cache(), self.clear_duplicate_cache()

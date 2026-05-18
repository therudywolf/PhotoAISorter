Runtime cache (gitignored). Keep this folder on a fast disk (SSD).

  app_state/          — sort DB, duplicate signatures, gui_settings.json
  clip_cache.sqlite3  — CLIP embedding vectors (per model + crop profile)
  file_hashes.sqlite3 — path+mtime → sha256 (avoids re-reading files on HDD)

Large assets: ../data/refs and ../data/clip_weights

Override location: set environment variable PHOTO_AI_SORTER_TMP

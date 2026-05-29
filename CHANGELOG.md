# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed — sorting accuracy
- **Removed aspect-ratio and brightness heuristics** that mislabelled photos.
  The old `aspect_ui_like` rule tagged any image with a phone-like aspect ratio
  (9:16, 3:4, 16:9) as a screenshot, and `bright_paper_like` tagged any bright
  photo as a document. Vertical phone photos share those exact ratios, so
  ordinary photos of people and pets were routed into `screenshot`/`document`.
  Leave-one-out evaluation on the bundled reference photos went from **15 % to
  100 %** once these rules were dropped. Filename-based detection
  (`screenshot`, `receipt`, `document`) is kept; CLIP decides the rest.
- **English-only CLIP prompts.** OpenAI CLIP is English-trained; the prompt
  builder previously mixed in Russian templates and raw tag ids
  (`furry_nsfw_canidae` is not a word), producing noisy class prototypes.
  Prompts are now English-only and built from the human-readable label plus the
  user description.
- **Idempotent file copy.** A hard crash mid-run could leave a copied file with
  no committed database record; on resume the file was duplicated as
  `name_1.ext`. `unique_dest_path` now reuses a byte-identical existing file
  (compared by SHA-256), so resume no longer duplicates output.

### Fixed — UI / UX
- Routed all remaining hard-coded dialog strings through `ui_texts`
  (LM Studio panel, cache dialog, profile manager, folder pickers, exemplars).
- Added a minimum window size so panels stop clipping on small screens.
- Hardened `Toplevel` dialog setup: `grab_set` is deferred and guarded, fixing
  a Windows crash when a dialog was not yet viewable.
- Worker-thread UI callbacks now route through a guarded helper, so closing the
  app mid-probe no longer prints teardown tracebacks.
- Fixed a thumbnail generation bug in the duplicate-groups viewer where a
  deferred callback captured the wrong loop variable, defeating the
  stale-generation check after a redraw.

### Changed — repository hygiene
- Cleaned the bundled tag set from 60 to 46 tags (removed view-variant,
  `*_alt` legacy, and combo tags); legacy tags fold back into canonical
  folders via `BUILTIN_STORAGE_ALIASES`.
- `data/refs/` no longer un-ignores personal reference photos — only its
  `README.txt` is tracked.
- Added `ruff` to CI with the `B` (bugbear) ruleset enabled.
- Synced `pyproject.toml` version with the `VERSION` file.

### Added
- `scripts/eval_accuracy.py` — leave-one-out accuracy evaluation over
  `data/refs/<tag>/`, reporting per-tag and overall accuracy.
- Regression tests: filename heuristics, CLIP prompts, `pick_tag` reweighting
  rules, alias resolution, and `unique_dest_path` crash-recovery idempotency.

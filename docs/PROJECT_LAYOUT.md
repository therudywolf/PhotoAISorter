# Project layout

Canonical folders for a working copy of Photo AI Sorter.

## Tracked in git

| Path | Purpose |
|------|---------|
| `app/` | Application source code |
| `tests/` | Pytest suite |
| `main.py` | Entry point |
| `Photo AI Sorter.cmd` | Windows entry point: double-click → setup + GUI, makes a desktop shortcut on first run |
| `run.bat` | Windows engine (venv, deps, GUI, tests); used by the launcher and for console runs |
| `run.sh` | Linux/macOS launcher (same roles as `run.bat`) |
| `requirements.txt` | Runtime pip dependencies |
| `requirements-dev.txt` | Dev deps (`-r requirements.txt` + pytest) |
| `requirements-gpu.txt` | Optional CUDA PyTorch overlay |
| `scripts/install_torch_cuda.ps1` | GPU PyTorch install helper |
| `examples/` | Sample JSON configs (not used at runtime) |
| `docs/` | User/developer documentation |
| `data/README.txt` | Explains local `data/` layout |
| `tmp/README.txt` | Explains local `tmp/` cache layout |

## Local only (gitignored)

| Path | Purpose |
|------|---------|
| `.venv/` | Python virtual environment (Windows and Linux) |
| `tmp/app_state/` | SQLite DBs, `gui_settings.json`, `context_tags.json`, secrets |
| `tmp/*.sqlite3` | CLIP embedding cache, file hash cache |
| `data/refs/<tag>/` | CLIP exemplar photos |
| `data/clip_weights/*.pt` | Downloaded CLIP weights |
| `ffmpeg-runtime/` | Bundled ffmpeg binaries |
| `.env.local` | Optional API base/key for LM Studio |
| `<sort-output>/_review_runs/` | Review-first manifests |

## Do not keep at repo root

- `local_presets.json` — legacy; migrated to `tmp/app_state/context_tags.json` on startup
- `.cache/` — stray download cache; use `data/clip_weights/` instead
- `.venv-linux/` — renamed to `.venv` by `run.sh` when present

## Environment overrides

- `PHOTO_AI_SORTER_TMP` — alternate directory for `tmp/` caches
- `PHOTO_AI_SORTER_FFMPEG` — path to ffmpeg binary or `bin` folder
- `PHOTO_AI_SORTER_API_BASE`, `PHOTO_AI_SORTER_API_KEY`, `PHOTO_AI_SORTER_MODEL`

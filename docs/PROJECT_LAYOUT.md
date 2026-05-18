# Project layout

Canonical folders for a working copy of Photo AI Sorter.

## Tracked in git

| Path | Purpose |
|------|---------|
| `app/` | Application source code |
| `tests/` | Pytest suite |
| `main.py` | Entry point |
| `run.bat` | Windows launcher (venv, deps, GUI, tests) |
| `run.sh` | Linux/macOS launcher (same roles as `run.bat`) |
| `START_Photo_AI_Sorter.sh` | Thin wrapper → `run.sh` |
| `START_Photo_AI_Sorter.cmd` | Double-click GUI wrapper → `run.bat gui` |
| `CREATE_DESKTOP_SHORTCUT.cmd` | Creates a desktop shortcut to the `.cmd` launcher |
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

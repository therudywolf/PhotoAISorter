# 🐺 Photo AI Sorter

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Local desktop tool for sorting mixed photo/video libraries with a vision model, finding duplicates, and reviewing risky cleanup decisions before deleting files.

AGPL v3 Copyleft applies to reuse, modification, and network deployment of derived versions.

The app is designed for private local media libraries that may contain personal photos, screenshots, generated images, downloaded archives, memes, videos, GIFs, and duplicates. Processing stays on your machine when the configured OpenAI-compatible API server is local. If you point the app at a remote API server, selected images or video frames are sent to that server for classification.

## Features

- Photo, video, and GIF scanning.
- Local vision classification through any OpenAI-compatible API (LM Studio, Ollama, etc.).
- Four search profiles (built-in category presets):
  - **SFW** — safe-for-work categories only.
  - **NSFW** — SFW + explicit content categories.
  - **Furry SFW** — adds furry art categories (SFW only).
  - **Furry NSFW** — full set including furry NSFW.
- Three flexible tag modes:
  - **Auto categories** — stabilized model-generated folder names.
  - **Free model tags** — unrestricted hierarchical tags for experiments.
  - **Custom list** — user-defined tag sets with optional recognition descriptions.
- **Fast CLIP (hybrid)** — local OpenCLIP scoring with heuristics, exemplar photos, and optional VLM fallback for uncertain files (best for large libraries with custom tag sets).
- User-defined tag sets: create named collections of output categories, each with an optional description that tells the model what to look for.
- Structured JSON classification with legacy tag fallback.
- Model profiles for classifier, duplicate verifier, screenshot OCR, and quick preview roles.
- Editable auto-category aliases.
- Review-first sort manifests under `_review_runs`.
- Resumable sort sessions with saved progress after stop, crash, or reboot.
- LM health telemetry in the GUI.
- Hash cache for instant skipping of already-sorted files.
- Exact and perceptual duplicate detection.
- Semantic/colorhash duplicate candidates for deep LLM verification.
- Optional LLM verification for ambiguous duplicate pairs.
- Duplicate review UI with keep/delete selection and safer deletion flow.
- FFmpeg/OpenCV video frame extraction fallback.

## Performance Notes

Duplicate search has several modes with different costs:

- `Only exact copies` first groups files by byte size, skips unique-size files, and hashes only possible exact-copy candidates. It does not decode images or video frames.
- `Balanced` and stricter modes compute visual signatures. Video and animated GIF files need representative frames, so installing FFmpeg usually improves speed and reliability.
- The duplicate tab exposes local scan workers next to the scan button. More workers can help on SSDs and multi-core CPUs, but very large videos can still become disk-bound.
- Optional LLM verification is intentionally slower because it sends ambiguous pairs to the selected vision model.

For large video libraries, start with `Only exact copies`, then run `Balanced` only on folders where near-duplicates matter.

## Quick Start on Windows

Requirements: Python 3.10 or newer. Python 3.11 is used by the Docker test image.

Double-click:

```text
START_Photo_AI_Sorter.cmd
```

The launcher creates `.venv`, installs dependencies when needed, runs import checks, and starts the GUI. The first launch can take a few minutes while packages are installed; later launches reuse the existing `.venv`.

To create a real desktop shortcut, double-click once:

```text
CREATE_DESKTOP_SHORTCUT.cmd
```

For a diagnostic console run, use:

```bat
run.bat
```

Run tests:

```bat
run.bat test
```

The app does not require a committed API key. For servers that enforce bearer auth, set:

```bash
PHOTO_AI_SORTER_API_BASE=http://127.0.0.1:1234
PHOTO_AI_SORTER_API_KEY=your-local-key
PHOTO_AI_SORTER_MODEL=local-model
```

For your LM Studio setup you can also create a local `.env.local` next to `run.bat`.
It is ignored by git:

```bash
PHOTO_AI_SORTER_API_BASE=http://your-server:port
PHOTO_AI_SORTER_API_KEY=your-lm-studio-token
```

The GUI also has an `API key` field. That value is stored under the app data
directory, not inside the repository.

## Manual Setup

Runtime dependencies are in `requirements.txt`. Test/development dependencies are in `requirements-dev.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python main.py
```

On Linux/macOS-style shells:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## LM Studio Setup

1. Start LM Studio server with an OpenAI-compatible API.
2. Use a vision-capable model.
3. Set the base URL in the GUI.
4. Use `Refresh models`, then run `Vision self-test`.

Recommended model characteristics:

- Vision/multimodal support.
- Stable instruction following.
- Low hallucination on short tag-only outputs.
- Enough context to process the classification prompt.

For LM Studio stability, keep LM-request concurrency at 1 (the default). The app prepares/hashes/copies files in parallel while serializing calls to the local model.

More setup notes are in [docs/LM_STUDIO.md](docs/LM_STUDIO.md).

## Docker Test Image

The Dockerfile is for repeatable compile/test checks, not for running the desktop GUI:

```bash
docker build -t photo-ai-sorter:test .
docker run --rm photo-ai-sorter:test
```

## Release Hygiene

Publish from git, not by zipping the whole working directory. Local files such as `.env.local`, `local_presets.json`, `.venv`, `ffmpeg-runtime`, SQLite caches, review runs, and duplicate journals are intentionally ignored and can contain private data or bundled binaries.

For a clean source archive:

```bash
git archive --format=zip --output photo-ai-sorter-source.zip HEAD
```

Before pushing a release branch, run the checks in [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Model Profiles

Profiles store API base, model id, worker count, timeout, max tokens, temperature, and optional prompt additions.

Built-in profile names:

- `classifier`
- `duplicate_verifier`
- `screenshot_ocr`
- `fast_preview`

The sorter uses the active profile. Duplicate workflows keep their own model controls, but the profile system is ready for role-specific expansion.

## Search Profiles

The app ships with four built-in search profiles that determine which output categories are available:

| Profile | Content |
|---------|---------|
| **SFW** | Generic categories (humans, animals, vehicles, tech, art, etc.) — no NSFW |
| **NSFW** | SFW + explicit human/AI-generated categories |
| **Furry SFW** | SFW + furry art SFW categories |
| **Furry NSFW** | All of the above + furry NSFW |

Select a profile in the GUI's segmented button bar before starting a sort.

## Custom Tag Sets

For personalized presets (e.g., recognizing your pet, specific people, niche categories), use the **Custom list** mode with user-defined tag sets stored locally in `context_tags.json` (gitignored). This way personal recognition context never leaves your machine or enters the repository.

## Sorting Modes

`Auto categories` normalizes model output so near-duplicates such as `car`, `cars`, `vehicle`, and `auto` do not create separate folder trees.

`Free model tags` is intentionally free-form and can create many folders. Use it for small test batches or exploration.

## Review-First Mode

Enable `Review-first` to classify files without copying them. The app writes JSONL manifests into:

```text
<output>/_review_runs/sort-YYYYMMDD-HHMMSS/manifest.jsonl
```

Each line includes the source path, SHA-256, category, candidates, confidence, short reason, review flag, and raw model output excerpt.

## Resuming Large Runs

Sorting progress is saved into the local SQLite app data database. If a run is
stopped or the PC shuts down, reopen the app and start the same input/output/mode
again and choose `Да` when prompted.

The resume path stores per-file path, size, mtime, SHA-256, category, and session
status locally. API keys are not written into session records.

## Category Aliases

`Aliases categories...` opens a JSON editor. Example:

```json
{
  "auto": "vehicles",
  "bmw/car": "vehicles/bmw",
  "screen": "screenshots"
}
```

Aliases apply to smart auto categories before folder names are created.

See [examples/category_aliases.example.json](examples/category_aliases.example.json) for a larger starter map.

For hybrid **Fast CLIP** mode, virtual tag names (finer labels for the model) can map to parent folders on disk — see [examples/hybrid_virtual_aliases.example.json](examples/hybrid_virtual_aliases.example.json). Copy the pattern into your local `category_aliases.json` under app data (that file is gitignored).

### Fast CLIP (hybrid) mode

1. Install optional deps: `pip install torch open-clip-torch` (included in `requirements.txt`).
2. In the GUI, choose **Fast CLIP**, configure tags via **Tags…**, and add reference photos via **References…** (`refs/<tag>/` under app data).
3. Enable **VLM fallback** only for files CLIP marks as uncertain — keeps large runs fast.

Tune thresholds in `gui_settings.json` → `fast_classify` (`confidence_threshold`, `min_margin`, `exemplar_boost`). That settings file is local and gitignored.

## Before you commit or push

- Do **not** commit `local_presets.json`, `context_tags.json`, `category_aliases.json`, `gui_settings.json`, `refs/`, or SQLite databases — they are listed in `.gitignore`.
- Run `python -m pytest -q` and `python -m compileall -q app tests main.py`.
- Scan for secrets if you use [gitleaks](https://github.com/gitleaks/gitleaks): `gitleaks detect --source .` (optional).

## Development Checks

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app tests main.py
python -m pytest -q
```

The repository normalizes text files through `.gitattributes` and `.editorconfig` to keep diffs readable.

## Contributing

Contributions are welcome. Please open an issue first for large changes and read [CONTRIBUTING.md](CONTRIBUTING.md).

Report security issues using [SECURITY.md](SECURITY.md). Third-party dependency notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

Copyright (C) 2026 Photo AI Sorter contributors.

In short: you may use, modify, and distribute this software freely, but any modified version that is accessible over a network must also share its source code under the same license.

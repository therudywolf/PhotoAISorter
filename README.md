# Photo AI Sorter

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Local desktop tool for sorting mixed photo/video libraries with a vision model, finding duplicates, and reviewing risky cleanup decisions before deleting files.

The app is designed for private local media libraries that may contain personal photos, screenshots, generated images, downloaded archives, memes, videos, GIFs, and duplicates. All processing happens locally — no data leaves your machine.

## Features

- Photo, video, and GIF scanning.
- Local vision classification through any OpenAI-compatible API (LM Studio, Ollama, etc.).
- Five tag modes:
  - **Furry** — fixed built-in category list.
  - **General** — broader preset for geek, IT, car culture, fitness, nightlife, LGBTQ+ galleries.
  - **Auto categories** — stabilized model-generated folder names.
  - **Free model tags** — unrestricted hierarchical tags for experiments.
  - **Custom list** — user-defined tag sets with optional recognition descriptions.
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

## Quick Start on Windows

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
PHOTO_AI_SORTER_API_BASE=http://10.77.77.2:29931
PHOTO_AI_SORTER_API_KEY=your-lm-studio-token
```

The GUI also has an `API key` field. That value is stored under the app data
directory, not inside the repository.

## Manual Setup

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

## Model Profiles

Profiles store API base, model id, worker count, timeout, max tokens, temperature, and optional prompt additions.

Built-in profile names:

- `classifier`
- `duplicate_verifier`
- `screenshot_ocr`
- `fast_preview`

The sorter uses the active profile. Duplicate workflows keep their own model controls, but the profile system is ready for role-specific expansion.

## Sorting Modes

`Furry` is the safest original preset for large libraries because it only writes the original known folders.

`General` is the recommended fixed preset for mixed personal libraries. It keeps all `Furry` tags and adds broader buckets for geek, IT, car, fitness, nightlife, and LGBTQ+ content without allowing arbitrary folder creation.

`Auto categories` is intended for mixed libraries where you want new folders. It normalizes model output so near-duplicates such as `car`, `cars`, `vehicle`, and `auto` do not create separate folder trees.

`Free model tags` is intentionally free-form and can create many folders. Use it for small test batches or exploration.

## Review-First Mode

Enable `Review-first` to classify files without copying them. The app writes JSONL manifests into:

```text
<output>/_review_runs/sort-YYYYMMDD-HHMMSS/manifest.jsonl
```

Each line includes the source path, SHA-256, category, candidates, confidence, short reason, review flag, and raw model output excerpt.

## Resuming Large Runs

Sorting progress is saved into the local SQLite app data database. If a run is
stopped or the PC shuts down, reopen the app and use `Продолжить сессию`, or
start the same input/output/mode again and choose `Да` when prompted.

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

## Development Checks

```bash
python -m compileall -q app tests main.py
python -m pytest -q
```

The repository normalizes text files through `.gitattributes` and `.editorconfig` to keep diffs readable.

Before publishing a release, use [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

In short: you may use, modify, and distribute this software freely, but any modified version that is accessible over a network must also share its source code under the same license.

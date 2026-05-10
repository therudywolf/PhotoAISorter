# Photo AI Sorter

Local desktop tool for sorting mixed photo/video libraries with a vision model, finding duplicates, and reviewing risky cleanup decisions before deleting files.

The app is designed for private local media libraries that may contain personal photos, screenshots, generated images, downloaded archives, memes, videos, GIFs, and duplicates.

## Current Features

- Photo, video, and GIF scanning.
- Local vision classification through an OpenAI-compatible LM Studio API.
- Three tag modes:
  - `Preset: my tags`: fixed built-in category list.
  - `Smart auto categories`: built-in preset first, then stabilized new folders.
  - `Model tags`: free-form model folders for experiments.
- Structured JSON classification with legacy tag fallback.
- Model profiles for classifier, duplicate verifier, screenshot OCR, and quick preview roles.
- Lightweight LM Studio vision benchmark for choosing a model.
- Prompt composer and editable auto-category aliases.
- Review-first sort manifests under `_review_runs`.
- LM health telemetry in the GUI.
- Hash cache for resumable sorting.
- Exact and perceptual duplicate detection.
- Semantic/colorhash duplicate candidates for deep LLM verification.
- Optional LLM verification for ambiguous duplicate pairs.
- Duplicate review UI with keep/delete selection and safer deletion flow.
- FFmpeg/OpenCV video frame extraction fallback.

## Quick Start on Windows

```bat
run.bat
```

The launcher creates `.venv`, installs dependencies, runs import checks, and starts the GUI.

Run tests:

```bat
run.bat test
```

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

Use `Benchmark` after refreshing models to probe the first visible candidates with the built-in vision test card. The best model is written into the active model profile.

## Model Profiles

Profiles store API base, model id, worker count, timeout, max tokens, temperature, and optional prompt additions.

Built-in profile names:

- `classifier`
- `duplicate_verifier`
- `screenshot_ocr`
- `fast_preview`

The sorter uses the active profile. Duplicate workflows keep their own model controls, but the profile system is ready for role-specific expansion.

## Sorting Modes

`Preset: my tags` is the safest mode for large libraries because it only writes known folders.

`Smart auto categories` is intended for mixed libraries. It normalizes model output so near-duplicates such as `car`, `cars`, `vehicle`, and `auto` do not create separate folder trees.

`Model tags` is intentionally free-form and can create many folders. Use it for small test batches or exploration.

## Review-First Mode

Enable `Review-first` to classify files without copying them. The app writes JSONL manifests into:

```text
<output>/_review_runs/sort-YYYYMMDD-HHMMSS/manifest.jsonl
```

Each line includes the source path, SHA-256, category, candidates, confidence, short reason, review flag, and raw model output excerpt.

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

## Development Checks

```bash
python -m compileall -q app tests main.py
python -m pytest -q
```

The repository normalizes text files through `.gitattributes` and `.editorconfig` to keep diffs readable.

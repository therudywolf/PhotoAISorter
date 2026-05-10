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
- Hash cache for resumable sorting.
- Exact and perceptual duplicate detection.
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

## Sorting Modes

`Preset: my tags` is the safest mode for large libraries because it only writes known folders.

`Smart auto categories` is intended for mixed libraries. It normalizes model output so near-duplicates such as `car`, `cars`, `vehicle`, and `auto` do not create separate folder trees.

`Model tags` is intentionally free-form and can create many folders. Use it for small test batches or exploration.

## Development Checks

```bash
python -m compileall -q app tests main.py
python -m pytest -q
```

The repository normalizes text files through `.gitattributes` and `.editorconfig` to keep diffs readable.

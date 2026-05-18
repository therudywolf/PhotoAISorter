# Release checklist

Before tagging or publishing a release from `main`:

1. **Tests** — `run.bat test` (Windows) or `./run.sh test` (Linux)
2. **Compile** — `python -m compileall -q app tests main.py`
3. **Secrets** — `gitleaks detect --source .` (optional; pre-commit hook uses `.gitleaks.toml`)
4. **Archive** — publish from git, not a working folder zip:
   ```bash
   git archive --format=zip --output photo-ai-sorter-source.zip HEAD
   ```
5. **Exclude local data** — no `.venv`, `tmp/`, `data/clip_weights`, `data/refs`, `.env.local`, `ffmpeg-runtime`, review manifests, or personal JSON presets

After pipeline changes, bump `PIPELINE_VERSION` in `app/constants.py` so sort caches invalidate correctly.

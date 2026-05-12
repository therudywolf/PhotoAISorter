# Contributing

Thanks for helping improve Photo AI Sorter.

## Development setup

Use Python 3.10 or newer. Python 3.11 is used by the Docker test image.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m compileall -q app tests main.py
.venv/bin/python -m pytest -q
```

On Windows, `run.bat test` creates or updates the local virtual environment and runs the test suite.

## Pull requests

- Keep user-specific presets, API keys, generated databases, media libraries, and local `.env*` files out of commits.
- Add focused tests for behavior changes.
- Keep the application usable without a network service unless the feature explicitly needs one.
- Document new user-visible settings in `README.md`.

## License

By contributing, you agree that your contribution is provided under the project's AGPL-3.0-only license.

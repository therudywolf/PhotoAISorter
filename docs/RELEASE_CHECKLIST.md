# Release Checklist

Use this checklist before pushing changes to `main`.

1. Confirm the working tree is clean or contains only intended changes.
2. Run:

   ```bash
   python -m compileall -q app tests main.py
   python -m pytest -q
   git diff --check
   ```

3. Start the app with `run.bat` on Windows for a GUI smoke test.
4. Run `Vision self-test` against the intended LM Studio server.
5. Test sorting on a small sample with `Review-first` enabled.
6. Inspect `_review_runs/.../manifest.jsonl` before a real library run.
7. Push to `main` only after tests and smoke checks pass.

# Release Checklist

Use this checklist before pushing changes to `main`.

1. Confirm the working tree is clean or contains only intended changes.
2. Run:

   ```bash
   python -m compileall -q app tests main.py
   python -m pytest -q
   git diff --check
   ```

3. Check that ignored local files are still ignored and not staged:

   ```bash
   git status --short --ignored
   git ls-files -ci --exclude-standard
   ```

4. Search tracked files for accidental secrets:

   ```bash
   git grep -n -E 'api[_-]?key|secret|token|password|bearer|authorization|PRIVATE KEY|AKIA|sk-|gh[pousr]_' -- .
   ```

5. Start the app with `START_Photo_AI_Sorter.cmd` on Windows for a one-click GUI smoke test.
6. Run `Vision self-test` against the intended LM Studio server.
7. Test sorting on a small sample with `Review-first` enabled.
8. Inspect `_review_runs/.../manifest.jsonl` before a real library run.
9. Create source archives from git only, for example:

   ```bash
   git archive --format=zip --output photo-ai-sorter-source.zip HEAD
   ```

10. Push to `main` only after tests, hygiene checks, and smoke checks pass.

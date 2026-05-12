# Security Policy

## Reporting a Vulnerability

Please report security issues privately by opening a GitHub security advisory for this repository. If advisories are unavailable, contact the maintainer through GitHub and avoid posting exploit details publicly until there is a fix or mitigation.

Include:

- affected version or commit
- operating system and Python version
- steps to reproduce
- impact and any known workaround

Do not include real API tokens, private media, or personal presets in reports. Redact local paths and secrets where possible.

## Local Data

Photo AI Sorter stores GUI settings, optional API credentials, caches, and run state in local app data outside the repository. Files such as `.env.local`, `context_tags.json`, `local_presets.json`, SQLite databases, review manifests, and duplicate journals must not be committed.

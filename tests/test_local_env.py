"""Local .env loading."""

from pathlib import Path

from app.local_env import load_project_env


def test_load_project_env_reads_local_without_overriding(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("PHOTO_AI_SORTER_API_BASE", raising=False)
    monkeypatch.delenv("PHOTO_AI_SORTER_MODEL", raising=False)
    monkeypatch.setenv("PHOTO_AI_SORTER_API_KEY", "existing")
    (tmp_path / ".env.local").write_text(
        "PHOTO_AI_SORTER_API_BASE=http://192.168.1.100:1234\n"
        "PHOTO_AI_SORTER_API_KEY=local-secret\n"
        "PHOTO_AI_SORTER_MODEL='vision-model'\n",
        encoding="utf-8",
    )
    loaded = load_project_env(tmp_path)
    assert loaded == [tmp_path / ".env.local"]
    import os

    assert os.environ["PHOTO_AI_SORTER_API_BASE"] == "http://192.168.1.100:1234"
    assert os.environ["PHOTO_AI_SORTER_API_KEY"] == "existing"
    assert os.environ["PHOTO_AI_SORTER_MODEL"] == "vision-model"

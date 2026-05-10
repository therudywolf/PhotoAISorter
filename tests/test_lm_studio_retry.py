"""Unit tests for retry and channel-error handling."""

from __future__ import annotations

import requests

from app import lm_studio


def test_empty_api_key_omits_authorization_header() -> None:
    assert "Authorization" not in lm_studio._auth_headers_json("")
    assert lm_studio._auth_headers_get("") == {}


def test_retryable_request_error_channel_message() -> None:
    assert lm_studio._retryable_request_error(RuntimeError("Channel Error")) is True


def test_run_completion_retries_fast_retry_for_channel_error(monkeypatch: object) -> None:
    calls = {"n": 0}
    slept: list[float] = []
    retry_msgs: list[str] = []

    def op() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Channel Error")
        return "ok"

    monkeypatch.setattr(lm_studio.time, "sleep", lambda sec: slept.append(sec))
    out = lm_studio._run_completion_retries(
        op,
        on_retry=lambda m: retry_msgs.append(m),
        attempt_label="retry",
    )
    assert out == "ok"
    assert calls["n"] == 2
    assert slept == []
    assert any("transient backoff" in m.lower() for m in retry_msgs)


def test_run_completion_retries_uses_jittered_sleep(monkeypatch: object) -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def op() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("timeout")
        return "ok"

    monkeypatch.setattr(lm_studio.time, "sleep", lambda sec: slept.append(sec))
    monkeypatch.setattr(lm_studio.random, "random", lambda: 0.0)
    out = lm_studio._run_completion_retries(op, on_retry=None, attempt_label="retry")
    assert out == "ok"
    assert len(slept) == 1
    assert slept[0] > 0

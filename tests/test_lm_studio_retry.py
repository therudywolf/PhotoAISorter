# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Unit tests for retry and channel-error handling."""

from __future__ import annotations

import requests
from app import lm_studio


def test_normalize_api_base_accepts_endpoint_prefixes() -> None:
    assert lm_studio.normalize_api_base("http://host:1234/v1") == "http://host:1234"
    assert lm_studio.normalize_api_base("http://host:1234/api/v1") == "http://host:1234"
    assert lm_studio.normalize_api_base("http://host:1234/") == "http://host:1234"


def test_empty_api_key_omits_authorization_header() -> None:
    assert "Authorization" not in lm_studio._auth_headers_json("")
    assert lm_studio._auth_headers_get("") == {}


def test_retryable_request_error_channel_message() -> None:
    assert lm_studio._retryable_request_error(RuntimeError("Channel Error")) is True


def test_extract_assistant_text_strips_closed_think_block() -> None:
    msg = {"content": "<think>reasoning with human_real_sfw</think>\n{\"primary_category\":\"tech/desk\"}"}
    assert lm_studio._extract_assistant_text(msg) == '{"primary_category":"tech/desk"}'


def test_extract_assistant_text_prefers_final_channel() -> None:
    msg = {"content": "<|channel|>analysis\nhuman_real_sfw\n<|channel|>final\n{\"primary_category\":\"tech/desk\"}"}
    assert lm_studio._extract_assistant_text(msg) == '{"primary_category":"tech/desk"}'


def test_extract_assistant_text_ignores_unclosed_think_block() -> None:
    msg = {"content": "<think>\nreasoning only"}
    assert lm_studio._extract_assistant_text(msg) == ""


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
    assert slept == [0.5]
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


def test_loaded_model_instances_flattens_lm_studio_v1(monkeypatch: object) -> None:
    class Resp:
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "models": [
                    {
                        "key": "google/gemma-3-4b",
                        "display_name": "Gemma",
                        "capabilities": {"vision": True},
                        "loaded_instances": [
                            {"id": "google/gemma-3-4b", "config": {"context_length": 1024, "parallel": 4}},
                            {"id": "google/gemma-3-4b:2", "config": {"context_length": 8192, "parallel": 4}},
                        ],
                    }
                ]
            }

    monkeypatch.setattr(lm_studio.requests, "get", lambda *_a, **_k: Resp())

    rows = lm_studio.loaded_model_instances("http://x")
    assert [r["instance_id"] for r in rows] == ["google/gemma-3-4b", "google/gemma-3-4b:2"]
    assert rows[0]["parallel"] == 4


def test_unload_duplicate_model_instances_removes_inactive_and_duplicate(monkeypatch: object) -> None:
    rows = [
        {"model_key": "old/model", "instance_id": "old/model"},
        {"model_key": "active/model", "instance_id": "active/model"},
        {"model_key": "active/model", "instance_id": "active/model:2"},
    ]
    unloaded: list[str] = []

    monkeypatch.setattr(lm_studio, "loaded_model_instances", lambda *_a, **_k: rows)
    monkeypatch.setattr(
        lm_studio,
        "unload_model_instance",
        lambda _base, instance_id, **_k: unloaded.append(instance_id) or instance_id,
    )

    out = lm_studio.unload_duplicate_model_instances("http://x", keep_models={"active/model"})

    assert out == ["old/model", "active/model:2"]
    assert unloaded == out

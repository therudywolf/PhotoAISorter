# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""LM Studio OpenAI-compatible chat completions (vision)."""

from __future__ import annotations

import itertools
import json
import random
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from app.constants import (
    API_MAX_RETRIES,
    API_PROBE_TIMEOUT_SEC,
    API_RETRY_BACKOFF_SEC,
    CHAT_COMPLETIONS_PATH,
    CHAT_COMPLETION_MAX_TOKENS,
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    GENERAL_CATEGORY_WHITELIST,
    GIF_EXTENSION,
    MODELS_PATH,
    PRIORITY_RULES_BLOCK,
    REQUEST_CONNECT_TIMEOUT_SEC,
    REQUEST_READ_TIMEOUT_SEC,
    UNCATEGORIZED,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
    VISION_PROBE_MAX_TOKENS,
    VISION_TEST_TIMEOUT_SEC,
)
from app.tag_config import TagMode, ResolvedTagConfig
from app.text_utils import strip_thinking_sections

LM_STUDIO_MODELS_PATH = "/api/v1/models"
LM_STUDIO_UNLOAD_PATH = "/api/v1/models/unload"


class EndpointPool:
    """Round-robin load balancer across multiple API base URLs.

    Usage:
        pool = EndpointPool(["http://server1:1234", "http://server2:1235"])
        base = pool.next()  # returns next healthy endpoint
    """

    def __init__(self, endpoints: list[str] | str | None = None) -> None:
        if endpoints is None:
            self._endpoints: list[str] = []
        elif isinstance(endpoints, str):
            self._endpoints = [e.strip() for e in endpoints.split(",") if e.strip()]
        else:
            self._endpoints = [e.strip() for e in endpoints if e.strip()]
        self._endpoints = [normalize_api_base(e) for e in self._endpoints]
        self._cycle = itertools.cycle(self._endpoints) if self._endpoints else None
        self._lock = threading.Lock()
        self._error_until: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._endpoints)

    @property
    def endpoints(self) -> list[str]:
        return list(self._endpoints)

    def next(self) -> str:
        """Get next endpoint via round-robin, skipping temporarily unhealthy ones."""
        if not self._endpoints:
            return normalize_api_base("")
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._endpoints)):
                assert self._cycle is not None
                ep = next(self._cycle)
                cooldown_until = self._error_until.get(ep, 0.0)
                if now >= cooldown_until:
                    return ep
            # All in cooldown — return the one with shortest remaining cooldown
            return min(self._endpoints, key=lambda e: self._error_until.get(e, 0.0))

    def mark_error(self, endpoint: str, cooldown_sec: float = 5.0) -> None:
        """Mark endpoint as temporarily unhealthy after an error."""
        with self._lock:
            self._error_until[endpoint] = time.monotonic() + cooldown_sec

    def mark_ok(self, endpoint: str) -> None:
        """Clear error state for an endpoint."""
        with self._lock:
            self._error_until.pop(endpoint, None)


def normalize_api_base(api_base: str) -> str:
    """Accept server root or pasted endpoint prefixes and return the server root."""
    base = str(api_base or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return base or DEFAULT_API_BASE.rstrip("/")


def _resolve_api_key(api_key: str | None) -> str:
    if api_key is not None:
        return api_key.strip()
    return str(DEFAULT_API_KEY).strip()


def _auth_headers_json(api_key: str | None) -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    key = _resolve_api_key(api_key)
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _auth_headers_get(api_key: str | None) -> dict[str, str]:
    key = _resolve_api_key(api_key)
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _raise_for_status_with_hint(response: requests.Response, endpoint: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        hint = ""
        if response.status_code in (401, 403):
            hint = " (LM Studio requires a valid API key; paste it in the app or set PHOTO_AI_SORTER_API_KEY)"
        elif response.status_code == 404:
            hint = " (check LM Studio API base URL; use the server root such as http://localhost:1234)"
        elif response.status_code == 400:
            body = response.text.strip().replace("\n", " ")[:360]
            if body:
                hint = f" (bad request detail: {body})"
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} at {endpoint}{hint}",
            response=response,
        ) from e


def endpoint_status(
    api_base: str,
    path: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> tuple[int | None, str]:
    base = normalize_api_base(api_base)
    url = f"{base}{path}"
    try:
        response = requests.get(url, headers=_auth_headers_get(api_key), timeout=timeout)
    except requests.RequestException as e:
        return None, str(e)
    return response.status_code, response.reason


_HEADER_AUTO = (
    "You are a local backend automated tagging script. Create stable short lowercase categories. "
    "The known tags below are reference examples, not a whitelist. Use them only when they are the best literal match. "
    "If a clearer category exists, create it. Use broad roots such as "
    "vehicles, humans, animals, nature, landscape, screenshots, memes, documents, art, "
    "illustration, food, travel, architecture, city, objects. Avoid near-duplicate words "
    "and overly specific one-off folders. Output one line with candidate tags separated by commas "
    "(most probable first). Example: vehicles/bmw, nature/forest, screenshots/chat."
)
_HEADER_FREE = (
    "You are a local backend automated tagging script. "
    "Your primary job is one concise lowercase hierarchical tag (slash-separated), "
    "e.g. art/photography/street or tech/desk/monitors. "
    "Use a preset tag from the list below ONLY when it is the best literal match; "
    "do NOT pick a preset just because its name appears inside your reasoning. "
    "When in doubt, output a new specific hierarchical path."
)

CUSTOM_CLASSIFICATION_GUIDANCE = """
You are an image classification expert.

Task:
You receive ONE image and a FIXED list of possible classes.
Each class has:
- a folder_name (machine label, lowercase with underscores)
- a natural language description that explains what images belong there.

Your job is to:
1) Carefully read ALL class descriptions.
2) Look at the image in detail.
3) Choose the SINGLE BEST class whose description matches the image content.
4) If you are not fully sure, still pick the closest class, but explain your reasoning.

Very important rules:
- Focus on the MAIN subject of the image, not small details in the background.
- If the image shows the user himself, always prioritize classes related to "iam" or "personal_user_*".
- If the image shows the user's personal pets, prioritize "my_cat" and "my_dog" classes over generic "cat" and "dog".
- Distinguish clearly between:
  - real photos vs AI-generated images
  - SFW (no explicit nudity/sex) vs NSFW (explicit nudity/sex)
  - furry/anthro characters vs real animals vs humans
  - canidae furry (wolves, foxes, dogs) vs other furry species
- If an image matches both SFW and NSFW classes, ALWAYS choose an NSFW class.
- If an image matches both a generic animal class and a more specific furry or personal pet class, ALWAYS choose the more specific class.
- If two classes are similarly likely, list both in top_candidates with honest confidence scores and set needs_review to true.
- Children must ALWAYS be assigned only to strictly SFW classes.
- Never invent new classes. Only choose from the given list.
""".strip()

_CUSTOM_STRUCTURED_OUTPUT_RULE = (
    "Respond ONLY with a single JSON object (no markdown fences, no extra text) using exactly these fields:\n"
    "{\n"
    '  "best_folder_name": "<one folder_name from CLASSES>",\n'
    '  "confidence": <number from 0 to 1>,\n'
    '  "top_candidates": [\n'
    '    {"folder_name": "<folder_name>", "confidence": <number from 0 to 1>},\n'
    "    ...\n"
    "  ],\n"
    '  "reasoning": "<short explanation in English why this class fits best>"\n'
    "}\n"
    f"If unsure, set best_folder_name to {UNCATEGORIZED}, confidence below 0.55, and explain why in reasoning.\n"
    "You may also use primary_category instead of best_folder_name (same value) for compatibility."
)

_CUSTOM_PLAIN_OUTPUT_RULE = (
    "Do not apologize, do not explain, do not refuse. Output ONLY the raw folder_name from CLASSES, "
    f"lowercase, no punctuation. If unsure, output {UNCATEGORIZED}."
)


def build_system_prompt(
    cfg: ResolvedTagConfig,
    *,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> str:
    """Build the classification system prompt from a resolved tag config."""
    tag_keys = cfg.categories

    if cfg.mode == TagMode.AUTO:
        header = _HEADER_AUTO
    elif cfg.mode == TagMode.FREE:
        header = _HEADER_FREE
    elif cfg.mode in (TagMode.CUSTOM, TagMode.HYBRID):
        header = CUSTOM_CLASSIFICATION_GUIDANCE
    else:
        header = (
            "You are a local backend automated tagging script. You have no conversational ability. "
            "Your ONLY job is to output a single exact string from the preset category list: "
            f"{', '.join(tag_keys)}."
        )

    if cfg.mode in (TagMode.CUSTOM, TagMode.HYBRID):
        output_rule = _CUSTOM_STRUCTURED_OUTPUT_RULE if structured_output else _CUSTOM_PLAIN_OUTPUT_RULE
    elif structured_output:
        output_rule = (
            "Output a single compact JSON object with keys: primary_category, candidates, confidence, "
            "reason_short, needs_review. primary_category and candidates must use lowercase tag strings only. "
            "confidence is 0..1. reason_short must be short. If unsure, set primary_category to "
            f"{UNCATEGORIZED}, confidence below 0.55, and needs_review true."
        )
    else:
        output_rule = (
            "Do not apologize, do not explain, do not refuse. Output ONLY the raw tag, lowercase, "
            f"no punctuation. If unsure, output {UNCATEGORIZED}."
        )

    parts: list[str] = [header, output_rule]
    if cfg.mode not in (TagMode.CUSTOM, TagMode.HYBRID):
        parts.append("Do not write long reasoning. If you must think, put only the final answer after it.")
    parts.append("")

    if cfg.priority_rules_text:
        parts.append(cfg.priority_rules_text)
        parts.append("")

    if cfg.is_free:
        parts.append("Reference definitions (optional, not a whitelist):")
    elif cfg.mode in (TagMode.CUSTOM, TagMode.HYBRID):
        parts.append("CLASSES (folder_name: description):")
    else:
        parts.append("Definitions (tag: meaning):")

    for cat in tag_keys:
        desc = cfg.prompts.get(cat, cat.replace("_", " "))
        parts.append(f"{cat}: {desc}")
    parts.append("")

    if cfg.user_context.strip():
        parts.append(
            "User context for recognition (match these descriptions to identify personal subjects):\n"
            f"{cfg.user_context}"
        )
    else:
        parts.append("User context: not provided. Use only visual content for classification.")

    if prompt_extra.strip():
        parts.append("")
        parts.append("Extra prompt settings:")
        parts.append(prompt_extra.strip())

    return "\n".join(parts)


def build_classification_system_prompt(
    user_context: str,
    *,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> str:
    """Legacy wrapper — delegates to build_system_prompt(ResolvedTagConfig)."""
    from app.tag_config import resolve_tag_config
    if custom_categories:
        mode = TagMode.CUSTOM
    elif auto_mode:
        mode = TagMode.AUTO
    elif free_mode:
        mode = TagMode.FREE
    elif general_mode:
        mode = TagMode.PRESET
    else:
        mode = TagMode.PRESET
    cfg = resolve_tag_config(mode, user_context_override=user_context)
    if custom_categories:
        cfg = ResolvedTagConfig(
            mode=TagMode.CUSTOM,
            categories=custom_categories,
            prompts=custom_prompts or {},
            user_context=user_context,
        )
    return build_system_prompt(cfg, prompt_extra=prompt_extra, structured_output=structured_output)


def _retryable_request_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "channel error" in msg or "channel closed" in msg:
        return True
    if "model reloaded" in msg or "model crashed" in msg:
        return True
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code in (502, 503, 429):
            return True
        if resp is not None and resp.status_code == 400:
            body = (resp.text or "").lower()
            if "model reloaded" in body or "model crashed" in body or "exit code" in body:
                return True
        if resp is not None and resp.status_code == 500:
            return True
    if isinstance(exc, ValueError):
        return True
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return True
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    return False


def _is_model_reload_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "model reloaded" in msg or "model crashed" in msg or "exit code" in msg


def _run_completion_retries(
    operation: Callable[[], str],
    *,
    on_retry: Callable[[str], None] | None,
    attempt_label: str,
    max_retries: int | None = None,
) -> str:
    retries = max_retries if max_retries is not None else API_MAX_RETRIES
    last_exc: BaseException | None = None
    immediate_retry_done = False
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as e:
            last_exc = e
            if not _retryable_request_error(e):
                raise
            if attempt >= retries:
                break
            msg = str(e).lower()
            is_channel_error = "channel error" in msg or "channel closed" in msg
            is_reload = _is_model_reload_error(e)

            if is_reload:
                delay = min(12.0, 3.0 + attempt * 2.0)
                if on_retry:
                    on_retry(
                        f"{attempt_label} {attempt}/{retries}: model reload/crash detected, "
                        f"waiting {delay:.1f}s for recovery..."
                    )
                time.sleep(delay)
                continue

            if is_channel_error and not immediate_retry_done:
                immediate_retry_done = True
                if on_retry:
                    on_retry("Channel Error detected, applying transient backoff (fast retry).")
                time.sleep(0.5)
                continue

            delay = (
                API_RETRY_BACKOFF_SEC[attempt - 1]
                if attempt - 1 < len(API_RETRY_BACKOFF_SEC)
                else API_RETRY_BACKOFF_SEC[-1]
            )
            jittered_delay = min(8.0, max(0.1, delay * (0.75 + random.random() * 0.5)))
            if on_retry:
                on_retry(
                    f"{attempt_label} {attempt}/{retries}: {e!s} "
                    f"(через {jittered_delay:.1f} с)"
                )
            time.sleep(jittered_delay)
    assert last_exc is not None
    raise last_exc


def _normalize_content_field(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content)


_strip_thinking_sections = strip_thinking_sections


def _extract_assistant_text(msg: dict[str, Any]) -> str:
    """
    OpenAI-совместимый ответ: content. У Qwen3/LM Studio часто пусто, текст в reasoning_content.
    """
    text = _strip_thinking_sections(_normalize_content_field(msg.get("content")).strip())
    if text:
        return text
    for key in ("reasoning_content", "reasoning", "thought"):
        raw = msg.get(key)
        if isinstance(raw, str) and raw.strip():
            cleaned = _strip_thinking_sections(raw.strip())
            if cleaned:
                return cleaned
    return ""


def _user_text_single(structured: bool, *, cfg: ResolvedTagConfig | None = None) -> str:
    if structured and cfg is not None and cfg.mode in (TagMode.CUSTOM, TagMode.HYBRID):
        return (
            "Classify the attached image using only the CLASSES list above. "
            "Return one JSON object only (best_folder_name, confidence, top_candidates, reasoning)."
        )
    if structured:
        return (
            "Classify this image. Return one compact JSON object only. "
            "Use primary_category plus candidates sorted best first."
        )
    return (
        "Classify this image. Output exactly one final tag only, no explanation. "
        "In free mode this may be a hierarchical slash-separated tag. "
        "In auto mode output comma-separated candidates with the best first."
    )


def _user_text_multi(structured: bool, *, cfg: ResolvedTagConfig | None = None) -> str:
    if structured and cfg is not None and cfg.mode in (TagMode.CUSTOM, TagMode.HYBRID):
        return (
            "These images are frames from the same video file in chronological order. "
            "Classify the entire content using only the CLASSES list above. "
            "Return one JSON object only (best_folder_name, confidence, top_candidates, reasoning)."
        )
    if structured:
        return (
            "These images are frames from the same video file in chronological order. "
            "Classify the entire content. Return one compact JSON object only."
        )
    return (
        "These images are frames from the same video file in chronological order. "
        "Classify the entire content with exactly one final tag only, no explanation. "
        "In free mode this may be a hierarchical slash-separated tag. "
        "In auto mode output comma-separated candidates with the best first."
    )


# --- Config-based API (new, clean) ---


def build_messages_cfg(
    image_data_uri: str,
    cfg: ResolvedTagConfig,
    *,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> list[dict[str, Any]]:
    """Build single-image messages from a ResolvedTagConfig."""
    system_text = build_system_prompt(cfg, prompt_extra=prompt_extra, structured_output=structured_output)
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _user_text_single(structured_output, cfg=cfg)},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


def build_messages_multi_cfg(
    image_data_uris: list[str],
    cfg: ResolvedTagConfig,
    *,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> list[dict[str, Any]]:
    """Build multi-frame messages from a ResolvedTagConfig."""
    system_text = build_system_prompt(cfg, prompt_extra=prompt_extra, structured_output=structured_output)
    user_parts: list[dict[str, Any]] = [
        {"type": "text", "text": _user_text_multi(structured_output, cfg=cfg)},
    ]
    for uri in image_data_uris:
        user_parts.append({"type": "image_url", "image_url": {"url": uri}})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_parts},
    ]


def _completion_once_cfg(
    messages: list[dict[str, Any]],
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    timeout: tuple[float, float] | float,
    temperature: float,
    max_tokens: int,
) -> str:
    """Low-level single request (no retries)."""
    base = normalize_api_base(api_base)
    url = f"{base}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps(payload),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, CHAT_COMPLETIONS_PATH)
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON response: {e}") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    msg = choices[0].get("message") or {}
    text = _extract_assistant_text(msg)
    if not text:
        raise ValueError("empty assistant message (no content and no reasoning_content)")
    return text


def chat_completion_cfg(
    image_data_uri: str,
    cfg: ResolvedTagConfig,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
    temperature: float = 0.2,
    max_tokens: int = CHAT_COMPLETION_MAX_TOKENS,
) -> str:
    """Config-based single-image classification with retries."""
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)
    msgs = build_messages_cfg(image_data_uri, cfg, prompt_extra=prompt_extra, structured_output=structured_output)
    return _run_completion_retries(
        lambda: _completion_once_cfg(
            msgs,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=t,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        on_retry=on_retry,
        attempt_label="Повтор",
    )


def chat_completion_multi_cfg(
    image_data_uris: list[str],
    cfg: ResolvedTagConfig,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
    temperature: float = 0.2,
    max_tokens: int = CHAT_COMPLETION_MAX_TOKENS,
) -> str:
    """Config-based multi-image classification with retries."""
    if not image_data_uris:
        raise ValueError("no images")
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)
    msgs = build_messages_multi_cfg(image_data_uris, cfg, prompt_extra=prompt_extra, structured_output=structured_output)
    return _run_completion_retries(
        lambda: _completion_once_cfg(
            msgs,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=t,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        on_retry=on_retry,
        attempt_label="Повтор multi",
    )


def classify_frames_cfg(
    image_data_uris: list[str],
    cfg: ResolvedTagConfig,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    prompt_extra: str = "",
) -> str:
    """Config-based frame classification with multi→single fallback cascade."""
    from app.categorizer import merge_tags_by_priority, normalize_tag, normalize_tag_auto, normalize_tag_free

    def _normalize(raw: str) -> str:
        if cfg.mode == TagMode.AUTO:
            return normalize_tag_auto(raw)
        if cfg.mode == TagMode.FREE:
            return normalize_tag_free(raw)
        wl = cfg.whitelist or GENERAL_CATEGORY_WHITELIST
        return normalize_tag(raw, whitelist=wl)

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    uris = [u for u in image_data_uris if u][:VIDEO_FRAME_COUNT]
    if not uris:
        return UNCATEGORIZED
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)

    if len(uris) == 1:
        try:
            raw = chat_completion_cfg(uris[0], cfg, api_base=api_base, model=model, api_key=api_key,
                                      timeout=t, on_retry=on_retry, prompt_extra=prompt_extra)
            return _normalize(raw)
        except Exception as e:
            log(f"rollback: single frame: {e!s}")
            return UNCATEGORIZED

    multi_candidates: list[list[str]] = []
    if len(uris) >= 3:
        multi_candidates.append(uris[:3])
    if len(uris) >= 2:
        multi_candidates.append(uris[:2])
    multi_candidates.append(uris[:1])

    seen: set[tuple[str, ...]] = set()
    for subset in multi_candidates:
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)
        if len(subset) >= 2:
            try:
                raw = chat_completion_multi_cfg(subset, cfg, api_base=api_base, model=model,
                                               api_key=api_key, timeout=t, on_retry=on_retry,
                                               prompt_extra=prompt_extra)
                return _normalize(raw)
            except Exception as e:
                log(f"rollback: multi n={len(subset)}: {e!s}")
        else:
            try:
                raw = chat_completion_cfg(subset[0], cfg, api_base=api_base, model=model,
                                         api_key=api_key, timeout=t, on_retry=on_retry,
                                         prompt_extra=prompt_extra)
                return _normalize(raw)
            except Exception as e:
                log(f"rollback: single after multi fail: {e!s}")

    tags: list[str] = []
    for i, u in enumerate(uris):
        try:
            raw = chat_completion_cfg(u, cfg, api_base=api_base, model=model, api_key=api_key,
                                      timeout=t, on_retry=on_retry, prompt_extra=prompt_extra)
            tags.append(_normalize(raw))
        except Exception as e:
            log(f"rollback: per-frame {i}: {e!s}")

    if not tags:
        return UNCATEGORIZED
    if cfg.is_free:
        counts: dict[str, int] = {}
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts, key=lambda k: (-counts[k], len(k), k))[0]
    wl = cfg.whitelist or GENERAL_CATEGORY_WHITELIST
    return merge_tags_by_priority(tags, whitelist=wl)


# --- Legacy API (preserved for backward compat / tests) ---


def _legacy_resolved_cfg(
    user_context: str,
    *,
    free_mode: bool,
    auto_mode: bool,
    general_mode: bool,
    custom_categories: tuple[str, ...] | None,
    custom_prompts: dict[str, str] | None,
) -> ResolvedTagConfig:
    if custom_categories:
        return ResolvedTagConfig(
            mode=TagMode.CUSTOM,  # legacy API
            categories=custom_categories,
            prompts=custom_prompts or {},
            whitelist=frozenset(custom_categories),
        )
    from app.tag_config import resolve_tag_config

    if auto_mode:
        mode = TagMode.AUTO
    elif free_mode:
        mode = TagMode.FREE
    else:
        mode = TagMode.PRESET
    return resolve_tag_config(mode, user_context_override=user_context)


def build_messages(
    image_data_uri: str,
    user_context: str,
    *,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> list[dict[str, Any]]:
    """Legacy: System + user with vision image_url (OpenAI-compatible)."""
    legacy_cfg = _legacy_resolved_cfg(
        user_context,
        free_mode=free_mode,
        auto_mode=auto_mode,
        general_mode=general_mode,
        custom_categories=custom_categories,
        custom_prompts=custom_prompts,
    )
    system_text = build_classification_system_prompt(
        user_context,
        free_mode=free_mode,
        auto_mode=auto_mode,
        general_mode=general_mode,
        custom_categories=custom_categories,
        custom_prompts=custom_prompts,
        prompt_extra=prompt_extra,
        structured_output=structured_output,
    )
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _user_text_single(structured_output, cfg=legacy_cfg)},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


def build_messages_multi(
    image_data_uris: list[str],
    user_context: str,
    *,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
) -> list[dict[str, Any]]:
    """Legacy: Several frames from one video/GIF — один тег на весь контент."""
    legacy_cfg = _legacy_resolved_cfg(
        user_context,
        free_mode=free_mode,
        auto_mode=auto_mode,
        general_mode=general_mode,
        custom_categories=custom_categories,
        custom_prompts=custom_prompts,
    )
    system_text = build_classification_system_prompt(
        user_context,
        free_mode=free_mode,
        auto_mode=auto_mode,
        general_mode=general_mode,
        custom_categories=custom_categories,
        custom_prompts=custom_prompts,
        prompt_extra=prompt_extra,
        structured_output=structured_output,
    )
    user_parts: list[dict[str, Any]] = [
        {"type": "text", "text": _user_text_multi(structured_output, cfg=legacy_cfg)},
    ]
    for uri in image_data_uris:
        user_parts.append({"type": "image_url", "image_url": {"url": uri}})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_parts},
    ]


def _chat_completion_once(
    image_data_uri: str,
    user_context: str,
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    timeout: tuple[float, float] | float,
    free_mode: bool,
    auto_mode: bool,
    general_mode: bool,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str,
    structured_output: bool,
    temperature: float,
    max_tokens: int,
) -> str:
    base = normalize_api_base(api_base)
    url = f"{base}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": build_messages(
            image_data_uri,
            user_context,
            free_mode=free_mode,
            auto_mode=auto_mode,
            general_mode=general_mode,
            custom_categories=custom_categories,
            custom_prompts=custom_prompts,
            prompt_extra=prompt_extra,
            structured_output=structured_output,
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps(payload),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, CHAT_COMPLETIONS_PATH)
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON response: {e}") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    msg = choices[0].get("message") or {}
    text = _extract_assistant_text(msg)
    if not text:
        raise ValueError("empty assistant message (no content and no reasoning_content)")
    return text


def _chat_completion_multi_once(
    image_data_uris: list[str],
    user_context: str,
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    timeout: tuple[float, float] | float,
    free_mode: bool,
    auto_mode: bool,
    general_mode: bool,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str,
    structured_output: bool,
    temperature: float,
    max_tokens: int,
) -> str:
    if not image_data_uris:
        raise ValueError("no images")
    base = normalize_api_base(api_base)
    url = f"{base}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": build_messages_multi(
            image_data_uris,
            user_context,
            free_mode=free_mode,
            auto_mode=auto_mode,
            general_mode=general_mode,
            custom_categories=custom_categories,
            custom_prompts=custom_prompts,
            prompt_extra=prompt_extra,
            structured_output=structured_output,
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps(payload),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, CHAT_COMPLETIONS_PATH)
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON response: {e}") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    msg = choices[0].get("message") or {}
    text = _extract_assistant_text(msg)
    if not text:
        raise ValueError("empty assistant message (no content and no reasoning_content)")
    return text


def chat_completion_multi(
    image_data_uris: list[str],
    user_context: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
    temperature: float = 0.2,
    max_tokens: int = CHAT_COMPLETION_MAX_TOKENS,
) -> str:
    """Multi-image POST с теми же ретраями, что и chat_completion."""
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)
    return _run_completion_retries(
        lambda: _chat_completion_multi_once(
            image_data_uris,
            user_context,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=t,
            free_mode=free_mode,
            auto_mode=auto_mode,
            general_mode=general_mode,
            custom_categories=custom_categories,
            custom_prompts=custom_prompts,
            prompt_extra=prompt_extra,
            structured_output=structured_output,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        on_retry=on_retry,
        attempt_label="Повтор multi",
    )


def classify_frames(
    image_data_uris: list[str],
    user_context: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
) -> str:
    """
    Классификация по кадрам: multi-image → сужение до 1 кадра → покадровые запросы + merge по приоритету.
    Возвращает нормализованный тег (строка whitelist).
    """
    from app.categorizer import merge_tags_by_priority, normalize_tag, normalize_tag_auto, normalize_tag_free

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    uris = [u for u in image_data_uris if u][:VIDEO_FRAME_COUNT]
    if not uris:
        return UNCATEGORIZED
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)

    if len(uris) == 1:
        try:
            raw = chat_completion(
                uris[0],
                user_context,
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout=t,
                on_retry=on_retry,
                free_mode=free_mode,
                auto_mode=auto_mode,
                general_mode=general_mode,
                custom_categories=custom_categories,
                custom_prompts=custom_prompts,
                prompt_extra=prompt_extra,
            )
            if auto_mode:
                return normalize_tag_auto(raw)
            if free_mode:
                return normalize_tag_free(raw)
            if general_mode:
                return normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)
            return normalize_tag(raw)
        except Exception as e:
            log(f"rollback: single frame: {e!s}")
            return UNCATEGORIZED

    multi_candidates: list[list[str]] = []
    if len(uris) >= 3:
        multi_candidates.append(uris[:3])
    if len(uris) >= 2:
        multi_candidates.append(uris[:2])
    multi_candidates.append(uris[:1])

    seen: set[tuple[str, ...]] = set()
    for subset in multi_candidates:
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)
        if len(subset) >= 2:
            try:
                raw = chat_completion_multi(
                    subset,
                    user_context,
                    api_base=api_base,
                    model=model,
                    api_key=api_key,
                    timeout=t,
                    on_retry=on_retry,
                    free_mode=free_mode,
                    auto_mode=auto_mode,
                    general_mode=general_mode,
                    custom_categories=custom_categories,
                    custom_prompts=custom_prompts,
                    prompt_extra=prompt_extra,
                )
                if auto_mode:
                    return normalize_tag_auto(raw)
                if free_mode:
                    return normalize_tag_free(raw)
                if general_mode:
                    return normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)
                return normalize_tag(raw)
            except Exception as e:
                log(f"rollback: multi n={len(subset)}: {e!s}")
        else:
            try:
                raw = chat_completion(
                    subset[0],
                    user_context,
                    api_base=api_base,
                    model=model,
                    api_key=api_key,
                    timeout=t,
                    on_retry=on_retry,
                    free_mode=free_mode,
                    auto_mode=auto_mode,
                    general_mode=general_mode,
                    custom_categories=custom_categories,
                    custom_prompts=custom_prompts,
                    prompt_extra=prompt_extra,
                )
                if auto_mode:
                    return normalize_tag_auto(raw)
                if free_mode:
                    return normalize_tag_free(raw)
                if general_mode:
                    return normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST)
                return normalize_tag(raw)
            except Exception as e:
                log(f"rollback: single after multi fail: {e!s}")

    tags: list[str] = []
    for i, u in enumerate(uris):
        try:
            raw = chat_completion(
                u,
                user_context,
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout=t,
                on_retry=on_retry,
                free_mode=free_mode,
                auto_mode=auto_mode,
                general_mode=general_mode,
                custom_categories=custom_categories,
                custom_prompts=custom_prompts,
                prompt_extra=prompt_extra,
            )
            if auto_mode:
                tags.append(normalize_tag_auto(raw))
            elif free_mode:
                tags.append(normalize_tag_free(raw))
            elif general_mode:
                tags.append(normalize_tag(raw, whitelist=GENERAL_CATEGORY_WHITELIST))
            else:
                tags.append(normalize_tag(raw))
        except Exception as e:
            log(f"rollback: per-frame {i}: {e!s}")
    if not tags:
        return UNCATEGORIZED
    if free_mode or auto_mode:
        counts: dict[str, int] = {}
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
        # Prefer the most frequent free tag; tie-breaker by shortest (more general) then lexical.
        return sorted(counts, key=lambda k: (-counts[k], len(k), k))[0]
    return merge_tags_by_priority(tags, whitelist=GENERAL_CATEGORY_WHITELIST if general_mode else None)


def chat_completion(
    image_data_uri: str,
    user_context: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
    free_mode: bool = False,
    auto_mode: bool = False,
    general_mode: bool = False,
    custom_categories: tuple[str, ...] | None = None,
    custom_prompts: dict[str, str] | None = None,
    prompt_extra: str = "",
    structured_output: bool = False,
    temperature: float = 0.2,
    max_tokens: int = CHAT_COMPLETION_MAX_TOKENS,
) -> str:
    """
    POST /v1/chat/completions с ретраями при таймаутах/502/503/429 и пустом ответе.
    timeout: (connect, read); по умолчанию из constants.
    """
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)
    return _run_completion_retries(
        lambda: _chat_completion_once(
            image_data_uri,
            user_context,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=t,
            free_mode=free_mode,
            auto_mode=auto_mode,
            general_mode=general_mode,
            custom_categories=custom_categories,
            custom_prompts=custom_prompts,
            prompt_extra=prompt_extra,
            structured_output=structured_output,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        on_retry=on_retry,
        attempt_label="Повтор",
    )


def build_vision_probe_messages(image_data_uri: str) -> list[dict[str, Any]]:
    """Lightweight prompt for vision channel test (no heavy tag list)."""
    return [
        {
            "role": "system",
            "content": (
                "You are a vision API connectivity test. Answer briefly in English. "
                "Describe what you see in one short sentence."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colors and shapes are visible? Reply in one sentence."},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


def _vision_probe_once(
    image_data_uri: str,
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    timeout: tuple[float, float] | float,
) -> str:
    base = normalize_api_base(api_base)
    url = f"{base}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": build_vision_probe_messages(image_data_uri),
        "temperature": 0.2,
        "max_tokens": VISION_PROBE_MAX_TOKENS,
    }
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps(payload),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, CHAT_COMPLETIONS_PATH)
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    msg = choices[0].get("message") or {}
    text = _extract_assistant_text(msg)
    if not text:
        raise ValueError("empty assistant message (no content and no reasoning_content)")
    return text


def vision_probe_completion(
    image_data_uri: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
) -> str:
    """POST chat/completions with a small probe prompt (not full classification)."""
    t = timeout if timeout is not None else (API_PROBE_TIMEOUT_SEC, VISION_TEST_TIMEOUT_SEC)
    last_exc: BaseException | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return _vision_probe_once(
                image_data_uri,
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout=t,
            )
        except Exception as e:
            last_exc = e
            if not _retryable_request_error(e):
                raise
            if attempt >= API_MAX_RETRIES:
                break
            delay = (
                API_RETRY_BACKOFF_SEC[attempt - 1]
                if attempt - 1 < len(API_RETRY_BACKOFF_SEC)
                else API_RETRY_BACKOFF_SEC[-1]
            )
            if on_retry:
                on_retry(f"Повтор probe {attempt}/{API_MAX_RETRIES}: {e!s} (через {delay:.0f} с)")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def list_models(
    api_base: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> list[str]:
    """GET /v1/models — OpenAI-compatible model ids."""
    base = normalize_api_base(api_base)
    url = f"{base}{MODELS_PATH}"
    r = requests.get(url, headers=_auth_headers_get(api_key), timeout=timeout)
    _raise_for_status_with_hint(r, MODELS_PATH)
    data = r.json()
    items = data.get("data") or []
    ids: list[str] = []
    for m in items:
        if isinstance(m, dict) and m.get("id"):
            ids.append(str(m["id"]))
    return sorted(set(ids), key=str.lower)


def list_lm_studio_models(
    api_base: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """LM Studio native model inventory with loaded instance metadata."""
    base = normalize_api_base(api_base)
    url = f"{base}{LM_STUDIO_MODELS_PATH}"
    r = requests.get(url, headers=_auth_headers_get(api_key), timeout=timeout)
    _raise_for_status_with_hint(r, LM_STUDIO_MODELS_PATH)
    data = r.json()
    models = data.get("models") or data.get("data") or []
    return [m for m in models if isinstance(m, dict)]


def loaded_model_instances(
    api_base: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """Flatten LM Studio loaded_instances into rows useful for UI and cleanup."""
    rows: list[dict[str, Any]] = []
    for model in list_lm_studio_models(api_base, api_key=api_key, timeout=timeout):
        key = str(model.get("key") or model.get("id") or "")
        loaded = model.get("loaded_instances") or []
        if not loaded and str(model.get("state") or "").lower() == "loaded":
            loaded = [{"id": key, "config": {"context_length": model.get("loaded_context_length")}}]
        if not isinstance(loaded, list):
            continue
        capabilities = model.get("capabilities") or {}
        vision = capabilities.get("vision") if isinstance(capabilities, dict) else None
        for inst in loaded:
            if not isinstance(inst, dict):
                continue
            cfg = inst.get("config") if isinstance(inst.get("config"), dict) else {}
            rows.append(
                {
                    "model_key": key,
                    "instance_id": str(inst.get("id") or key),
                    "display_name": str(model.get("display_name") or key),
                    "context_length": cfg.get("context_length"),
                    "parallel": cfg.get("parallel"),
                    "remaining_ttl_seconds": inst.get("remaining_ttl_seconds"),
                    "vision": vision,
                }
            )
    return rows


def unload_model_instance(
    api_base: str,
    instance_id: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> str:
    """Unload one LM Studio loaded model instance by instance_id."""
    instance_id = str(instance_id or "").strip()
    if not instance_id:
        raise ValueError("empty instance_id")
    base = normalize_api_base(api_base)
    url = f"{base}{LM_STUDIO_UNLOAD_PATH}"
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps({"instance_id": instance_id}),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, LM_STUDIO_UNLOAD_PATH)
    data = r.json() if r.text.strip() else {}
    return str(data.get("instance_id") or instance_id)


def unload_duplicate_model_instances(
    api_base: str,
    *,
    keep_models: set[str],
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> list[str]:
    """
    Unload LM Studio instances that are not needed for the active app workers.

    For active model keys, keeps an exact active instance id when possible,
    otherwise keeps the first loaded instance and unloads duplicates. For inactive
    model keys, unloads all loaded instances.
    """
    keep_models = {str(m).strip() for m in keep_models if str(m).strip()}
    keep_keys = {m.rsplit(":", 1)[0] if m.rsplit(":", 1)[-1].isdigit() else m for m in keep_models}
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in loaded_model_instances(api_base, api_key=api_key, timeout=timeout):
        by_key.setdefault(str(row.get("model_key") or ""), []).append(row)
    unloaded: list[str] = []
    for key, rows in by_key.items():
        is_active_key = key in keep_keys or any(str(r.get("instance_id") or "") in keep_models for r in rows)
        if not is_active_key:
            keep_id = ""
        else:
            exact_keep = next((r for r in rows if str(r.get("instance_id") or "") in keep_models), None)
            keep = exact_keep or rows[0]
            keep_id = str(keep.get("instance_id") or "")
        for row in rows:
            instance_id = str(row.get("instance_id") or "")
            if not instance_id or instance_id == keep_id:
                continue
            unloaded.append(unload_model_instance(api_base, instance_id, api_key=api_key, timeout=timeout))
    return unloaded


def vision_hint_from_model_dict(obj: dict[str, Any]) -> bool | None:
    """
    If server exposes vision capability in /v1/models entry, return True/False.
    Otherwise None (need runtime self-test).
    """
    if not isinstance(obj, dict):
        return None
    for key in ("modalities", "capabilities"):
        val = obj.get(key)
        if isinstance(val, list):
            joined = " ".join(str(x).lower() for x in val)
            if "vision" in joined or "image" in joined or "multimodal" in joined:
                return True
            if val and all(str(x).lower() in ("text", "language") for x in val):
                return False
    v = obj.get("vision")
    if isinstance(v, bool):
        return v
    arch = str(obj.get("architecture") or "").lower()
    if "vision" in arch or "vl" in arch or "mm" in arch:
        return True
    return None


def find_model_object(
    api_base: str,
    model_id: str,
    *,
    api_key: str | None = None,
    timeout: float = API_PROBE_TIMEOUT_SEC,
) -> dict[str, Any] | None:
    base = normalize_api_base(api_base)
    url = f"{base}{MODELS_PATH}"
    r = requests.get(url, headers=_auth_headers_get(api_key), timeout=timeout)
    _raise_for_status_with_hint(r, MODELS_PATH)
    data = r.json()
    for m in data.get("data") or []:
        if isinstance(m, dict) and str(m.get("id")) == model_id:
            return m
    return None


def vision_self_test(
    api_base: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout: float = VISION_TEST_TIMEOUT_SEC,
    image_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    Vision POST with a synthetic test card or a user-selected image file.
    Uses a lightweight probe prompt (not full tag classification) to reduce server errors.
    """
    from app.images import image_to_jpeg_base64_data_uri, pil_image_to_jpeg_data_uri, vision_test_card_data_uri

    if image_path is not None:
        p = Path(image_path)
        if not p.is_file():
            return False, f"Файл не найден: {p}"
        try:
            suf = p.suffix.lower()
            if suf in VIDEO_EXTENSIONS or suf == GIF_EXTENSION:
                from app.video_frames import extract_frames_for_classification, is_animated_gif

                if suf == GIF_EXTENSION and not is_animated_gif(p):
                    uri = image_to_jpeg_base64_data_uri(p)
                else:
                    frames = extract_frames_for_classification(p, 1, on_log=lambda _m: None)
                    if not frames:
                        return False, "Не удалось извлечь кадр (установите ffmpeg или проверьте файл)."
                    uri = pil_image_to_jpeg_data_uri(frames[0])
            else:
                uri = image_to_jpeg_base64_data_uri(p)
        except Exception as e:
            return False, str(e)
    else:
        uri = vision_test_card_data_uri()
    try:
        text = vision_probe_completion(
            uri,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        return True, (text[:400] + ("…" if len(text) > 400 else ""))
    except Exception as e:
        return False, str(e)


def full_api_self_test(
    api_base: str,
    model: str,
    *,
    api_key: str | None = None,
    image_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    1) GET /v1/models
    2) Optional metadata vision hint
    3) vision_self_test with selected model
    Returns (ok, human-readable report).
    """
    lines: list[str] = []
    lines.append(f"Base URL: {normalize_api_base(api_base)}")
    lines.append("Auth: " + ("API key present" if _resolve_api_key(api_key) else "no API key"))
    for path in (MODELS_PATH, "/api/v1/models"):
        code, reason = endpoint_status(api_base, path, api_key=api_key)
        status = str(code) if code is not None else "network error"
        lines.append(f"Endpoint GET {path}: {status} {reason}".strip())
    try:
        models = list_models(api_base, api_key=api_key)
        lines.append(f"Моделей в /v1/models: {len(models)}")
    except Exception as e:
        return False, f"GET /v1/models: {e!s}"

    if model:
        meta = find_model_object(api_base, model, api_key=api_key)
        if meta:
            hint = vision_hint_from_model_dict(meta)
            if hint is True:
                lines.append("Метаданные: похоже на vision-модель.")
            elif hint is False:
                lines.append("Метаданные: похоже на текстовую модель.")
            else:
                lines.append("Метаданные: признак vision не указан — проверка запросом.")
        else:
            lines.append(f"Модель «{model}» не найдена в списке (всё равно пробуем vision).")

    lines.append(
        "Проверка vision: встроенная тест-карта"
        if image_path is None
        else f"Проверка vision: файл {image_path}"
    )
    ok, detail = vision_self_test(
        api_base,
        model or DEFAULT_MODEL,
        api_key=api_key,
        image_path=image_path,
    )
    lines.append("Vision POST /v1/chat/completions (probe): " + ("OK" if ok else "ошибка"))
    lines.append(detail)
    return ok, "\n".join(lines)


def build_duplicate_pair_messages(uri_a: str, uri_b: str) -> list[dict[str, Any]]:
    system = (
        "You compare two images. If they show the same scene, the same photo, or duplicate/near-duplicate content, "
        "reply YES. If they are clearly different images, reply NO. Output exactly one word: YES or NO."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First image:"},
                {"type": "image_url", "image_url": {"url": uri_a}},
                {"type": "text", "text": "Second image:"},
                {"type": "image_url", "image_url": {"url": uri_b}},
            ],
        },
    ]


def _pair_dup_once(
    uri_a: str,
    uri_b: str,
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    timeout: tuple[float, float] | float,
) -> str:
    base = normalize_api_base(api_base)
    url = f"{base}{CHAT_COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": build_duplicate_pair_messages(uri_a, uri_b),
        "temperature": 0.1,
        "max_tokens": 16,
    }
    r = requests.post(
        url,
        headers=_auth_headers_json(api_key),
        data=json.dumps(payload),
        timeout=timeout,
    )
    _raise_for_status_with_hint(r, CHAT_COMPLETIONS_PATH)
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON response: {e}") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    msg = choices[0].get("message") or {}
    text = _extract_assistant_text(msg)
    if not text:
        raise ValueError("empty assistant message (no content and no reasoning_content)")
    return text


def pair_images_duplicate_decision(
    uri_a: str,
    uri_b: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: tuple[float, float] | None = None,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Vision API: whether two images are duplicates / same scene (YES/NO)."""
    t = timeout if timeout is not None else (REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_READ_TIMEOUT_SEC)
    raw = _run_completion_retries(
        lambda: _pair_dup_once(
            uri_a,
            uri_b,
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=t,
        ),
        on_retry=on_retry,
        attempt_label="Повтор dup pair",
    )
    u = _strip_thinking_sections(raw).strip().upper()
    words = u.split()
    first = words[0] if words else ""
    if first.startswith("N") or "NO" == first:
        return False
    if first.startswith("Y") or first == "YES":
        return True
    return False

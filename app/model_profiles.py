"""LM Studio model profiles and lightweight benchmark helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    api_base: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_sec: float = 600.0
    workers: int = 3
    api_workers: int = 1
    prompt_extra: str = ""


def default_profiles(api_base: str, model: str) -> dict[str, ModelProfile]:
    base = str(api_base or "").strip()
    mid = str(model or "").strip()
    return {
        "classifier": ModelProfile("classifier", "sort classification", base, mid, workers=3, api_workers=1),
        "duplicate_verifier": ModelProfile(
            "duplicate_verifier",
            "ambiguous duplicate pair verification",
            base,
            mid,
            temperature=0.0,
            max_tokens=256,
            timeout_sec=240.0,
            workers=2,
            api_workers=1,
        ),
        "screenshot_ocr": ModelProfile(
            "screenshot_ocr",
            "screenshots and text-heavy images",
            base,
            mid,
            temperature=0.0,
            max_tokens=768,
            timeout_sec=240.0,
            workers=2,
            api_workers=1,
            prompt_extra="Pay attention to UI text, app screenshots, chat captures, and memes.",
        ),
        "fast_preview": ModelProfile(
            "fast_preview",
            "quick low-cost sampling",
            base,
            mid,
            temperature=0.0,
            max_tokens=256,
            timeout_sec=120.0,
            workers=1,
            api_workers=1,
        ),
    }


def profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    return asdict(profile)


def profile_from_dict(name: str, raw: Any, fallback: ModelProfile) -> ModelProfile:
    if not isinstance(raw, dict):
        return fallback
    return ModelProfile(
        name=str(raw.get("name") or name or fallback.name),
        role=str(raw.get("role") or fallback.role),
        api_base=str(raw.get("api_base") or fallback.api_base),
        model=str(raw.get("model") or fallback.model),
        temperature=max(0.0, min(2.0, float(raw.get("temperature", fallback.temperature)))),
        max_tokens=max(1, min(4096, int(raw.get("max_tokens", fallback.max_tokens)))),
        timeout_sec=max(5.0, min(1800.0, float(raw.get("timeout_sec", fallback.timeout_sec)))),
        workers=max(1, min(16, int(raw.get("workers", fallback.workers)))),
        api_workers=max(1, min(16, int(raw.get("api_workers", fallback.api_workers)))),
        prompt_extra=str(raw.get("prompt_extra") or fallback.prompt_extra),
    )


def merge_profiles(raw: Any, *, api_base: str, model: str) -> dict[str, ModelProfile]:
    defaults = default_profiles(api_base, model)
    if not isinstance(raw, dict):
        return defaults
    out: dict[str, ModelProfile] = {}
    for name, fallback in defaults.items():
        out[name] = profile_from_dict(name, raw.get(name), fallback)
    for name, value in raw.items():
        if name not in out:
            fallback = ModelProfile(str(name), "custom", api_base, model)
            out[str(name)] = profile_from_dict(str(name), value, fallback)
    return out


def profiles_to_settings(profiles: dict[str, ModelProfile]) -> dict[str, dict[str, Any]]:
    return {name: profile_to_dict(profile) for name, profile in profiles.items()}

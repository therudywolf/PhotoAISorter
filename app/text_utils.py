"""Shared text processing utilities for model output."""

from __future__ import annotations

import re


def strip_thinking_sections(text: str) -> str:
    """Remove known LLM reasoning wrappers before downstream parsing.

    Supported forms:
    - <think> ... </think>
    - <|channel>thought ... <channel|>
    - <|channel|> ... final ... (extract only final answer)
    """
    if not text:
        return ""
    cleaned = text
    final_match = re.search(r"(?is)<\|channel\|>\s*final\b(.*)$", cleaned)
    if final_match:
        cleaned = final_match.group(1)
    cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned)
    cleaned = re.sub(r"(?is)<think>.*$", " ", cleaned)
    cleaned = re.sub(r"(?is)<\|channel\>\s*thought\b.*?<channel\|>", " ", cleaned)
    cleaned = re.sub(r"(?is)<\|channel\|>\s*(?:analysis|thought)\b.*?<\|channel\|>\s*final\b", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*<think>\s*$", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*</think>\s*$", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

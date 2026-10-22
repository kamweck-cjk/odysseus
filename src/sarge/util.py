"""Small shared helpers for Sarge."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

CAMPAIGN_ID_RE = re.compile(r"^[a-f0-9]{16}$")
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9-]{1,128}$")
STEP_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip(value: Any, limit: int, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def as_string_list(value: Any, limit: int = 12, item_chars: int = 500) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = clip(item, item_chars)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def safe_unicode(value: Any, limit: int = 500) -> str:
    text = clip(value, limit)
    return text.encode("utf-8", "replace").decode("utf-8")


def select_intel(text: str, budget: int, focus_terms: Optional[list[str]] = None) -> str:
    """Select the most relevant slices of a long research report.

    Research reports start with an executive summary, so the opening is always
    retained. The rest of the budget goes to paragraphs that mention the current
    step's focus terms, then to the report tail (usually recommendations).
    """
    if not text or len(text) <= budget:
        return text or ""
    budget = max(1_000, int(budget))
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return clip(text, budget)
    terms = [str(term).lower() for term in (focus_terms or []) if str(term).strip()]
    head: list[str] = []
    used = 0
    for paragraph in paragraphs:
        cost = len(paragraph) + 2
        if used + cost > max(1_000, budget // 3):
            break
        head.append(paragraph)
        used += cost
    tail: list[str] = []
    tail_used = 0
    for paragraph in reversed(paragraphs):
        if paragraph in head:
            break
        cost = len(paragraph) + 2
        if tail_used + cost > max(1_000, budget // 4):
            break
        tail.append(paragraph)
        tail_used += cost
    picked: list[str] = []
    picked_used = used + tail_used
    for paragraph in paragraphs[len(head):]:
        if paragraph in tail:
            continue
        cost = len(paragraph) + 2
        if picked_used + cost > budget:
            continue
        haystack = paragraph.lower()
        if not terms or any(term in haystack for term in terms):
            picked.append(paragraph)
            picked_used += cost
    sections = head + picked
    if tail:
        sections.append("## Report tail (selected)")
        sections.extend(reversed(tail))
    result = "\n\n".join(sections)
    return result if len(result) <= budget else clip(result, budget)


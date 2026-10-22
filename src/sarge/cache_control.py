"""Best-effort provider-side KV/prefix cache eviction for Sarge steps.

Odysseus cannot clear a remote model's KV cache directly. For self-hosted
servers we use the documented admin endpoints where they exist:

* vLLM: ``POST /reset_prefix_cache``
* llama.cpp server: ``GET /slots`` then ``POST /slots/{id}?action=erase``

Ollama, LM Studio, and hosted APIs have no portable reset endpoint. Sarge still
isolates those calls with a fresh session id and a fresh message context.
Reset requests are deliberately short and failures never stop a campaign.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 2.0
_support_cache: dict[str, dict[str, bool]] = {}


def _origin(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return ""


def _candidate_roots(url: str) -> list[str]:
    from src.endpoint_resolver import normalize_base

    roots: list[str] = []
    base = normalize_base(url).rstrip("/")
    origin = _origin(url)
    for candidate in (base, origin):
        if candidate and candidate not in roots:
            roots.append(candidate)
    if base.endswith("/v1"):
        trimmed = base[:-3].rstrip("/")
        if trimmed and trimmed not in roots:
            roots.append(trimmed)
    return roots


async def _post(client: httpx.AsyncClient, url: str, headers: dict) -> bool:
    try:
        response = await client.post(url, headers=headers, timeout=_TIMEOUT)
        return response.status_code < 400
    except Exception:
        return False


async def clear_provider_cache(
    url: str,
    model: str = "",
    headers: Optional[dict] = None,
    *,
    enabled: bool = True,
) -> dict:
    """Ask a local OpenAI-compatible server to drop cached prompt state."""
    from src.model_context import is_local_endpoint

    if not enabled or not url or not is_local_endpoint(url):
        return {"attempted": False, "local": False, "methods": [], "errors": []}
    origin = _origin(url) or url
    support = _support_cache.setdefault(origin, {})
    methods: list[str] = []
    errors: list[str] = []
    auth_headers = {}
    if headers:
        for key in ("Authorization", "authorization", "x-api-key", "X-Api-Key"):
            if headers.get(key):
                auth_headers[key] = headers[key]

    roots = _candidate_roots(url)
    async with httpx.AsyncClient() as client:
        vllm_tasks = []
        for root in roots:
            if support.get("vllm", True):
                vllm_tasks.append(_post(client, f"{root}/reset_prefix_cache", auth_headers))
        if vllm_tasks:
            vllm_results = await asyncio.gather(*vllm_tasks, return_exceptions=True)
            if any(result is True for result in vllm_results):
                methods.append("vllm:reset_prefix_cache")
                support["vllm"] = True
            else:
                support["vllm"] = False

        if support.get("llamacpp", True):
            slots_root = ""
            slots = None
            for root in roots:
                try:
                    response = await client.get(f"{root}/slots", headers=auth_headers, timeout=_TIMEOUT)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            slots = data
                            slots_root = root
                            break
                except Exception:
                    continue
            if slots_root:
                support["llamacpp"] = True
                erased = 0
                for slot in slots or []:
                    if not isinstance(slot, dict):
                        continue
                    slot_id = slot.get("id")
                    if slot_id is None:
                        continue
                    try:
                        response = await client.post(
                            f"{slots_root}/slots/{slot_id}?action=erase",
                            headers=auth_headers,
                            timeout=_TIMEOUT,
                        )
                        if response.status_code < 400:
                            erased += 1
                    except Exception:
                        continue
                if erased:
                    methods.append(f"llamacpp:erase_slots({erased})")
            else:
                support["llamacpp"] = False

    return {
        "attempted": True,
        "local": True,
        "methods": methods,
        "errors": errors,
    }

"""Best-effort provider-side KV/prefix cache eviction for Sarge steps.

Odysseus cannot clear a remote model's KV cache directly. For self-hosted
servers we use the documented admin endpoints where they exist:

* vLLM: ``POST /reset_prefix_cache``
* llama.cpp server: ``GET /slots`` then ``POST /slots/{id}?action=erase``

Ollama, LM Studio, and hosted APIs have no portable reset endpoint. That is the
common case (LM Studio is the default local runtime for many operators), so
Sarge never relies on a reset succeeding. Instead the guaranteed isolation is
per-request: every step gets a brand-new ``session_id``, a fresh message
context, and ``cache_prompt: false`` on the OpenAI-compatible payload, which
LM Studio and other llama.cpp-derived servers honour.

Successful probes are remembered and failed probes are negative-cached per
origin, so an LM Studio server costs at most one short probing round on the
first step and zero requests afterwards. Reset requests are deliberately short
and failures never stop a campaign.
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
# Origins we have positively identified as LM Studio. LM Studio's server is
# llama.cpp-derived but does NOT expose /slots or /reset_prefix_cache, so once
# identified we skip probing entirely and return immediately.
_lmstudio_origins: set[str] = set()


def _origin(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return ""


async def _probe_provider(client: httpx.AsyncClient, root: str) -> Optional[str]:
    """Identify the local server software using the same fingerprints as
    ``src/model_discovery.py`` so Sarge and the model scanner agree.

    * LM Studio: ``GET /api/v1/models`` with ``models[0]`` carrying ``key`` and
      ``architecture``. LM Studio's OpenAI-compatible ``/v1`` surface looks
      identical to vLLM/llama.cpp, so the native endpoint is required.
    * llama.cpp (llama-server): ``GET /props`` describing slots/chat template.
    """
    try:
        response = await client.get(f"{root}/api/v1/models", timeout=_TIMEOUT)
        if response.status_code == 200:
            models = (response.json() or {}).get("models")
            if (
                isinstance(models, list)
                and models
                and isinstance(models[0], dict)
                and "key" in models[0]
                and "architecture" in models[0]
            ):
                return "lmstudio"
    except Exception:
        pass
    try:
        response = await client.get(f"{root}/props", timeout=_TIMEOUT)
        if response.status_code == 200:
            props = response.json() or {}
            if isinstance(props, dict) and (
                "default_generation_settings" in props
                or "total_slots" in props
                or "chat_template" in props
            ):
                return "llamacpp"
    except Exception:
        pass
    return None


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
        return {
            "attempted": False,
            "local": False,
            "provider": "remote",
            "methods": [],
            "errors": [],
            "isolation": "remote endpoint; stateless per request",
        }
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
    if origin in _lmstudio_origins:
        return {
            "attempted": True,
            "local": True,
            "provider": "lmstudio",
            "methods": [],
            "errors": [],
            "isolation": "fresh session + cache_prompt=false (LM Studio has no reset endpoint)",
        }

    async with httpx.AsyncClient() as client:
        if not support.get("fingerprinted"):
            support["fingerprinted"] = True
            identified = ""
            for root in roots:
                found = await _probe_provider(client, root)
                if found:
                    identified = found
                    break
            if identified:
                support[identified] = True
                if identified == "lmstudio":
                    _lmstudio_origins.add(origin)
            if identified == "lmstudio":
                return {
                    "attempted": True,
                    "local": True,
                    "provider": "lmstudio",
                    "methods": [],
                    "errors": [],
                    "isolation": "fresh session + cache_prompt=false (LM Studio has no reset endpoint)",
                }
            if identified != "llamacpp":
                # Not llama.cpp: don't waste a /slots round on every origin.
                support["llamacpp"] = False

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

    provider = "unknown-local"
    if methods:
        provider = "vllm" if any(m.startswith("vllm") for m in methods) else "llamacpp"
    elif support.get("llamacpp") or support.get("vllm"):
        provider = "llamacpp" if support.get("llamacpp") else "vllm"
    else:
        provider = "generic-local"
    isolation = (
        "provider reset: " + ", ".join(methods)
        if methods
        else "fresh session + cache_prompt=false (no provider reset endpoint)"
    )
    return {
        "attempted": True,
        "local": True,
        "provider": provider,
        "methods": methods,
        "errors": errors,
        "isolation": isolation,
    }

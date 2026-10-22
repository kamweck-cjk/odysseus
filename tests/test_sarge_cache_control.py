"""Sarge cache control must never require a provider reset endpoint.

The common local runtime is LM Studio, which exposes an OpenAI-compatible
``/v1`` surface but **no** vLLM ``/reset_prefix_cache`` and **no** llama.cpp
``/slots``. Sarge's isolation guarantee is therefore the per-request one
(fresh session id + fresh context + ``cache_prompt: false``); the provider
probes are strictly best-effort and must be skipped once the server has been
identified.

These tests stand up a fake httpx client so no real server is needed. They pin:

* LM Studio is fingerprinted from ``/api/v1/models`` (same signal as
  ``src/model_discovery.py``), and after that no reset request is attempted.
* A generic local server that answers neither fingerprint gets at most one
  probing round, then zero requests on later steps.
* Remote endpoints are never probed at all.
"""

from __future__ import annotations

import pytest

from src.sarge import cache_control


class _FakeResponse:
    def __init__(self, status_code=404, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Route fake responses by URL prefix and record every request."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        return self._match(url)

    async def post(self, url, headers=None, timeout=None, json=None):
        self.calls.append(("POST", url))
        return self._match(url)

    def _match(self, url):
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response
        return _FakeResponse(404)


@pytest.fixture(autouse=True)
def _reset_caches():
    cache_control._support_cache.clear()
    cache_control._lmstudio_origins.clear()
    yield
    cache_control._support_cache.clear()
    cache_control._lmstudio_origins.clear()


def _install_client(monkeypatch, routes):
    holder = {}

    def factory(*args, **kwargs):
        client = _FakeAsyncClient(routes)
        holder["client"] = client
        return client

    monkeypatch.setattr(cache_control.httpx, "AsyncClient", factory)
    monkeypatch.setattr("src.model_context.is_local_endpoint", lambda url: True)
    return holder


LMSTUDIO_ROUTES = {
    "http://127.0.0.1:1234/api/v1/models": _FakeResponse(
        200,
        {"models": [{"key": "qwen2.5-7b-instruct", "architecture": "qwen2"}]},
    ),
}


@pytest.mark.asyncio
async def test_lmstudio_identified_and_never_probes_reset(monkeypatch):
    holder = _install_client(monkeypatch, LMSTUDIO_ROUTES)

    first = await cache_control.clear_provider_cache("http://127.0.0.1:1234/v1")
    assert first["provider"] == "lmstudio"
    assert first["methods"] == []
    assert first["local"] is True
    requested = [url for _, url in holder["client"].calls]
    assert any(url.endswith("/api/v1/models") for url in requested)
    # LM Studio does not expose either reset endpoint, so neither must be
    # probed once the fingerprint answers.
    assert not any("reset_prefix_cache" in url for url in requested)
    assert not any(url.endswith("/slots") for url in requested)

    # Second step: identified origin is a pure no-op with zero HTTP traffic.
    holder["client"].calls.clear()
    second = await cache_control.clear_provider_cache("http://127.0.0.1:1234/v1")
    assert second["provider"] == "lmstudio"
    assert holder["client"].calls == []


@pytest.mark.asyncio
async def test_generic_local_probes_once_then_stops(monkeypatch):
    # No fingerprint matches, no reset endpoint answers.
    holder = _install_client(monkeypatch, {})

    first = await cache_control.clear_provider_cache("http://127.0.0.1:9999/v1")
    assert first["provider"] == "generic-local"
    assert first["methods"] == []
    assert first["isolation"].startswith("fresh session")

    holder["client"].calls.clear()
    second = await cache_control.clear_provider_cache("http://127.0.0.1:9999/v1")
    assert second["provider"] == "generic-local"
    # Negative-cached probes: no further HTTP requests on later steps.
    assert holder["client"].calls == []


@pytest.mark.asyncio
async def test_llamacpp_props_identifies_and_uses_slots(monkeypatch):
    routes = {
        "http://127.0.0.1:8080/props": _FakeResponse(
            200,
            {"default_generation_settings": {"n_ctx": 4096}, "total_slots": 1},
        ),
        "http://127.0.0.1:8080/slots": _FakeResponse(200, [{"id": 0}]),
        "http://127.0.0.1:8080/slots/0": _FakeResponse(200, {}),
    }
    holder = _install_client(monkeypatch, routes)

    result = await cache_control.clear_provider_cache("http://127.0.0.1:8080/v1")
    assert result["provider"] == "llamacpp"
    assert any(m.startswith("llamacpp:erase_slots") for m in result["methods"])
    requested = [url for _, url in holder["client"].calls]
    assert any(url.endswith("/slots") for url in requested)


@pytest.mark.asyncio
async def test_remote_endpoint_is_never_probed(monkeypatch):
    holder = {}

    def factory(*args, **kwargs):
        client = _FakeAsyncClient({})
        holder["client"] = client
        return client

    monkeypatch.setattr(cache_control.httpx, "AsyncClient", factory)
    monkeypatch.setattr("src.model_context.is_local_endpoint", lambda url: False)

    result = await cache_control.clear_provider_cache("https://api.example.com/v1")
    assert result["local"] is False
    assert result["methods"] == []
    assert "isolation" in result
    assert holder == {}  # client never even constructed

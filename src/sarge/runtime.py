"""Resolve the model endpoint a Sarge campaign runs on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.endpoint_resolver import (
    _endpoint_enabled_models,
    _first_chat_model,
    build_chat_url,
    build_headers,
    resolve_endpoint,
    resolve_endpoint_runtime,
)
from src.model_context import is_local_endpoint


class SargeRuntimeError(Exception):
    pass


@dataclass
class RuntimeConfig:
    url: str
    model: str
    headers: dict
    endpoint_id: str
    endpoint_name: str
    local: bool
    context_length: int = 0


def _endpoint_config(owner: str, endpoint_id: str, model: str) -> Optional[RuntimeConfig]:
    from core.database import ModelEndpoint, SessionLocal
    from src.auth_helpers import owner_filter

    db = SessionLocal()
    try:
        query = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,  # noqa: E712
        )
        query = owner_filter(query, ModelEndpoint, owner)
        endpoint = query.first()
        if endpoint is None:
            return None
        base, api_key = resolve_endpoint_runtime(endpoint, owner=owner or None)
        resolved_model = model or _first_chat_model(_endpoint_enabled_models(endpoint)) or ""
        if not resolved_model:
            raise SargeRuntimeError(
                "The selected endpoint has no chat model. Refresh its model list in Settings."
            )
        url = build_chat_url(base)
        headers = build_headers(api_key, base)
        return RuntimeConfig(
            url=url,
            model=resolved_model,
            headers=headers,
            endpoint_id=endpoint.id,
            endpoint_name=endpoint.name or "",
            local=is_local_endpoint(url),
        )
    finally:
        db.close()


def resolve_runtime(owner: str, campaign: dict) -> RuntimeConfig:
    endpoint_id = str(campaign.get("endpoint_id") or "").strip()
    model = str(campaign.get("model") or "").strip()
    if endpoint_id:
        config = _endpoint_config(owner, endpoint_id, model)
        if config is None:
            raise SargeRuntimeError(
                "The selected endpoint is unavailable. Choose a model again in the campaign edit screen."
            )
    else:
        url, resolved_model, headers = resolve_endpoint("default", owner=owner or None)
        if not url or not resolved_model:
            url, resolved_model, headers = resolve_endpoint("utility", owner=owner or None)
        if not url or not resolved_model:
            config = _first_owned_endpoint(owner)
            if config is None:
                raise SargeRuntimeError(
                    "No chat model is configured. Add an endpoint and model in Settings first."
                )
        else:
            config = RuntimeConfig(
                url=url,
                model=resolved_model,
                headers=dict(headers or {}),
                endpoint_id="",
                endpoint_name="Default chat model",
                local=is_local_endpoint(url),
            )
    if str(campaign.get("privacy") or "local") == "local" and not config.local:
        raise SargeRuntimeError(
            "This campaign is local-only, but the selected model is not on a local/private endpoint. "
            "Choose a local endpoint or switch the campaign to full-web mode."
        )
    return config


def _first_owned_endpoint(owner: str) -> Optional[RuntimeConfig]:
    from core.database import ModelEndpoint, SessionLocal
    from src.auth_helpers import owner_filter

    db = SessionLocal()
    try:
        query = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)  # noqa: E712
        query = owner_filter(query, ModelEndpoint, owner)
        for endpoint in query.order_by(ModelEndpoint.id.asc()).all():
            model = _first_chat_model(_endpoint_enabled_models(endpoint))
            if not model:
                continue
            base, api_key = resolve_endpoint_runtime(endpoint, owner=owner or None)
            url = build_chat_url(base)
            return RuntimeConfig(
                url=url,
                model=model,
                headers=build_headers(api_key, base),
                endpoint_id=endpoint.id,
                endpoint_name=endpoint.name or "",
                local=is_local_endpoint(url),
            )
    finally:
        db.close()
    return None

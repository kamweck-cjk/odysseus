"""Sarge campaign API - /api/sarge/*."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.auth_helpers import require_user
from src.sarge.planner import SargePlanError
from src.sarge.runtime import SargeRuntimeError
from src.sarge.service import SargeBusyError, SargeService
from src.sarge.store import SargeStoreError
from src.sarge.util import clip, utcnow

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {
    "draft",
    "awaiting_approval",
    "paused",
    "blocked",
    "completed",
    "failed",
    "cancelled",
}


class SargeCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=8_000)
    title: Optional[str] = Field(default=None, max_length=160)
    brief: Optional[str] = Field(default=None, max_length=16_000)
    research_session_id: Optional[str] = Field(default=None, max_length=128)
    source_text: Optional[str] = Field(default=None, max_length=500_000)
    endpoint_id: Optional[str] = Field(default=None, max_length=128)
    endpoint_name: Optional[str] = Field(default=None, max_length=300)
    model: Optional[str] = Field(default=None, max_length=400)
    privacy: Literal["local", "web"] = "local"
    autonomy: Literal["review", "unattended"] = "review"
    verify_steps: bool = True
    max_rounds: int = Field(default=10, ge=2, le=30)
    max_tool_calls: int = Field(default=30, ge=4, le=120)


class SargeUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=160)
    objective: Optional[str] = Field(default=None, max_length=8_000)
    brief: Optional[str] = Field(default=None, max_length=16_000)
    endpoint_id: Optional[str] = Field(default=None, max_length=128)
    endpoint_name: Optional[str] = Field(default=None, max_length=300)
    model: Optional[str] = Field(default=None, max_length=400)
    privacy: Optional[Literal["local", "web"]] = None
    autonomy: Optional[Literal["review", "unattended"]] = None
    verify_steps: Optional[bool] = None
    max_rounds: Optional[int] = Field(default=None, ge=2, le=30)
    max_tool_calls: Optional[int] = Field(default=None, ge=4, le=120)


class SargeApproveRequest(BaseModel):
    plan: Optional[dict] = None


def _check(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, "Campaign not found")
    except SargeBusyError as exc:
        raise HTTPException(409, str(exc))
    except (SargePlanError, SargeRuntimeError, SargeStoreError, ValueError) as exc:
        raise HTTPException(400, str(exc))


def _require_campaign(service: SargeService, owner: str, campaign_id: str) -> dict:
    campaign = service.store.load(owner, campaign_id)
    if campaign is None:
        raise HTTPException(404, "Campaign not found")
    return campaign


def setup_sarge_routes(service: Optional[SargeService] = None) -> APIRouter:
    router = APIRouter(tags=["sarge"])

    def _service() -> SargeService:
        if service is not None:
            return service
        from src.sarge.service import get_service

        return get_service()

    @router.get("/api/sarge/health")
    async def sarge_health(owner: str = Depends(require_user)):
        svc = _service()
        return {
            "ok": True,
            "campaigns": len(svc.store.list(owner, limit=500)),
            "workspace_root": str(svc.store.workspace_root),
        }

    @router.get("/api/sarge/research-sources")
    async def sarge_research_sources(
        owner: str = Depends(require_user),
        limit: int = Query(default=100, ge=1, le=300),
    ):
        from src.constants import DEEP_RESEARCH_DIR

        root = Path(DEEP_RESEARCH_DIR).resolve()
        items = []
        if root.exists():
            for entry in root.iterdir():
                if not entry.is_file() or entry.suffix != ".json":
                    continue
                try:
                    data = json.loads(entry.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                if str(data.get("owner") or "") != owner:
                    continue
                if not (data.get("raw_report") or data.get("result")):
                    continue
                items.append(
                    {
                        "session_id": entry.stem,
                        "query": clip(data.get("query") or entry.stem, 300),
                        "created_at": data.get("created_at") or data.get("updated_at"),
                        "archived": bool(data.get("archived")),
                        "size": entry.stat().st_size,
                    }
                )
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"items": items[:limit], "total": len(items)}

    @router.get("/api/sarge/campaigns")
    async def sarge_list(owner: str = Depends(require_user)):
        return {"items": _service().store.list(owner)}

    @router.post("/api/sarge/campaigns", status_code=201)
    async def sarge_create(body: SargeCreateRequest, owner: str = Depends(require_user)):
        svc = _service()
        intel_parts: list[str] = []
        source: dict = {}
        if body.research_session_id:
            try:
                research_text, source = svc.store.import_research(owner, body.research_session_id)
                intel_parts.append(research_text)
            except SargeStoreError as exc:
                raise HTTPException(404 if "not found" in str(exc).lower() else 400, str(exc))
        if body.source_text and body.source_text.strip():
            intel_parts.append(
                "# Operator-supplied intel\n\n" + body.source_text.strip()
            )
            source = {**source, "kind": source.get("kind") or "operator_text"}
        intel = "\n\n---\n\n".join(intel_parts)
        campaign = _check(
            svc.store.create,
            owner,
            title=body.title or "",
            objective=body.objective.strip(),
            brief=body.brief or "",
            source=source,
            privacy=body.privacy,
            autonomy=body.autonomy,
            endpoint_id=body.endpoint_id,
            endpoint_name=body.endpoint_name,
            model=body.model,
            verify_steps=body.verify_steps,
            max_rounds=body.max_rounds,
            max_tool_calls=body.max_tool_calls,
            intel_text=intel,
        )
        return svc.store.detail(campaign)

    @router.get("/api/sarge/campaigns/{campaign_id}")
    async def sarge_detail(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.patch("/api/sarge/campaigns/{campaign_id}")
    async def sarge_update(
        campaign_id: str,
        body: SargeUpdateRequest,
        owner: str = Depends(require_user),
    ):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        if campaign.get("status") in {"planning", "running"}:
            raise HTTPException(409, "Pause the campaign before editing it.")
        updates = body.model_dump(exclude_unset=True)
        if not updates:
            return svc.store.detail(campaign)

        def mutate(item: dict) -> None:
            item.update(updates)
            if body.objective is not None:
                item["objective"] = body.objective.strip()
            if body.title is not None:
                item["title"] = (body.title or "").strip() or clip(item.get("objective"), 80)

        _check(svc.store.mutate, owner, campaign_id, mutate)
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.delete("/api/sarge/campaigns/{campaign_id}")
    async def sarge_delete(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        if campaign.get("status") in {"planning", "running"}:
            raise HTTPException(409, "Cancel the campaign before deleting it.")
        ok = _check(svc.store.delete, owner, campaign_id)
        if not ok:
            raise HTTPException(404, "Campaign not found")
        return {"ok": True}

    @router.post("/api/sarge/campaigns/{campaign_id}/plan")
    async def sarge_plan(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        campaign = _check(svc.start_plan, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/approve")
    async def sarge_approve(
        campaign_id: str,
        body: SargeApproveRequest,
        owner: str = Depends(require_user),
    ):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        if campaign.get("status") != "awaiting_approval":
            raise HTTPException(409, "This campaign is not awaiting plan approval.")
        campaign = _check(svc.approve_and_start, owner, campaign_id, body.plan)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/start")
    async def sarge_start(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        if campaign.get("status") in {"running", "planning"}:
            raise HTTPException(409, "Campaign is already active")
        campaign = _check(svc.approve_and_start, owner, campaign_id, None)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/pause")
    async def sarge_pause(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        campaign = _check(svc.pause, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/resume")
    async def sarge_resume(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        campaign = _check(svc.resume, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/cancel")
    async def sarge_cancel(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        campaign = _check(svc.cancel, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.post("/api/sarge/campaigns/{campaign_id}/retry")
    async def sarge_retry(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        campaign = _check(svc.retry, owner, campaign_id)
        return svc.store.detail(campaign)

    @router.get("/api/sarge/campaigns/{campaign_id}/artifacts")
    async def sarge_artifacts(campaign_id: str, owner: str = Depends(require_user)):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        return {"items": svc.store.list_artifacts(campaign)}

    @router.get("/api/sarge/campaigns/{campaign_id}/artifacts/{artifact_path:path}")
    async def sarge_artifact(
        campaign_id: str,
        artifact_path: str,
        owner: str = Depends(require_user),
        download: bool = Query(default=False),
    ):
        svc = _service()
        campaign = _check(_require_campaign, svc, owner, campaign_id)
        try:
            path = svc.store.artifact_path(campaign, artifact_path)
        except SargeStoreError as exc:
            raise HTTPException(404, str(exc))
        return FileResponse(
            str(path),
            filename=path.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    @router.get("/api/sarge/campaigns/{campaign_id}/events")
    async def sarge_events(
        campaign_id: str,
        request: Request,
        owner: str = Depends(require_user),
        after: int = Query(default=0, ge=0),
        follow: bool = Query(default=False),
    ):
        svc = _service()
        _check(_require_campaign, svc, owner, campaign_id)
        if not follow:
            return {"events": svc.store.read_events(campaign_id, after=after, limit=1_000)}

        async def stream():
            last_seq = int(after)
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    break
                events = svc.store.read_events(campaign_id, after=last_seq, limit=500)
                for event in events:
                    last_seq = int(event.get("seq") or last_seq)
                    yield f"id: {last_seq}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if events:
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                    if idle_ticks >= 20:
                        yield ": keep-alive\n\n"
                        idle_ticks = 0
                current = svc.store.load(owner, campaign_id)
                if current is None:
                    yield "event: end\ndata: {}\n\n"
                    break
                if (
                    str(current.get("status") or "") in _TERMINAL_STATUSES
                    and not events
                ):
                    yield "event: end\ndata: {}\n\n"
                    break
                await asyncio.sleep(0.9)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router

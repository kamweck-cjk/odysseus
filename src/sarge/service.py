"""Background campaign orchestration for Sarge."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .planner import SargePlanError, _normalize_plan, plan_campaign
from .runner import SargeCancelled, SargeRunner
from .runtime import SargeRuntimeError, resolve_runtime
from .store import SargeStore
from .util import clip, utcnow

logger = logging.getLogger(__name__)


class SargeBusyError(Exception):
    pass


class SargeService:
    def __init__(self, store: Optional[SargeStore] = None):
        self.store = store or SargeStore()
        self.runner = SargeRunner(self.store)
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def recover(self) -> int:
        """Mark campaigns interrupted by a previous process as blocked."""
        recovered = 0
        for campaign in self.store.iter_all():
            status = str(campaign.get("status") or "")
            if status not in {"planning", "running"}:
                continue
            owner = str(campaign.get("owner") or "")
            campaign_id = str(campaign.get("id") or "")
            try:
                self.store.mutate(
                    owner,
                    campaign_id,
                    lambda item: item.update(
                        {
                            "status": "blocked",
                            "status_detail": "Execution was interrupted by an Odysseus restart.",
                            "error": "Interrupted by application restart. Review the field log and resume.",
                            "pause_requested": True,
                        }
                    ),
                )
                self.store.append_event(
                    owner,
                    campaign_id,
                    "recovered",
                    "Execution was interrupted by an Odysseus restart; campaign marked blocked.",
                    {"previous_status": status},
                )
                recovered += 1
            except Exception as exc:
                logger.warning("Could not recover Sarge campaign %s: %s", campaign_id, exc)
        return recovered

    async def shutdown(self) -> None:
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._stopping = False

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def start_plan(self, owner: str, campaign_id: str) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") in {"planning", "running"}:
            raise SargeBusyError("Campaign is already active")
        # Resolve before changing status so a configuration error is returned to
        # the caller instead of silently turning into a failed campaign.
        resolve_runtime(owner, campaign)
        self._update(
            owner,
            campaign_id,
            status="planning",
            status_detail="Planning campaign…",
            error="",
            pause_requested=False,
            cancel_requested=False,
            finished_at=None,
        )
        self.store.append_event(
            owner,
            campaign_id,
            "planning",
            "Campaign planner started.",
            {"model": campaign.get("model")},
        )
        self._start(
            owner,
            campaign_id,
            self._plan_task(owner, campaign_id),
        )
        return self.store.load(owner, campaign_id) or campaign

    async def _plan_task(self, owner: str, campaign_id: str) -> None:
        try:
            campaign = self.store.load(owner, campaign_id)
            if campaign is None:
                return
            runtime = resolve_runtime(owner, campaign)
            intel = self.store.read_intel(campaign)
            plan = await plan_campaign(campaign, intel, runtime)
            self._update(
                owner,
                campaign_id,
                plan=plan,
                status="awaiting_approval",
                status_detail="Plan ready for review.",
                stats={
                    **(campaign.get("stats") or {}),
                    "steps_total": len(plan.get("steps") or []),
                    "steps_completed": 0,
                    "steps_failed": 0,
                },
                started_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "plan_ready",
                f"Plan ready: {len(plan.get('steps') or [])} steps.",
                {
                    "title": plan.get("title"),
                    "steps": [
                        {"id": step.get("id"), "title": step.get("title")}
                        for step in (plan.get("steps") or [])
                    ],
                },
            )
        except asyncio.CancelledError:
            if not self._stopping:
                self._update(
                    owner,
                    campaign_id,
                    status="draft",
                    status_detail="Planning cancelled.",
                    finished_at=utcnow(),
                )
            raise
        except Exception as exc:
            logger.exception("Sarge planning failed for %s", campaign_id)
            self._update(
                owner,
                campaign_id,
                status="failed",
                status_detail="Planning failed.",
                error=clip(f"{type(exc).__name__}: {exc}", 4_000),
                finished_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "failed",
                "Planning failed.",
                {"error": clip(str(exc), 2_000)},
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def approve_and_start(
        self,
        owner: str,
        campaign_id: str,
        edited_plan: Optional[dict] = None,
    ) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") in {"running", "planning"}:
            raise SargeBusyError("Campaign is already active")
        plan = edited_plan if isinstance(edited_plan, dict) else campaign.get("plan")
        if not isinstance(plan, dict):
            raise SargePlanError("There is no plan to approve.")
        if edited_plan is not None:
            plan = _normalize_plan(json.dumps(edited_plan, ensure_ascii=False), campaign)
        resolve_runtime(owner, campaign)
        plan["approved"] = True
        plan["approved_at"] = utcnow()
        self._update(
            owner,
            campaign_id,
            plan=plan,
            status="running",
            status_detail="Campaign is running.",
            approved_at=plan["approved_at"],
            started_at=campaign.get("started_at") or utcnow(),
            finished_at=None,
            pause_requested=False,
            cancel_requested=False,
            current_step_id=None,
            error="",
            stats={
                **(campaign.get("stats") or {}),
                "steps_total": len(plan.get("steps") or []),
            },
        )
        self.store.append_event(
            owner,
            campaign_id,
            "approved",
            "Plan approved. Campaign execution started.",
            {"steps": len(plan.get("steps") or [])},
        )
        self._start(owner, campaign_id, self._run_task(owner, campaign_id))
        return self.store.load(owner, campaign_id) or campaign

    def resume(self, owner: str, campaign_id: str) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") in {"running", "planning"}:
            raise SargeBusyError("Campaign is already active")
        plan = campaign.get("plan")
        if not isinstance(plan, dict) or not plan.get("steps"):
            raise SargePlanError("Approve a plan before resuming.")
        resolve_runtime(owner, campaign)
        self._update(
            owner,
            campaign_id,
            status="running",
            status_detail="Campaign resumed.",
            pause_requested=False,
            cancel_requested=False,
            finished_at=None,
            error="",
        )
        self.store.append_event(owner, campaign_id, "resumed", "Campaign resumed.")
        self._start(owner, campaign_id, self._run_task(owner, campaign_id))
        return self.store.load(owner, campaign_id) or campaign

    def pause(self, owner: str, campaign_id: str) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") != "running":
            return campaign
        self._update(
            owner,
            campaign_id,
            status="paused",
            status_detail="Pause requested. Sarge stops after the current tool call.",
            pause_requested=True,
        )
        self.store.append_event(owner, campaign_id, "paused", "Pause requested.")
        return self.store.load(owner, campaign_id) or campaign

    def cancel(self, owner: str, campaign_id: str) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") in {"completed", "cancelled"}:
            return campaign
        self._update(
            owner,
            campaign_id,
            status="cancelled",
            status_detail="Campaign cancelled.",
            cancel_requested=True,
            pause_requested=False,
            finished_at=utcnow(),
        )
        self.store.append_event(owner, campaign_id, "cancelled", "Campaign cancelled by the owner.")
        task = self._tasks.get((owner, campaign_id))
        if task and not task.done():
            task.cancel()
        return self.store.load(owner, campaign_id) or campaign

    def retry(self, owner: str, campaign_id: str) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        if campaign.get("status") in {"running", "planning"}:
            raise SargeBusyError("Campaign is already active")
        resolve_runtime(owner, campaign)
        self._update(
            owner,
            campaign_id,
            status="running",
            status_detail="Retrying incomplete steps.",
            pause_requested=False,
            cancel_requested=False,
            error="",
            finished_at=None,
        )
        self.store.append_event(owner, campaign_id, "retry", "Retrying incomplete steps.")
        self._start(owner, campaign_id, self._run_task(owner, campaign_id))
        return self.store.load(owner, campaign_id) or campaign

    async def _run_task(self, owner: str, campaign_id: str) -> None:
        try:
            campaign = self.store.load(owner, campaign_id)
            if campaign is None:
                return
            runtime = resolve_runtime(owner, campaign)
            await self.runner.run(owner, campaign_id, runtime)
            final = self.store.load(owner, campaign_id)
            if final and final.get("status") == "running":
                self._update(
                    owner,
                    campaign_id,
                    status="blocked",
                    status_detail="Execution stopped without a terminal state.",
                    error="Runner stopped unexpectedly. Resume to continue.",
                    finished_at=utcnow(),
                )
                self.store.append_event(
                    owner,
                    campaign_id,
                    "blocked",
                    "Execution stopped without a terminal state.",
                )
        except SargeCancelled:
            self._update(
                owner,
                campaign_id,
                status="cancelled",
                status_detail="Campaign cancelled.",
                finished_at=utcnow(),
            )
        except asyncio.CancelledError:
            if not self._stopping:
                latest = self.store.load(owner, campaign_id)
                if latest and not latest.get("cancel_requested"):
                    self._update(
                        owner,
                        campaign_id,
                        status="paused",
                        status_detail="Execution paused.",
                        pause_requested=True,
                    )
                    self.store.append_event(owner, campaign_id, "paused", "Execution paused.")
            else:
                self._update(
                    owner,
                    campaign_id,
                    status="paused",
                    status_detail="Interrupted by application shutdown.",
                    pause_requested=True,
                )
                self.store.append_event(
                    owner,
                    campaign_id,
                    "paused",
                    "Interrupted by application shutdown; resume after restart.",
                )
            raise
        except Exception as exc:
            logger.exception("Sarge campaign failed for %s", campaign_id)
            self._update(
                owner,
                campaign_id,
                status="failed",
                status_detail="Campaign execution failed.",
                error=clip(f"{type(exc).__name__}: {exc}", 4_000),
                finished_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "failed",
                "Campaign execution failed.",
                {"error": clip(str(exc), 2_000)},
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _start(self, owner: str, campaign_id: str, coroutine) -> None:
        key = (owner or "", campaign_id)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            raise SargeBusyError("Campaign is already active")
        task = asyncio.create_task(coroutine)
        self._tasks[key] = task

        def _done(_task: asyncio.Task) -> None:
            if self._tasks.get(key) is _task:
                self._tasks.pop(key, None)

        task.add_done_callback(_done)

    def _update(self, owner: str, campaign_id: str, **fields) -> dict:
        return self.store.mutate(owner, campaign_id, lambda campaign: campaign.update(fields))


_service: Optional[SargeService] = None


def get_service() -> SargeService:
    global _service
    if _service is None:
        _service = SargeService()
    return _service

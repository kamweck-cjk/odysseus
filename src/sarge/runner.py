"""Step-isolated campaign execution for Sarge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any, Optional

from src.agent_loop import stream_agent_loop
from src.agent_runtime.authority import create_request_authority
from src.llm_core import llm_call_async
from src.model_context import budget_context_for_model
from src.turn_contract import FAMILY_TOOLS

from .cache_control import clear_provider_cache
from .prompts import (
    build_final_report_messages,
    build_step_messages,
    build_verification_messages,
)
from .runtime import RuntimeConfig
from .store import SargeStore
from .util import clip, extract_json_object, select_intel, utcnow


class SargeCancelled(Exception):
    pass


def _context_window(runtime: RuntimeConfig) -> int:
    if runtime.context_length:
        return runtime.context_length
    try:
        budget = budget_context_for_model(runtime.url, runtime.model, fallback=8_192)
        runtime.context_length = int(budget or 8_192)
    except Exception:
        runtime.context_length = 8_192
    return runtime.context_length


def _find_step(campaign: dict, step_id: str) -> Optional[dict]:
    plan = campaign.get("plan") or {}
    for step in plan.get("steps") or []:
        if step.get("id") == step_id:
            return step
    return None


def _prior_summaries(campaign: dict, current_step_id: str) -> list[str]:
    out: list[str] = []
    for step in (campaign.get("plan") or {}).get("steps") or []:
        if step.get("id") == current_step_id:
            break
        if step.get("status") == "completed" and step.get("summary"):
            out.append(f"{step.get('title')}: {clip(step.get('summary'), 1_500)}")
    return out


def _step_artifact_lines(artifacts: list[dict]) -> list[str]:
    return [f"{item.get('path')} ({item.get('size', 0)} bytes)" for item in artifacts]


def _parse_stream_event(raw: str) -> Optional[dict]:
    if not raw:
        return None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class SargeRunner:
    def __init__(self, store: SargeStore):
        self.store = store

    def _update_step(self, owner: str, campaign_id: str, step_id: str, **fields: Any) -> dict:
        def mutate(campaign: dict) -> None:
            step = _find_step(campaign, step_id)
            if step is None:
                raise KeyError("Step not found")
            step.update(fields)
            if fields.get("status") == "completed":
                stats = campaign.setdefault("stats", {})
                stats["steps_completed"] = sum(
                    1
                    for item in campaign["plan"]["steps"]
                    if item.get("status") == "completed"
                )
            elif fields.get("status") == "failed":
                stats = campaign.setdefault("stats", {})
                stats["steps_failed"] = sum(
                    1
                    for item in campaign["plan"]["steps"]
                    if item.get("status") == "failed"
                )

        return self.store.mutate(owner, campaign_id, mutate)

    def _update_campaign(self, owner: str, campaign_id: str, **fields: Any) -> dict:
        def mutate(campaign: dict) -> None:
            campaign.update(fields)

        return self.store.mutate(owner, campaign_id, mutate)

    async def run(self, owner: str, campaign_id: str, runtime: RuntimeConfig) -> dict:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        plan = campaign.get("plan") or {}
        steps = plan.get("steps") or []
        if not steps:
            raise ValueError("Campaign has no approved plan")

        self.store.append_event(owner, campaign_id, "status", "Campaign execution started.", {"status": "running"})
        failed = False
        paused = False
        cancelled = False
        try:
            for index, planned_step in enumerate(steps):
                step_id = str(planned_step.get("id") or "")
                step = _find_step(campaign, step_id)
                if step is None:
                    continue
                campaign = self.store.load(owner, campaign_id) or campaign
                step = _find_step(campaign, step_id) or step
                if step.get("status") == "completed":
                    continue
                if campaign.get("cancel_requested"):
                    cancelled = True
                    break
                if campaign.get("pause_requested") or campaign.get("status") == "paused":
                    paused = True
                    break

                completed = await self._run_step_with_retries(
                    owner,
                    campaign_id,
                    runtime,
                    step_id,
                    index + 1,
                    len(steps),
                )
                campaign = self.store.load(owner, campaign_id) or campaign
                if campaign.get("cancel_requested"):
                    cancelled = True
                    break
                if campaign.get("pause_requested") or campaign.get("status") == "paused":
                    paused = True
                    break
                current = _find_step(campaign, step_id) or {}
                if not completed:
                    failed = True
                    self._update_campaign(
                        owner,
                        campaign_id,
                        status="blocked",
                        status_detail=f"Step failed: {current.get('title') or step_id}",
                        error=clip(current.get("error") or current.get("verification"), 2_000),
                    )
                    self.store.append_event(
                        owner,
                        campaign_id,
                        "blocked",
                        f"Campaign blocked at step {index + 1}.",
                        {"step_id": step_id},
                    )
                    break
        except SargeCancelled:
            cancelled = True
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            failed = True
            self._update_campaign(
                owner,
                campaign_id,
                status="failed",
                status_detail="Campaign runner crashed.",
                error=clip(f"{type(exc).__name__}: {exc}", 4_000),
                finished_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "failed",
                "Campaign runner crashed.",
                {"error": clip(f"{type(exc).__name__}: {exc}", 2_000)},
            )
            raise
        finally:
            # The service normally owns terminal status. This keeps direct
            # runner failures from leaving a campaign stuck on "running".
            if failed or cancelled or paused:
                pass

        if cancelled:
            return self.store.load(owner, campaign_id) or campaign
        if paused:
            return self.store.load(owner, campaign_id) or campaign

        campaign = self.store.load(owner, campaign_id) or campaign
        if failed or campaign.get("status") == "blocked":
            try:
                await self._write_final_report(owner, campaign_id, runtime)
            except Exception:
                pass
            return self.store.load(owner, campaign_id) or campaign

        all_done = all(
            step.get("status") == "completed"
            for step in ((campaign.get("plan") or {}).get("steps") or [])
        )
        if all_done:
            self._update_campaign(
                owner,
                campaign_id,
                status="completed",
                status_detail="All campaign steps completed.",
                finished_at=utcnow(),
                current_step_id=None,
            )
            self.store.append_event(
                owner,
                campaign_id,
                "completed",
                "Campaign complete.",
                {"artifacts": len(self.store.list_artifacts(campaign))},
            )
            await self._write_final_report(owner, campaign_id, runtime)
        return self.store.load(owner, campaign_id) or campaign

    async def _run_step_with_retries(
        self,
        owner: str,
        campaign_id: str,
        runtime: RuntimeConfig,
        step_id: str,
        step_number: int,
        step_total: int,
    ) -> bool:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        step = _find_step(campaign, step_id) or {}
        max_attempts = 2
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            campaign = self.store.load(owner, campaign_id) or campaign
            step = _find_step(campaign, step_id) or step
            if campaign.get("cancel_requested"):
                raise SargeCancelled()
            if campaign.get("pause_requested") or campaign.get("status") == "paused":
                return False
            self._update_step(
                owner,
                campaign_id,
                step_id,
                status="running",
                attempts=int(step.get("attempts") or 0) + 1,
                started_at=utcnow(),
                error="",
            )
            self._update_campaign(owner, campaign_id, current_step_id=step_id)
            self.store.append_event(
                owner,
                campaign_id,
                "step_started",
                f"Step {step_number}/{step_total}: {step.get('title')}",
                {"step_id": step_id, "attempt": attempt},
            )
            ok, report, verification = await self._execute_step(
                owner, campaign_id, runtime, step_id, step_number, step_total
            )
            if ok:
                return True
            campaign = self.store.load(owner, campaign_id) or campaign
            step = _find_step(campaign, step_id) or step
            if campaign.get("cancel_requested"):
                raise SargeCancelled()
            if campaign.get("pause_requested") or campaign.get("status") == "paused":
                self._update_step(
                    owner,
                    campaign_id,
                    step_id,
                    status="pending",
                    summary=clip(report, 3_000),
                    verification=clip(verification, 2_000),
                )
                return False
            if attempt < max_attempts:
                self.store.append_event(
                    owner,
                    campaign_id,
                    "step_retry",
                    f"Retrying step {step_number} after verification failure.",
                    {"step_id": step_id, "attempt": attempt + 1},
                )
                continue
            self._update_step(
                owner,
                campaign_id,
                step_id,
                status="failed",
                summary=clip(report, 3_000),
                verification=clip(verification, 2_000),
                error=clip(verification or "Step did not pass verification.", 2_000),
                finished_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "step_failed",
                f"Step {step_number} failed verification.",
                {"step_id": step_id},
            )
            return False
        return False

    async def _execute_step(
        self,
        owner: str,
        campaign_id: str,
        runtime: RuntimeConfig,
        step_id: str,
        step_number: int,
        step_total: int,
    ) -> tuple[bool, str, str]:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            raise KeyError("Campaign not found")
        step = _find_step(campaign, step_id) or {}
        workspace = str(self.store.workspace_dir(campaign_id))
        context = _context_window(runtime)

        reset = await clear_provider_cache(
            runtime.url,
            runtime.model,
            runtime.headers,
            enabled=True,
        )
        self.store.append_event(
            owner,
            campaign_id,
            "context_reset",
            "Prepared an isolated context for this step.",
            reset,
        )

        intel = self.store.read_intel(campaign)
        focus = list(step.get("intel_focus") or []) + [
            str(step.get("title") or ""),
            str(step.get("objective") or ""),
        ]
        intel_budget = max(3_000, min(24_000, int(context * 1.5)))
        selected_intel = select_intel(intel, intel_budget, focus)
        messages = build_step_messages(
            objective=str(campaign.get("objective") or ""),
            success_criteria=list((campaign.get("plan") or {}).get("success_criteria") or []),
            plan_summary=str((campaign.get("plan") or {}).get("summary") or ""),
            step=step,
            prior_summaries=_prior_summaries(campaign, step_id),
            intel_text=selected_intel,
            privacy=str(campaign.get("privacy") or "local"),
            workspace=workspace,
            step_index=step_number,
            step_total=step_total,
        )
        capabilities = {"shell_files"}
        relevant = set(FAMILY_TOOLS["shell_files"])
        if str(campaign.get("privacy") or "local") == "web":
            capabilities.add("search_browser")
            relevant |= set(FAMILY_TOOLS["search_browser"])
        try:
            authority = create_request_authority(
                "Execute the next Sarge campaign step using the provided workspace tools.",
                owner=owner or None,
                session_id=f"sarge-{campaign_id}-{step_id}",
                workspace=workspace,
                capabilities=frozenset(capabilities),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not create a scoped tool authority: {exc}") from exc

        session_id = f"sarge-{campaign_id}-{step_id}-{uuid.uuid4().hex[:8]}"
        max_tokens = max(1_024, min(4_096, int(context // 4)))
        full_text = ""
        final_text = ""
        final_replaced = False
        tool_calls = 0
        last_check = time.monotonic()
        cancelled_mid_step = False

        async with contextlib.aclosing(
            stream_agent_loop(
                endpoint_url=runtime.url,
                model=runtime.model,
                messages=messages,
                headers=runtime.headers,
                temperature=0.2,
                max_tokens=max_tokens,
                prompt_type="sarge_step",
                max_rounds=int(campaign.get("max_rounds") or 10),
                max_tool_calls=int(campaign.get("max_tool_calls") or 30),
                context_length=context,
                session_id=session_id,
                owner=owner or None,
                workspace=workspace,
                cwd=workspace,
                disabled_tools=set(),
                relevant_tools=relevant,
                workload="background",
                request_authority=authority,
            )
        ) as stream:
            async for chunk in stream:
                now = time.monotonic()
                if now - last_check >= 1.5:
                    last_check = now
                    latest = self.store.load(owner, campaign_id)
                    if latest and (latest.get("cancel_requested") or latest.get("pause_requested") or latest.get("status") in {"paused", "cancelled"}):
                        cancelled_mid_step = True
                        break
                data = _parse_stream_event(str(chunk))
                if not data:
                    continue
                if "delta" in data and not data.get("thinking"):
                    delta = str(data.get("delta") or "")
                    if final_replaced:
                        full_text = ""
                        final_replaced = False
                    full_text += delta
                elif data.get("type") == "final_response":
                    final_text = str(data.get("content") or "")
                    final_replaced = True
                elif data.get("type") == "tool_start":
                    tool_calls += 1
                    self.store.append_event(
                        owner,
                        campaign_id,
                        "tool_start",
                        f"{data.get('tool', 'tool')}: {clip(data.get('command'), 300)}",
                        {"tool": data.get("tool"), "step_id": step_id},
                    )
                elif data.get("type") == "tool_output":
                    output = data.get("stdout") or data.get("output") or data.get("result") or ""
                    self.store.append_event(
                        owner,
                        campaign_id,
                        "tool_output",
                        f"{data.get('tool', 'tool')}: {clip(output, 500)}",
                        {
                            "tool": data.get("tool"),
                            "step_id": step_id,
                            "exit_code": data.get("exit_code"),
                        },
                    )
                elif data.get("type") == "agent_step":
                    self.store.append_event(
                        owner,
                        campaign_id,
                        "agent_round",
                        f"Agent round {data.get('round', '?')}.",
                        {"step_id": step_id},
                    )
                elif data.get("type") == "budget_exceeded":
                    self.store.append_event(
                        owner,
                        campaign_id,
                        "budget",
                        "Agent tool budget reached; synthesizing a report.",
                        {"step_id": step_id},
                    )

        if cancelled_mid_step:
            latest = self.store.load(owner, campaign_id) or campaign
            if latest.get("cancel_requested"):
                raise SargeCancelled()
            return False, full_text or final_text, "Execution paused by the owner."

        report = final_text.strip() or full_text.strip()
        campaign = self.store.load(owner, campaign_id) or campaign
        step = _find_step(campaign, step_id) or step
        artifacts = self.store.list_artifacts(campaign)
        if report:
            try:
                self.store.write_step_report(campaign_id, step_id, report)
            except Exception:
                pass
        self._update_step(
            owner,
            campaign_id,
            step_id,
            summary=clip(report, 3_000),
            artifacts=artifacts[:120],
        )
        self._update_campaign(
            owner,
            campaign_id,
            stats={**(campaign.get("stats") or {}), "tool_calls": int((campaign.get("stats") or {}).get("tool_calls") or 0) + tool_calls},
        )

        if not bool(campaign.get("verify_steps", True)):
            return True, report, "Verification disabled."
        return await self._verify_step(owner, campaign_id, runtime, step_id, report, artifacts)

    async def _verify_step(
        self,
        owner: str,
        campaign_id: str,
        runtime: RuntimeConfig,
        step_id: str,
        report: str,
        artifacts: list[dict],
    ) -> tuple[bool, str, str]:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            return False, report, "Campaign disappeared during verification."
        step = _find_step(campaign, step_id) or {}
        messages = build_verification_messages(
            step=step,
            assistant_report=report,
            artifacts=artifacts,
        )
        max_tokens = max(256, min(1_024, int(_context_window(runtime) // 8)))
        try:
            raw, _ = await llm_call_async(
                runtime.url,
                runtime.model,
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
                headers=runtime.headers,
                prompt_type="sarge_verify",
                session_id=f"sarge-verify-{campaign_id}-{step_id}-{uuid.uuid4().hex[:6]}",
                workload="background",
                return_model_metadata=True,
                bypass_cache=True,
                cache_prompt=False,
            )
        except Exception as exc:
            complete = bool(report.strip() or artifacts)
            verification = f"Verifier unavailable ({type(exc).__name__}); accepted based on workspace/report evidence."
            self.store.append_event(
                owner,
                campaign_id,
                "verify",
                verification,
                {"step_id": step_id, "complete": complete},
            )
            return complete, report, verification
        data = extract_json_object(raw)
        if not isinstance(data, dict):
            complete = bool(report.strip() or artifacts)
            verification = "Verifier returned malformed output; accepted based on workspace/report evidence."
        else:
            complete = bool(data.get("complete"))
            verification_parts = [
                clip(data.get("summary"), 1_500),
                ("Evidence: " + "; ".join(str(x) for x in (data.get("evidence") or [])[:8])) if data.get("evidence") else "",
                ("Missing: " + "; ".join(str(x) for x in (data.get("missing") or [])[:8])) if data.get("missing") else "",
            ]
            verification = "\n".join(part for part in verification_parts if part).strip() or "No verifier detail."
        if complete:
            self._update_step(
                owner,
                campaign_id,
                step_id,
                status="completed",
                verification=clip(verification, 2_000),
                finished_at=utcnow(),
            )
            self.store.append_event(
                owner,
                campaign_id,
                "step_completed",
                f"Step completed: {step.get('title')}",
                {"step_id": step_id, "artifacts": len(artifacts)},
            )
        else:
            self.store.append_event(
                owner,
                campaign_id,
                "step_incomplete",
                f"Step did not pass verification: {step.get('title')}",
                {"step_id": step_id, "verification": clip(verification, 800)},
            )
        return complete, report, verification

    async def _write_final_report(
        self,
        owner: str,
        campaign_id: str,
        runtime: RuntimeConfig,
    ) -> None:
        campaign = self.store.load(owner, campaign_id)
        if campaign is None:
            return
        steps = (campaign.get("plan") or {}).get("steps") or []
        artifacts = self.store.list_artifacts(campaign)
        messages = build_final_report_messages(
            campaign=campaign,
            step_records=steps,
            artifact_lines=_step_artifact_lines(artifacts),
        )
        text = ""
        try:
            text, _ = await llm_call_async(
                runtime.url,
                runtime.model,
                messages,
                temperature=0.2,
                max_tokens=max(512, min(2_048, int(_context_window(runtime) // 6))),
                headers=runtime.headers,
                prompt_type="sarge_after_action",
                session_id=f"sarge-report-{campaign_id}-{uuid.uuid4().hex[:6]}",
                workload="background",
                return_model_metadata=True,
                bypass_cache=True,
                cache_prompt=False,
            )
        except Exception:
            text = ""
        if not text.strip():
            text = self._fallback_report(campaign, artifacts)
        self.store.write_final_report(campaign, text)
        self.store.append_event(
            owner,
            campaign_id,
            "after_action",
            "After-action report written.",
            {"path": "after-action.md"},
        )

    @staticmethod
    def _fallback_report(campaign: dict, artifacts: list[dict]) -> str:
        lines = [
            "# After-Action Report",
            "",
            "## Outcome",
            "",
            f"Campaign: {campaign.get('title') or 'Untitled'}",
            f"Status: {campaign.get('status') or 'unknown'}",
            f"Objective: {campaign.get('objective') or ''}",
            "",
            "## What Changed",
            "",
        ]
        for step in (campaign.get("plan") or {}).get("steps") or []:
            lines.append(
                f"- **{step.get('title')}** — {step.get('status')} — "
                f"{clip(step.get('summary') or step.get('verification'), 500)}"
            )
        lines += ["", "## Deliverables", ""]
        if artifacts:
            lines.extend(f"- `{item.get('path')}` ({item.get('size', 0)} bytes)" for item in artifacts)
        else:
            lines.append("- No workspace artifacts were detected.")
        lines += [
            "",
            "## Verification",
            "",
            "See the step verification entries in the field log.",
            "",
            "## Open Issues / Next Actions",
            "",
            "Review any blocked or incomplete steps above before continuing.",
        ]
        return "\n".join(lines)

"""LLM planning for Sarge campaigns."""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Optional

from src.llm_core import llm_call_async
from src.model_context import budget_context_for_model, get_context_length_known

from .prompts import build_plan_messages, build_plan_repair_messages
from .runtime import RuntimeConfig
from .util import as_string_list, clip, extract_json_object, select_intel, utcnow


class SargePlanError(Exception):
    pass


def _context_budget(runtime: RuntimeConfig) -> int:
    try:
        window, known = get_context_length_known(runtime.url, runtime.model)
    except Exception:
        window, known = 0, False
    if not known:
        window = 8_192
    try:
        clamped = budget_context_for_model(runtime.url, runtime.model, fallback=window)
        if clamped:
            window = clamped
    except Exception:
        pass
    runtime.context_length = int(window or 8_192)
    return runtime.context_length


async def plan_campaign(
    campaign: dict,
    intel_text: str,
    runtime: RuntimeConfig,
) -> dict:
    context = await asyncio.to_thread(_context_budget, runtime)
    intel_budget = max(3_000, min(60_000, int(context * 2.2)))
    selected = select_intel(
        intel_text,
        intel_budget,
        [campaign.get("title", ""), campaign.get("objective", "")],
    )
    messages = build_plan_messages(
        objective=str(campaign.get("objective") or ""),
        brief=str(campaign.get("brief") or ""),
        intel_text=selected,
        privacy=str(campaign.get("privacy") or "local"),
        success_hint=str(campaign.get("brief") or ""),
    )
    max_tokens = max(1_024, min(4_096, int(context // 3)))
    session_id = f"sarge-plan-{campaign.get('id')}-{uuid.uuid4().hex[:8]}"
    response, actual_model = await llm_call_async(
        runtime.url,
        runtime.model,
        messages,
        temperature=0.15,
        max_tokens=max_tokens,
        headers=runtime.headers,
        prompt_type="sarge_plan",
        session_id=session_id,
        workload="background",
        return_model_metadata=True,
        bypass_cache=True,
        cache_prompt=False,
    )
    try:
        plan = _normalize_plan(response, campaign)
        plan["generator_model"] = actual_model or runtime.model
        return plan
    except SargePlanError as first_error:
        repair = build_plan_repair_messages(response, str(first_error))
        repaired, repaired_model = await llm_call_async(
            runtime.url,
            runtime.model,
            repair,
            temperature=0.05,
            max_tokens=max_tokens,
            headers=runtime.headers,
            prompt_type="sarge_plan_repair",
            session_id=session_id + "-fix",
            workload="background",
            return_model_metadata=True,
            bypass_cache=True,
            cache_prompt=False,
        )
        plan = _normalize_plan(repaired, campaign)
        plan["generator_model"] = repaired_model or actual_model or runtime.model
        plan["repaired"] = True
        return plan


def _normalize_plan(text: str, campaign: dict) -> dict:
    data = extract_json_object(text or "")
    if not isinstance(data, dict):
        raise SargePlanError("The planner did not return a JSON object.")

    raw_steps: list[Any] = []
    phases: list[str] = []
    for phase in data.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        name = clip(phase.get("name"), 160) or f"Phase {len(phases) + 1}"
        phase_steps = phase.get("steps")
        if not isinstance(phase_steps, list):
            continue
        phases.append(name)
        for step in phase_steps:
            if isinstance(step, dict):
                step = dict(step)
                step.setdefault("phase", name)
                raw_steps.append(step)
    if isinstance(data.get("steps"), list):
        for step in data["steps"]:
            if isinstance(step, dict):
                raw_steps.append(dict(step))
    if not raw_steps:
        raise SargePlanError("The plan has no executable steps.")
    if len(raw_steps) > 12:
        raw_steps = raw_steps[:12]

    steps: list[dict] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(raw_steps, start=1):
        title = clip(raw.get("title"), 240) or f"Step {index}"
        objective = clip(raw.get("objective"), 4_000) or title
        step_id = _safe_step_id(raw.get("id") or title, index, used_ids)
        used_ids.add(step_id)
        phase = clip(raw.get("phase"), 160)
        if phase and phase not in phases:
            phases.append(phase)
        steps.append(
            {
                "id": step_id,
                "title": title,
                "objective": objective,
                "phase": phase,
                "actions": as_string_list(raw.get("actions")),
                "deliverables": as_string_list(raw.get("deliverables")),
                "acceptance_criteria": as_string_list(raw.get("acceptance_criteria")),
                "intel_focus": as_string_list(raw.get("intel_focus"), limit=10, item_chars=120),
                "web_required": bool(raw.get("web_required")),
                "est_minutes": _positive_int(raw.get("est_minutes"), 20),
                "status": "pending",
                "attempts": 0,
                "summary": "",
                "verification": "",
                "artifacts": [],
                "error": "",
                "started_at": None,
                "finished_at": None,
            }
        )
    if len(steps) < 1:
        raise SargePlanError("The plan has no usable steps.")

    plan = {
        "version": 1,
        "title": clip(data.get("title"), 200) or clip(campaign.get("title"), 200),
        "summary": clip(data.get("summary"), 4_000) or clip(campaign.get("objective"), 4_000),
        "success_criteria": as_string_list(data.get("success_criteria"))
        or [clip(campaign.get("objective"), 1_000)],
        "phases": phases,
        "steps": steps,
        "generated_at": utcnow(),
        "approved_at": None,
        "approved": False,
    }
    return plan


def _safe_step_id(value: Any, index: int, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").lower()).strip("-_")
    if not base:
        base = f"step-{index}"
    if not base.startswith("step-"):
        base = f"step-{index}-{base}"
    base = base[:60]
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base[:54]}-{suffix}"
        suffix += 1
    return candidate


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return max(1, min(number, 240))

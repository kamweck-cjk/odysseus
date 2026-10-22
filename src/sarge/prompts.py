"""Prompts used by Sarge's planner, operator, and verifier."""

from __future__ import annotations

from .util import clip

PLANNER_SYSTEM = """\
You are Sarge, a campaign planner inside Odysseus. Turn the supplied research
brief and objective into a concrete, executable project plan.

Rules:
- The plan must produce real artifacts in a dedicated workspace, not a reading
  list or a generic course outline.
- Use 3 to 10 ordered steps. Each step must have one coherent objective that a
  capable shell/file agent can complete in roughly 5-45 minutes.
- Prefer local commands and files. Never assume network access. If the campaign
  explicitly allows the web, set web_required only for steps that genuinely need
  it; otherwise set it false.
- Do not invent facts that are not supported by the intel or objective. When
  information is missing, add a short discovery step that inspects the workspace
  or asks the user to supply a file.
- Every step must produce at least one concrete deliverable, preferably a file
  path relative to the campaign workspace. Analysis-only steps may name an
  artifact such as `notes/analysis.md`.
- Make later steps depend only on artifacts or knowledge produced by earlier
  steps.
- Include measurable acceptance criteria for every step.

Return ONLY a JSON object with this shape:
{
  "title": "short campaign title",
  "summary": "2-4 sentences describing the finished outcome",
  "success_criteria": ["..."],
  "phases": [
    {
      "name": "phase name",
      "steps": [
        {
          "id": "step-1",
          "title": "short imperative title",
          "objective": "one concrete objective",
          "actions": ["specific action"],
          "deliverables": ["relative/path.md", "description of artifact"],
          "acceptance_criteria": ["observable criterion"],
          "intel_focus": ["keyword or theme"],
          "web_required": false,
          "est_minutes": 20
        }
      ]
    }
  ]
}
"""


def build_plan_messages(
    *,
    objective: str,
    brief: str,
    intel_text: str,
    privacy: str,
    success_hint: str = "",
) -> list[dict]:
    network = (
        "This campaign may use web search/fetch."
        if privacy == "web"
        else "This campaign is local-only. Web tools are disabled."
    )
    user = f"""\
Campaign objective:
{clip(objective, 8_000)}

Operator brief:
{clip(brief, 12_000) or "(none)"}

Privacy / network:
{network}

Additional success criteria:
{clip(success_hint, 4_000) or "(none)"}

Research brief / source material:
<<<INTEL
{intel_text}
INTEL>>>

Plan the campaign now. Keep the plan compact enough to execute, and make the
first step useful even if the research brief is incomplete. Return only JSON.
"""
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_plan_repair_messages(previous: str, error: str) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": (
                "Your previous response could not be accepted as a Sarge plan.\n"
                f"Validation error: {clip(error, 1_000)}\n\n"
                f"Previous response:\n{clip(previous, 12_000)}\n\n"
                "Return the corrected plan as one JSON object only."
            ),
        },
    ]


def build_step_messages(
    *,
    objective: str,
    success_criteria: list[str],
    plan_summary: str,
    step: dict,
    prior_summaries: list[str],
    intel_text: str,
    privacy: str,
    workspace: str,
    step_index: int,
    step_total: int,
) -> list[dict]:
    network = (
        "Web search/fetch tools are available if this step requires them."
        if privacy == "web"
        else "This campaign is local-only. Web search/fetch tools are disabled."
    )
    criteria = "\n".join(f"- {item}" for item in success_criteria[:12]) or "- Complete the objective."
    actions = "\n".join(f"- {item}" for item in (step.get("actions") or [])[:12]) or "- Use your judgment."
    deliverables = "\n".join(f"- {item}" for item in (step.get("deliverables") or [])[:12]) or "- Produce a useful artifact."
    acceptance = "\n".join(f"- {item}" for item in (step.get("acceptance_criteria") or [])[:12]) or "- Verify the deliverable exists and is correct."
    history = "\n".join(f"- {item}" for item in prior_summaries[-8:]) or "- This is the first step."
    system = f"""\
You are Sarge, an autonomous project operator. You are executing step
{step_index} of {step_total} in a campaign. Work in the campaign workspace.
You have shell, file, and workspace tools. Use them to create and verify the
deliverables. Do not ask the user questions during execution.

Rules:
- Complete this step only. Do not start future steps or rewrite completed work.
- Prefer inspecting existing files before creating anything.
- Keep outputs inside the campaign workspace.
- Run commands to verify your work. Use relative paths.
- Do not expose secrets, credentials, or unrelated user data.
- Finish with a short report of what you did, what you verified, and any gaps.
- {network}

Campaign workspace: {workspace}
Campaign summary: {clip(plan_summary, 3_000)}

Campaign success criteria:
{criteria}

Prior completed-step summaries:
{history}
"""
    user = f"""\
Current step: {clip(step.get("title"), 300)}
Objective: {clip(step.get("objective"), 4_000)}

Suggested actions:
{actions}

Required deliverables:
{deliverables}

Acceptance criteria:
{acceptance}

Relevant research intel:
<<<INTEL
{intel_text}
INTEL>>>

Execute this step now. Create the deliverables, verify them, and then report.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


VERIFIER_SYSTEM = """\
You are Sarge's completion verifier. Evaluate whether one campaign step was
actually completed. Be strict: a plan, a promise, or a partial script without
the required deliverable is not complete.

Return ONLY JSON:
{
  "complete": true,
  "confidence": 0.0,
  "summary": "one paragraph of evidence-based summary",
  "evidence": ["file or command evidence"],
  "missing": ["what is missing"],
  "next_hint": "short hint for a retry, or empty"
}
"""


def build_verification_messages(
    *,
    step: dict,
    assistant_report: str,
    artifacts: list[dict],
) -> list[dict]:
    artifact_lines = "\n".join(
        f"- {item.get('path')} ({item.get('size', 0)} bytes)" for item in artifacts[:120]
    ) or "- (no files detected in the campaign workspace)"
    user = f"""\
Step title: {clip(step.get("title"), 300)}
Objective: {clip(step.get("objective"), 3_000)}

Required deliverables:
{chr(10).join("- " + str(x) for x in (step.get("deliverables") or [])[:12]) or "- (none listed)"}

Acceptance criteria:
{chr(10).join("- " + str(x) for x in (step.get("acceptance_criteria") or [])[:12]) or "- (none listed)"}

Files currently in the workspace:
{artifact_lines}

Operator's final report:
<<<REPORT
{clip(assistant_report, 12_000)}
REPORT>>>

Judge whether the step is complete. Return only the JSON object.
"""
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": user},
    ]


FINAL_REPORT_SYSTEM = """\
You are writing the after-action report for a completed Sarge campaign. Use
only the supplied campaign record. Be direct and useful to the owner.

Return a Markdown report with these sections:
# After-Action Report
## Outcome
## What Changed
## Deliverables
## Verification
## Open Issues / Next Actions

Do not invent files or successes. Mark failed or incomplete items clearly.
"""


def build_final_report_messages(
    *,
    campaign: dict,
    step_records: list[dict],
    artifact_lines: list[str],
) -> list[dict]:
    steps = []
    for record in step_records:
        steps.append(
            f"### {record.get('title')}\n"
            f"Status: {record.get('status')}\n"
            f"Summary: {clip(record.get('summary'), 2_000)}\n"
            f"Verification: {clip(record.get('verification'), 1_000)}"
        )
    user = f"""\
Campaign: {clip(campaign.get("title"), 300)}
Objective: {clip(campaign.get("objective"), 4_000)}
Status: {campaign.get("status")}

Step record:
{chr(10).join(steps)}

Workspace deliverables:
{chr(10).join("- " + item for item in artifact_lines[:200]) or "- (none)"}

Write the report now.
"""
    return [
        {"role": "system", "content": FINAL_REPORT_SYSTEM},
        {"role": "user", "content": user},
    ]

"""Filesystem store for Sarge campaign metadata, events, and artifacts.

Campaign metadata lives under ``DATA_DIR/sarge``. Agent execution workspaces
live under ``AGENT_WORKSPACE_DIR/sarge`` because the agent's shell/file tools
are only allowed to read data roots below ``AGENT_WORKSPACE_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from core.atomic_io import atomic_write_json, atomic_write_text
from src.constants import AGENT_WORKSPACE_DIR, SARGE_DIR, SARGE_WORKSPACE_ROOT

from .util import CAMPAIGN_ID_RE, SESSION_ID_RE, clip, utcnow

_META_FILE = "campaign.json"
_EVENTS_FILE = "events.jsonl"
_INTEL_FILE = "intel.md"
_FINAL_FILE = "after-action.md"
_STEPS_DIR = "steps"

MAX_SOURCE_CHARS = 500_000
MAX_EVENT_MESSAGE = 2_000
MAX_EVENT_DATA = 8_000
MAX_ARTIFACTS = 400


class SargeStoreError(Exception):
    pass


class SargeStore:
    def __init__(self, root: Optional[str] = None, workspace_root: Optional[str] = None):
        self.root = Path(root or SARGE_DIR).resolve()
        self.workspace_root = Path(workspace_root or SARGE_WORKSPACE_ROOT).resolve()
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for path in (self.root, self.workspace_root):
            path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
        # A defensive check: campaign workspaces must remain inside the agent
        # data root. If a deployment misconfigures both paths, fail loudly.
        try:
            self.workspace_root.relative_to(Path(AGENT_WORKSPACE_DIR).resolve())
        except ValueError as exc:
            raise SargeStoreError(
                "SARGE_WORKSPACE_ROOT must be inside AGENT_WORKSPACE_DIR"
            ) from exc

    def _lock_for(self, campaign_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(campaign_id, threading.RLock())

    @staticmethod
    def validate_id(campaign_id: str) -> str:
        if not isinstance(campaign_id, str) or not CAMPAIGN_ID_RE.fullmatch(campaign_id):
            raise SargeStoreError("Invalid campaign id")
        return campaign_id

    def campaign_dir(self, campaign_id: str) -> Path:
        self.validate_id(campaign_id)
        return self.root / campaign_id

    def workspace_dir(self, campaign_id: str) -> Path:
        self.validate_id(campaign_id)
        safe = self.workspace_root / campaign_id
        safe.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(safe, 0o700)
        except OSError:
            pass
        return safe

    def _meta_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / _META_FILE

    def _events_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / _EVENTS_FILE

    def create(
        self,
        owner: str,
        *,
        title: str = "",
        objective: str,
        brief: str = "",
        source: Optional[dict] = None,
        privacy: str = "local",
        autonomy: str = "review",
        endpoint_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        model: Optional[str] = None,
        verify_steps: bool = True,
        max_rounds: int = 10,
        max_tool_calls: int = 30,
        intel_text: str = "",
    ) -> dict:
        campaign_id = uuid.uuid4().hex[:16]
        now = utcnow()
        campaign = {
            "id": campaign_id,
            "owner": owner or "",
            "title": clip(title, 160) or clip(objective, 80) or "Untitled campaign",
            "objective": clip(objective, 8_000),
            "brief": clip(brief, 16_000),
            "source": source or {},
            "status": "draft",
            "status_detail": "",
            "privacy": privacy if privacy in {"local", "web"} else "local",
            "autonomy": autonomy if autonomy in {"review", "unattended"} else "review",
            "endpoint_id": endpoint_id or "",
            "endpoint_name": endpoint_name or "",
            "model": model or "",
            "verify_steps": bool(verify_steps),
            "max_rounds": max(2, min(int(max_rounds or 10), 30)),
            "max_tool_calls": max(4, min(int(max_tool_calls or 30), 120)),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "plan": None,
            "approved_at": None,
            "pause_requested": False,
            "cancel_requested": False,
            "current_step_id": None,
            "event_seq": 0,
            "error": "",
            "stats": {
                "steps_total": 0,
                "steps_completed": 0,
                "steps_failed": 0,
                "tool_calls": 0,
            },
        }
        with self._lock_for(campaign_id):
            directory = self.campaign_dir(campaign_id)
            directory.mkdir(parents=True, exist_ok=False)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
            self.workspace_dir(campaign_id)
            if intel_text:
                atomic_write_text(str(directory / _INTEL_FILE), clip(intel_text, MAX_SOURCE_CHARS))
            atomic_write_json(str(self._meta_path(campaign_id)), campaign, indent=2)
            self._append_event_locked(
                campaign,
                "status",
                "Campaign drafted.",
                {"status": "draft"},
            )
        return self.load(owner, campaign_id, allow_empty_owner=True) or campaign

    def load(
        self,
        owner: str,
        campaign_id: str,
        *,
        allow_empty_owner: bool = False,
    ) -> Optional[dict]:
        try:
            self.validate_id(campaign_id)
        except SargeStoreError:
            return None
        path = self._meta_path(campaign_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        stored_owner = str(data.get("owner") or "")
        if stored_owner != (owner or ""):
            if not (allow_empty_owner and stored_owner == ""):
                return None
        return data

    def save(self, campaign: dict) -> dict:
        campaign_id = self.validate_id(str(campaign.get("id") or ""))
        campaign["updated_at"] = utcnow()
        with self._lock_for(campaign_id):
            atomic_write_json(str(self._meta_path(campaign_id)), campaign, indent=2)
        return campaign

    def mutate(
        self,
        owner: str,
        campaign_id: str,
        mutator: Callable[[dict], Any],
    ) -> dict:
        """Serialize a read/modify/write against one campaign."""
        self.validate_id(campaign_id)
        with self._lock_for(campaign_id):
            campaign = self.load(owner, campaign_id)
            if campaign is None:
                raise KeyError("Campaign not found")
            mutator(campaign)
            campaign["updated_at"] = utcnow()
            atomic_write_json(str(self._meta_path(campaign_id)), campaign, indent=2)
            return campaign

    def iter_all(self):
        """Yield raw campaign records. Used only for startup recovery."""
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or not CAMPAIGN_ID_RE.fullmatch(entry.name):
                continue
            try:
                data = json.loads((entry / _META_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                yield data

    def list(self, owner: str, limit: int = 200) -> list[dict]:
        items: list[dict] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return []
        for entry in entries:
            if not entry.is_dir() or not CAMPAIGN_ID_RE.fullmatch(entry.name):
                continue
            campaign = self.load(owner, entry.name)
            if campaign is None:
                continue
            items.append(self.summary(campaign))
        items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return items[: max(1, min(int(limit or 200), 500))]

    @staticmethod
    def summary(campaign: dict) -> dict:
        plan = campaign.get("plan") if isinstance(campaign.get("plan"), dict) else None
        stats = campaign.get("stats") if isinstance(campaign.get("stats"), dict) else {}
        steps = plan.get("steps") if plan and isinstance(plan.get("steps"), list) else []
        return {
            "id": campaign.get("id"),
            "title": campaign.get("title"),
            "objective": clip(campaign.get("objective"), 400),
            "status": campaign.get("status"),
            "status_detail": clip(campaign.get("status_detail"), 600),
            "model": campaign.get("model"),
            "endpoint_name": campaign.get("endpoint_name"),
            "privacy": campaign.get("privacy"),
            "autonomy": campaign.get("autonomy"),
            "created_at": campaign.get("created_at"),
            "updated_at": campaign.get("updated_at"),
            "started_at": campaign.get("started_at"),
            "finished_at": campaign.get("finished_at"),
            "current_step_id": campaign.get("current_step_id"),
            "steps_total": len(steps),
            "steps_completed": int(stats.get("steps_completed") or 0),
            "error": clip(campaign.get("error"), 800),
        }

    def detail(self, campaign: dict) -> dict:
        result = self.summary(campaign)
        result.update(
            {
                "brief": campaign.get("brief") or "",
                "source": campaign.get("source") or {},
                "plan": campaign.get("plan"),
                "endpoint_id": campaign.get("endpoint_id") or "",
                "verify_steps": bool(campaign.get("verify_steps")),
                "stats": campaign.get("stats") or {},
                "approved_at": campaign.get("approved_at"),
                "pause_requested": bool(campaign.get("pause_requested")),
                "cancel_requested": bool(campaign.get("cancel_requested")),
                "intel_excerpt": clip(self.read_intel(campaign), 4_000),
                "final_report": clip(self.read_final_report(campaign), 80_000),
                "events": self.read_events(str(campaign.get("id")), after=0, limit=300),
            }
        )
        return result

    def read_intel(self, campaign: dict) -> str:
        path = self.campaign_dir(str(campaign.get("id"))) / _INTEL_FILE
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write_intel(self, campaign: dict, text: str) -> None:
        self.validate_id(str(campaign.get("id")))
        atomic_write_text(
            str(self.campaign_dir(str(campaign.get("id"))) / _INTEL_FILE),
            clip(text, MAX_SOURCE_CHARS),
        )

    def read_final_report(self, campaign: dict) -> str:
        path = self.campaign_dir(str(campaign.get("id"))) / _FINAL_FILE
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write_final_report(self, campaign: dict, text: str) -> None:
        self.validate_id(str(campaign.get("id")))
        atomic_write_text(
            str(self.campaign_dir(str(campaign.get("id"))) / _FINAL_FILE),
            clip(text, 200_000),
        )

    def write_step_report(self, campaign_id: str, step_id: str, text: str) -> str:
        self.validate_id(campaign_id)
        if not re_safe_step_id(step_id):
            raise SargeStoreError("Invalid step id")
        path = self.campaign_dir(campaign_id) / _STEPS_DIR / f"{step_id}.md"
        atomic_write_text(str(path), clip(text, 120_000))
        return f"steps/{step_id}.md"

    def read_step_report(self, campaign_id: str, step_id: str) -> str:
        self.validate_id(campaign_id)
        if not re_safe_step_id(step_id):
            raise SargeStoreError("Invalid step id")
        path = self.campaign_dir(campaign_id) / _STEPS_DIR / f"{step_id}.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def append_event(
        self,
        owner: str,
        campaign_id: str,
        event_type: str,
        message: str = "",
        data: Optional[dict] = None,
    ) -> Optional[dict]:
        self.validate_id(campaign_id)
        with self._lock_for(campaign_id):
            campaign = self.load(owner, campaign_id)
            if campaign is None:
                return None
            event = self._append_event_locked(campaign, event_type, message, data)
            campaign["updated_at"] = utcnow()
            atomic_write_json(str(self._meta_path(campaign_id)), campaign, indent=2)
            return event

    def _append_event_locked(
        self,
        campaign: dict,
        event_type: str,
        message: str = "",
        data: Optional[dict] = None,
    ) -> dict:
        campaign_id = self.validate_id(str(campaign.get("id") or ""))
        seq = int(campaign.get("event_seq") or 0) + 1
        campaign["event_seq"] = seq
        event = {
            "seq": seq,
            "time": utcnow(),
            "type": clip(event_type, 64) or "log",
            "message": clip(message, MAX_EVENT_MESSAGE),
        }
        if data:
            try:
                encoded = json.dumps(data, ensure_ascii=False)
            except Exception:
                encoded = "{}"
            if len(encoded) > MAX_EVENT_DATA:
                encoded = json.dumps({"truncated": encoded[:MAX_EVENT_DATA]})
            try:
                event["data"] = json.loads(encoded)
            except Exception:
                event["data"] = {"value": encoded[:MAX_EVENT_DATA]}
        path = self._events_path(campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        return event

    def read_events(self, campaign_id: str, after: int = 0, limit: int = 500) -> list[dict]:
        try:
            self.validate_id(campaign_id)
        except SargeStoreError:
            return []
        path = self._events_path(campaign_id)
        events: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if int(event.get("seq") or 0) <= int(after or 0):
                        continue
                    events.append(event)
                    if len(events) >= max(1, min(int(limit or 500), 1_000)):
                        break
        except OSError:
            return []
        return events

    def list_artifacts(self, campaign: dict) -> list[dict]:
        campaign_id = self.validate_id(str(campaign.get("id") or ""))
        workspace = self.workspace_dir(campaign_id).resolve()
        items: list[dict] = []
        if not workspace.exists():
            return items
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [
                name
                for name in dirs
                if not name.startswith(".") and name not in {"__pycache__", "node_modules", ".git"}
            ]
            for filename in files:
                if filename.startswith("."):
                    continue
                path = Path(root) / filename
                try:
                    resolved = path.resolve()
                    resolved.relative_to(workspace)
                    stat = resolved.stat()
                except (OSError, ValueError):
                    continue
                if not resolved.is_file():
                    continue
                rel = resolved.relative_to(workspace).as_posix()
                items.append(
                    {
                        "path": rel,
                        "size": stat.st_size,
                        "modified": int(stat.st_mtime),
                    }
                )
                if len(items) >= MAX_ARTIFACTS:
                    return items
        items.sort(key=lambda item: item["path"])
        return items

    def artifact_path(self, campaign: dict, relative_path: str) -> Path:
        campaign_id = self.validate_id(str(campaign.get("id") or ""))
        workspace = self.workspace_dir(campaign_id).resolve()
        if not relative_path or "\x00" in relative_path:
            raise SargeStoreError("Invalid artifact path")
        candidate = (workspace / relative_path).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise SargeStoreError("Artifact path escapes the campaign workspace") from exc
        if not candidate.is_file():
            raise SargeStoreError("Artifact not found")
        return candidate

    def delete(self, owner: str, campaign_id: str) -> bool:
        campaign = self.load(owner, campaign_id)
        if campaign is None:
            return False
        workspace = self.workspace_dir(campaign_id).resolve()
        try:
            workspace.relative_to(self.workspace_root)
        except ValueError:
            return False
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(self.campaign_dir(campaign_id), ignore_errors=True)
        return True

    def import_research(self, owner: str, session_id: str) -> tuple[str, dict]:
        """Read an owned deep-research report and return (intel_text, source)."""
        if not SESSION_ID_RE.fullmatch(str(session_id or "")):
            raise SargeStoreError("Invalid research session id")
        from src.constants import DEEP_RESEARCH_DIR

        expected = f"{session_id}.json"
        root = Path(DEEP_RESEARCH_DIR).resolve()
        source_path: Optional[Path] = None
        for entry in root.iterdir() if root.exists() else []:
            if entry.is_file() and entry.name == expected:
                source_path = entry.resolve()
                break
        if source_path is None:
            raise SargeStoreError("Research report not found")
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SargeStoreError("Research report could not be read") from exc
        if not isinstance(data, dict):
            raise SargeStoreError("Research report is malformed")
        if str(data.get("owner") or "") != (owner or ""):
            raise SargeStoreError("Research report not found")

        query = str(data.get("query") or "Deep research report").strip()
        raw = data.get("raw_report")
        if not isinstance(raw, str) or not raw.strip():
            result = data.get("result")
            if isinstance(result, str):
                raw = result
            elif isinstance(result, dict):
                raw = result.get("report") or result.get("markdown") or json.dumps(result, ensure_ascii=False, indent=2)
            else:
                raw = ""
        findings = data.get("raw_findings")
        if findings and isinstance(findings, (list, dict)):
            try:
                raw = (raw or "") + "\n\n## Findings\n\n" + (
                    "\n".join(f"- {f}" for f in findings)
                    if isinstance(findings, list)
                    else json.dumps(findings, ensure_ascii=False, indent=2)
                )
            except Exception:
                pass
        if not raw or not raw.strip():
            raise SargeStoreError("Research report has no report text")
        text = (
            f"# Research brief: {query}\n\n"
            f"Source session: {session_id}\n\n"
            f"{raw.strip()}\n"
        )
        source = {
            "kind": "deep_research",
            "session_id": session_id,
            "query": query,
            "title": clip(query, 300),
            "imported_at": utcnow(),
        }
        return clip(text, MAX_SOURCE_CHARS), source


def re_safe_step_id(step_id: str) -> bool:
    from .util import STEP_ID_RE

    return bool(STEP_ID_RE.fullmatch(str(step_id or "")))

"""Low-resource hardware profile.

Odysseus runs perfectly well on a machine with no GPU, a 4 GB card, or 8 GB of
RAM — but the defaults it ships with assume a roomy machine:

* ``DEFAULT_MAX_TOKENS`` is 32768, so a local model is asked to reserve a
  32K-token output budget it will never use (and cannot afford on a small card).
* ``MAX_CONTEXT_MESSAGES`` is 90, which is far more history than a 4K-context
  local model can hold — the extra messages are pushed, then truncated, then
  re-tokenized on every single turn.
* The optional GPU features (TTS, speech-to-text, neural upscaling) load a
  CUDA context the moment they are imported, taking VRAM away from the model
  the user actually cares about.

This module detects what the machine really has, derives a tier from it, and
derives conservative limits from that tier.

Design notes
------------
* **A capable machine is untouched.** The ``standard`` tier means "no limits",
  so every clamp below is a no-op when the hardware is roomy. That keeps the
  feature strictly additive: upstream behaviour is preserved unless the
  machine is actually weak.
* **Only local endpoints are clamped.** A request to OpenAI or Anthropic is
  not running on the user's GPU, so shrinking its context/output budget would
  cost quality for no benefit. The clamps apply to ``localhost``/LAN model
  servers (llama.cpp, Ollama, vLLM, LM Studio) — the ones that actually have
  to fit in the box's VRAM/RAM.
* **Detection is slow, so it never blocks startup.** ``detect_system()``
  shells out to ``nvidia-smi``/PowerShell and takes a couple of seconds. It
  runs on a background thread; the profile reports itself as "not ready" until
  it lands, and every consumer treats "unknown" as "no clamp yet" rather than
  guessing.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Tier names, weakest first. "standard" is the no-limits tier.
TIER_MINIMAL = "minimal"
TIER_LOW = "low"
TIER_MEDIUM = "medium"
TIER_STANDARD = "standard"
TIERS = (TIER_MINIMAL, TIER_LOW, TIER_MEDIUM, TIER_STANDARD)

# Sentinel for "detect from hardware, don't use a stored override".
TIER_AUTO = "auto"

# Env var that overrides the tier outright — the escape hatch for users who
# know their machine better than the heuristic does (and for CI / containers
# where the detected host GPU is not the one in use).
TIER_ENV_VAR = "ODYSSEUS_LOW_RESOURCE_TIER"

# Machine-level (not per-user) settings file. The hardware belongs to the
# box, so every user of the same install shares the same profile.
_PROFILE_FILE_NAME = "low_resource_profile.json"

# Limits per tier. ``None`` means "do not clamp this".
#
# ``context_window_cap``  - max tokens of conversation the agent may keep for a
#                           LOCAL model. This is the single biggest VRAM lever:
#                           the KV cache grows linearly with context, and a
#                           32K context on a 4 GB card is an instant OOM.
# ``max_tokens_cap``      - max output tokens requested from a LOCAL model.
# ``context_messages_cap``- max messages of history kept for a LOCAL model.
# ``embedding_threads``   - ONNX Runtime thread pool for local embeddings.
#                           Each worker thread costs RAM; an unbounded pool on
#                           a small box starves everything else.
# ``allow_gpu_features``  - whether the optional in-process GPU features (TTS,
#                           STT, neural upscaling) may take VRAM. The user's own
#                           model server is never affected by this.
_TIER_LIMITS: Dict[str, Dict[str, Any]] = {
    TIER_MINIMAL: {
        "context_window_cap": 2048,
        "max_tokens_cap": 1024,
        "context_messages_cap": 12,
        "embedding_threads": 2,
        "allow_gpu_features": False,
    },
    TIER_LOW: {
        "context_window_cap": 4096,
        "max_tokens_cap": 2048,
        "context_messages_cap": 24,
        "embedding_threads": 4,
        "allow_gpu_features": True,
    },
    TIER_MEDIUM: {
        "context_window_cap": 8192,
        "max_tokens_cap": 4096,
        "context_messages_cap": 45,
        "embedding_threads": 8,
        "allow_gpu_features": True,
    },
    TIER_STANDARD: {
        "context_window_cap": None,
        "max_tokens_cap": None,
        "context_messages_cap": None,
        "embedding_threads": 0,  # 0 = leave ONNX Runtime's own default
        "allow_gpu_features": True,
    },
}

# VRAM thresholds (GB) used to pick a tier. Kept as explicit breakpoints
# rather than a formula so the behaviour is easy to reason about and to
# override.
_VRAM_BREAKPOINTS = (
    (3.0, TIER_MINIMAL),
    (6.0, TIER_LOW),
    (10.0, TIER_MEDIUM),
)
_MINIMAL_RAM_GB = 4.0
_LOW_RAM_GB = 8.0

_profile_lock = threading.Lock()
_profile: Optional[Dict[str, Any]] = None
_detect_started = False


def _profile_file() -> str:
    """Absolute path of the machine-level profile settings file."""
    try:
        from src.constants import DATA_DIR

        return os.path.join(DATA_DIR, _PROFILE_FILE_NAME)
    except Exception:  # pragma: no cover - constants should always import
        return os.path.join("data", _PROFILE_FILE_NAME)


def _read_stored_tier() -> Optional[str]:
    """Read the persisted tier override, if any. Returns None when unset."""
    path = _profile_file()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    tier = data.get("tier") if isinstance(data, dict) else None
    tier = (tier or "").strip().lower()
    return tier if tier in TIERS else None


def _write_stored_tier(tier: Optional[str]) -> None:
    """Persist (or clear) the tier override."""
    path = _profile_file()
    payload = {"tier": tier} if tier else {}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        from core.atomic_io import atomic_write_json

        atomic_write_json(path, payload, indent=2)
    except Exception as exc:
        logger.warning("Could not persist low-resource tier: %s", exc)


def _classify(system: Dict[str, Any]) -> str:
    """Pick a tier from a detected-hardware dict.

    ``system`` is the shape returned by ``services.hwfit.hardware.detect_system``.
    """
    has_gpu = bool(system.get("has_gpu"))
    try:
        vram_gb = float(system.get("gpu_vram_gb") or 0.0)
    except (TypeError, ValueError):
        vram_gb = 0.0
    try:
        ram_gb = float(system.get("total_ram_gb") or 0.0)
    except (TypeError, ValueError):
        ram_gb = 0.0

    # No usable GPU at all -> CPU-only box.
    if not has_gpu or vram_gb <= 0:
        tier = TIER_MINIMAL
    else:
        tier = TIER_STANDARD
        for threshold, candidate in _VRAM_BREAKPOINTS:
            if vram_gb < threshold:
                tier = candidate
                break

    # A tiny amount of system RAM makes every tier tighter: the model weights
    # themselves may not fit, so a generous context is pointless.
    if ram_gb and ram_gb < _MINIMAL_RAM_GB:
        tier = TIER_MINIMAL
    elif ram_gb and ram_gb < _LOW_RAM_GB and tier == TIER_STANDARD:
        tier = TIER_MEDIUM

    return tier


def _detect_hardware() -> Dict[str, Any]:
    """Run the hardware probe. Never raises — a failed probe means "unknown"."""
    try:
        from services.hwfit.hardware import detect_system

        return detect_system() or {}
    except Exception as exc:
        logger.info("Low-resource hardware probe failed: %s", exc)
        return {}


def _build_profile(system: Dict[str, Any], tier: str, source: str) -> Dict[str, Any]:
    limits = dict(_TIER_LIMITS.get(tier, _TIER_LIMITS[TIER_STANDARD]))
    return {
        "tier": tier,
        "source": source,  # "env" | "override" | "detected"
        "ready": True,
        "limits": limits,
        "hardware": {
            "has_gpu": bool(system.get("has_gpu")),
            "gpu_name": system.get("gpu_name"),
            "gpu_vram_gb": system.get("gpu_vram_gb"),
            "gpu_count": system.get("gpu_count"),
            "backend": system.get("backend"),
            "total_ram_gb": system.get("total_ram_gb"),
            "available_ram_gb": system.get("available_ram_gb"),
            "cpu_cores": system.get("cpu_cores"),
            "cpu_name": system.get("cpu_name"),
            "platform": system.get("platform"),
        },
    }


def _pending_profile() -> Dict[str, Any]:
    """The profile shape returned before detection has completed.

    Deliberately permissive: an unknown machine must not be clamped, because
    a wrong clamp is worse than no clamp.
    """
    return {
        "tier": TIER_STANDARD,
        "source": "pending",
        "ready": False,
        "limits": dict(_TIER_LIMITS[TIER_STANDARD]),
        "hardware": {},
    }


def get_profile() -> Dict[str, Any]:
    """Return the current profile, detecting it synchronously on first use."""
    global _profile
    with _profile_lock:
        if _profile is not None:
            return _profile

    # Detection takes a couple of seconds, so it happens outside the lock.
    system = _detect_hardware()
    profile = _resolve_profile(system)
    with _profile_lock:
        _profile = profile
    return profile


def _resolve_profile(system: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a hardware probe result into a profile, honouring any override."""
    env_tier = (os.getenv(TIER_ENV_VAR) or "").strip().lower()
    if env_tier in TIERS:
        return _build_profile(system, env_tier, "env")

    stored_tier = _read_stored_tier()
    if stored_tier:
        return _build_profile(system, stored_tier, "override")

    return _build_profile(system, _classify(system), "detected")


def refresh_profile() -> Dict[str, Any]:
    """Re-run detection and rebuild the profile, discarding the cache."""
    global _profile
    with _profile_lock:
        _profile = None
    return get_profile()


def warm_profile() -> None:
    """Kick off detection on a background thread so startup isn't delayed.

    Safe to call more than once — only the first call starts a probe. The
    thread also applies the profile's process-level defaults once it lands, so
    a lazily-initialised subsystem (the ONNX embedding runtime, for example)
    still picks them up.
    """
    global _detect_started
    with _profile_lock:
        if _detect_started or _profile is not None:
            return
        _detect_started = True

    def _worker():
        try:
            profile = get_profile()
            apply_env_defaults(profile)
            logger.info(
                "Low-resource profile: tier=%s (source=%s) gpu=%s vram=%sGB ram=%sGB",
                profile.get("tier"),
                profile.get("source"),
                (profile.get("hardware") or {}).get("gpu_name"),
                (profile.get("hardware") or {}).get("gpu_vram_gb"),
                (profile.get("hardware") or {}).get("total_ram_gb"),
            )
        except Exception as exc:
            logger.info("Low-resource warm-up failed: %s", exc)

    threading.Thread(target=_worker, name="low-resource-probe", daemon=True).start()


def profile_ready() -> bool:
    """True once the profile has been detected (or explicitly overridden)."""
    if (os.getenv(TIER_ENV_VAR) or "").strip().lower() in TIERS:
        return True
    with _profile_lock:
        return _profile is not None


def set_tier(tier: Optional[str]) -> Dict[str, Any]:
    """Persist a manual tier override. ``None``/``"auto"`` clears it."""
    normalized = (tier or "").strip().lower()
    if normalized in ("", TIER_AUTO):
        _write_stored_tier(None)
    elif normalized in TIERS:
        _write_stored_tier(normalized)
    else:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS} or 'auto'")
    return refresh_profile()


def get_tier_override() -> Optional[str]:
    """The persisted tier override, or None when detection is in charge."""
    return _read_stored_tier()


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

def _limit(name: str) -> Any:
    """Read one limit from the active profile. None when it must not clamp.

    Deliberately does NOT trigger detection: this runs on the request path,
    and the hardware probe takes a couple of seconds. Until the background
    probe lands, the profile is "unknown" and nothing is clamped — a wrong
    clamp would be worse than no clamp.
    """
    profile = _cached_profile()
    if profile is None:
        return None
    limits = profile.get("limits") or {}
    return limits.get(name)


def context_window_cap() -> Optional[int]:
    """Max conversation tokens the agent may keep for a LOCAL model."""
    cap = _limit("context_window_cap")
    return int(cap) if cap else None


def max_tokens_cap() -> Optional[int]:
    """Max output tokens to request from a LOCAL model."""
    cap = _limit("max_tokens_cap")
    return int(cap) if cap else None


def context_messages_cap() -> Optional[int]:
    """Max messages of history to keep for a LOCAL model."""
    cap = _limit("context_messages_cap")
    return int(cap) if cap else None


def gpu_features_allowed() -> bool:
    """Whether optional in-process GPU features (TTS/STT/upscaling) may use VRAM.

    This never affects the user's own model server — only the optional
    helpers that live inside this process.

    Defaults to True while the profile is unknown: an undetermined machine
    must keep the behaviour it shipped with.
    """
    allowed = _limit("allow_gpu_features")
    return True if allowed is None else bool(allowed)


def apply_env_defaults(profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply the profile's process-level defaults.

    Called once during startup (non-blocking variant) and again by the
    background probe once detection lands, so that CPU-bound helpers
    (ONNX Runtime embeddings) size their thread pools for the machine
    instead of for the host's core count.

    Never triggers detection itself — a caller that passes no profile and has
    no profile cached gets the permissive "unknown" profile, which applies
    nothing. That keeps startup off the critical path.
    """
    if profile is None:
        profile = _cached_profile() or _pending_profile()
    limits = profile.get("limits") or {}
    threads = int(limits.get("embedding_threads") or 0)
    if threads:
        # setdefault: an explicit operator setting always wins.
        os.environ.setdefault("FASTEMBED_THREADS", str(threads))
    tier = profile.get("tier")
    if tier and tier != TIER_STANDARD and profile.get("ready"):
        logger.info(
            "Low-resource mode active (tier=%s): local models are capped at "
            "%s context tokens / %s output tokens",
            tier,
            limits.get("context_window_cap"),
            limits.get("max_tokens_cap"),
        )
    return profile


def _cached_profile() -> Optional[Dict[str, Any]]:
    """The cached profile, or None when detection has not completed yet."""
    with _profile_lock:
        return _profile


def clamp_max_tokens(requested: Optional[int], is_local: bool) -> Optional[int]:
    """Clamp an output-token request for a local model.

    Remote endpoints are never clamped — they are not running on this
    machine's VRAM.
    """
    if not is_local:
        return requested
    cap = max_tokens_cap()
    if not cap:
        return requested
    if not requested or requested <= 0:
        return requested
    return min(int(requested), cap)


def clamp_context_window(context_tokens: int, is_local: bool) -> int:
    """Clamp a conversation-token budget for a local model."""
    if not is_local:
        return context_tokens
    cap = context_window_cap()
    if not cap:
        return context_tokens
    if not context_tokens or context_tokens <= 0:
        # An unproven window (0) means "unknown" to the callers — substituting
        # the cap here would turn "don't budget off this" into a hard limit.
        return context_tokens
    return min(int(context_tokens), cap)


def clamp_payload_max_tokens(payload: Dict[str, Any], is_local: bool) -> Dict[str, Any]:
    """Clamp the output budget inside an outgoing request payload.

    Handles both spellings the OpenAI-compatible world uses
    (``max_tokens`` and ``max_completion_tokens``), and leaves a payload with
    no explicit budget alone — the server's own default is a separate
    decision, and injecting one here would change behaviour for every caller
    that deliberately relied on the server default.
    """
    if not is_local:
        return payload
    cap = max_tokens_cap()
    if not cap:
        return payload
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            payload[key] = min(value, cap)
    return payload

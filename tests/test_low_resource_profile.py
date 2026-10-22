"""Regression tests for the low-resource hardware profile.

The feature is strictly additive: on a roomy machine the tier is ``standard``
and every clamp is a no-op, so upstream behaviour is preserved. These tests
pin that contract, plus the "only local endpoints are clamped" rule, which is
the single most important safety property of the feature — a hosted API runs
on someone else's hardware and must not be shrunk.
"""

import importlib
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture()
def low_resource(monkeypatch):
    """A fresh copy of the module with a clean profile cache."""
    monkeypatch.delenv("ODYSSEUS_LOW_RESOURCE_TIER", raising=False)
    mod = importlib.import_module("src.low_resource")
    importlib.reload(mod)
    monkeypatch.setattr(mod, "_profile", None, raising=False)
    monkeypatch.setattr(mod, "_detect_started", False, raising=False)
    # Keep the stored-override file out of the way of the tests.
    monkeypatch.setattr(mod, "_read_stored_tier", lambda: None, raising=False)
    monkeypatch.setattr(mod, "_write_stored_tier", lambda tier: None, raising=False)
    yield mod
    importlib.reload(mod)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "system, expected",
    [
        # No GPU at all -> CPU-only box -> the tightest tier.
        ({"has_gpu": False, "gpu_vram_gb": 0, "total_ram_gb": 32}, "minimal"),
        ({"has_gpu": True, "gpu_vram_gb": 0, "total_ram_gb": 32}, "minimal"),
        # Tiny card.
        ({"has_gpu": True, "gpu_vram_gb": 2, "total_ram_gb": 16}, "minimal"),
        # 4 GB card.
        ({"has_gpu": True, "gpu_vram_gb": 4, "total_ram_gb": 16}, "low"),
        # 8 GB card.
        ({"has_gpu": True, "gpu_vram_gb": 8, "total_ram_gb": 16}, "medium"),
        # 24 GB card on a roomy box -> no limits.
        ({"has_gpu": True, "gpu_vram_gb": 24, "total_ram_gb": 64}, "standard"),
        # Plenty of VRAM but very little RAM: the weights may not even fit.
        ({"has_gpu": True, "gpu_vram_gb": 24, "total_ram_gb": 2}, "minimal"),
        # 12 GB of VRAM but only 6 GB of RAM -> one step down from standard.
        ({"has_gpu": True, "gpu_vram_gb": 12, "total_ram_gb": 6}, "medium"),
    ],
)
def test_classify_maps_hardware_to_tier(low_resource, system, expected):
    assert low_resource._classify(system) == expected


def test_classify_survives_garbage_hardware(low_resource):
    """A malformed probe result must not raise — it just means "unknown"."""
    assert low_resource._classify({}) == "minimal"
    assert low_resource._classify({"has_gpu": True, "gpu_vram_gb": "eight"}) == "minimal"
    assert low_resource._classify({"has_gpu": True, "gpu_vram_gb": None}) == "minimal"


def test_standard_tier_has_no_limits(low_resource):
    """The whole point of "standard" is that it clamps nothing."""
    limits = low_resource._TIER_LIMITS["standard"]
    assert limits["context_window_cap"] is None
    assert limits["max_tokens_cap"] is None
    assert limits["context_messages_cap"] is None
    assert limits["allow_gpu_features"] is True


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def test_env_var_overrides_everything(low_resource, monkeypatch):
    monkeypatch.setenv("ODYSSEUS_LOW_RESOURCE_TIER", "minimal")
    profile = low_resource._resolve_profile(
        {"has_gpu": True, "gpu_vram_gb": 24, "total_ram_gb": 64}
    )
    assert profile["tier"] == "minimal"
    assert profile["source"] == "env"


def test_stored_override_beats_detection(low_resource, monkeypatch):
    monkeypatch.setattr(low_resource, "_read_stored_tier", lambda: "low")
    profile = low_resource._resolve_profile(
        {"has_gpu": True, "gpu_vram_gb": 24, "total_ram_gb": 64}
    )
    assert profile["tier"] == "low"
    assert profile["source"] == "override"


def test_set_tier_rejects_unknown_tier(low_resource):
    with pytest.raises(ValueError):
        low_resource.set_tier("bogus")


def test_set_tier_auto_clears_the_override(low_resource):
    low_resource.set_tier("minimal")
    low_resource.set_tier("auto")
    assert low_resource.get_tier_override() is None


# ---------------------------------------------------------------------------
# Clamping — the safety-critical part
# ---------------------------------------------------------------------------

def test_remote_endpoints_are_never_clamped(low_resource, monkeypatch):
    """A hosted API is not on this box's VRAM — leave it alone."""
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {},
    })
    payload = {"max_tokens": 32768, "max_completion_tokens": 32768}
    assert low_resource.clamp_payload_max_tokens(dict(payload), False) == payload
    assert low_resource.clamp_context_window(100_000, False) == 100_000


def test_local_payload_is_clamped(low_resource, monkeypatch):
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "medium",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["medium"]),
        "hardware": {},
    })
    out = low_resource.clamp_payload_max_tokens(
        {"max_tokens": 32768, "max_completion_tokens": 100_000}, True
    )
    assert out["max_tokens"] == 8192 or out["max_tokens"] == 4096
    assert out["max_completion_tokens"] == 4096


def test_payload_without_an_explicit_budget_is_left_alone(low_resource, monkeypatch):
    """Injecting a budget would change behaviour for callers that relied on
    the server's own default."""
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {},
    })
    payload = {"temperature": 0.7, "messages": []}
    assert low_resource.clamp_payload_max_tokens(dict(payload), True) == payload


def test_unknown_profile_clamps_nothing(low_resource, monkeypatch):
    """Before the background probe lands, the machine is "unknown"."""
    monkeypatch.setattr(low_resource, "_profile", None)
    payload = {"max_tokens": 32768}
    assert low_resource.clamp_payload_max_tokens(dict(payload), True) == payload
    assert low_resource.clamp_context_window(100_000, True) == 100_000
    # An unknown machine keeps its shipped behaviour for GPU features too.
    assert low_resource.gpu_features_allowed() is True


def test_unproven_context_window_stays_unproven(low_resource, monkeypatch):
    """0 means "unknown" to the callers; substituting the cap here would turn
    "don't budget off this" into a hard limit."""
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {},
    })
    assert low_resource.clamp_context_window(0, True) == 0


def test_standard_tier_clamp_is_a_no_op(low_resource, monkeypatch):
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "standard",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["standard"]),
        "hardware": {},
    })
    payload = {"max_tokens": 32768}
    assert low_resource.clamp_payload_max_tokens(dict(payload), True) == payload
    assert low_resource.clamp_context_window(131_072, True) == 131_072


# ---------------------------------------------------------------------------
# GPU feature gating
# ---------------------------------------------------------------------------

def test_gpu_features_disabled_only_on_the_tightest_tier(low_resource, monkeypatch):
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {},
    })
    assert low_resource.gpu_features_allowed() is False

    for tier in ("low", "medium", "standard"):
        monkeypatch.setattr(low_resource, "_profile", {
            "tier": tier,
            "source": "detected",
            "ready": True,
            "limits": dict(low_resource._TIER_LIMITS[tier]),
            "hardware": {},
        })
        assert low_resource.gpu_features_allowed() is True


# ---------------------------------------------------------------------------
# Serve-profile integration
# ---------------------------------------------------------------------------

def test_compute_serve_profiles_still_returns_profiles_on_a_capped_box(monkeypatch):
    """A tier cap below the context floor must yield one profile, not none."""
    from services.hwfit.profiles import compute_serve_profiles
    from src import low_resource

    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {"has_gpu": True, "gpu_vram_gb": 2},
    })

    system = {
        "backend": "cuda",
        "gpu_name": "NVIDIA RTX 3060",
        "gpu_vram_gb": 2.0,
        "total_ram_gb": 16.0,
        "cpu_cores": 8,
        "has_gpu": True,
    }
    model = {
        "name": "Test-7B",
        "quant": "Q4_K_M",
        "size_gb": 4.68,
        "context_length": 32768,
        "n_layers": 29,
        "n_kv_heads": 4,
        "head_dim": 128,
        "quantizations": [{"label": "Q4_K_M", "size_gb": 4.68, "name": "Q4_K_M"}],
    }
    profiles = compute_serve_profiles(system, model)
    assert profiles, "a capped box must still get at least one serve profile"
    assert all(p["ctx"] <= 2048 for p in profiles)


# ---------------------------------------------------------------------------
# Context-window integration
# ---------------------------------------------------------------------------

def test_local_window_is_capped_before_becoming_num_ctx(low_resource, monkeypatch):
    """Ollama reserves ``options.num_ctx`` as KV cache. A model advertising a
    128K window must not make an 8 GB card reserve 128K of it."""
    from src import model_context

    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "medium",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["medium"]),
        "hardware": {},
    })
    monkeypatch.setattr(model_context, "_query_context_length",
                        lambda url, model: (131_072, True))

    url = "http://127.0.0.1:11434/v1/chat/completions"
    assert model_context.get_context_length(url, "qwen3.5-4b") == \
        low_resource._TIER_LIMITS["medium"]["context_window_cap"]


def test_remote_window_survives_the_cap(low_resource, monkeypatch):
    """A hosted model's window costs this machine nothing — do not shrink it."""
    from src import model_context

    model_context._context_cache.clear()
    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {},
    })
    monkeypatch.setattr(model_context, "_query_context_length",
                        lambda url, model: (200_000, True))

    assert model_context.get_context_length(
        "https://api.openai.com/v1/chat/completions", "gpt-5"
    ) == 200_000


def test_compute_serve_profiles_offloads_nothing_without_a_gpu(monkeypatch):
    """`n_gpu_layers: 999` asks for a GPU offload; a CPU-only box cannot run it.

    This is the shape a Windows box with a display adapter but no CUDA reports:
    a GPU name and a small VRAM number, but a CPU backend.
    """
    from services.hwfit.profiles import compute_serve_profiles
    from src import low_resource

    monkeypatch.setattr(low_resource, "_profile", {
        "tier": "minimal",
        "source": "detected",
        "ready": True,
        "limits": dict(low_resource._TIER_LIMITS["minimal"]),
        "hardware": {"has_gpu": False, "gpu_vram_gb": 0},
    })

    system = {
        "backend": "cpu_x86",
        "gpu_name": "Microsoft Basic Display Adapter",
        "gpu_vram_gb": 0.5,
        "total_ram_gb": 16.0,
        "cpu_cores": 8,
        "has_gpu": False,
    }
    model = {
        "name": "Test-7B",
        "quant": "Q4_K_M",
        "size_gb": 4.68,
        "context_length": 32768,
        "n_layers": 29,
        "n_kv_heads": 4,
        "head_dim": 128,
        "quantizations": [{"label": "Q4_K_M", "size_gb": 4.68, "name": "Q4_K_M"}],
    }
    profiles = compute_serve_profiles(system, model)
    assert profiles
    assert all(p["n_gpu_layers"] == 0 for p in profiles)


def test_compute_serve_profiles_still_offloads_on_a_gpu_box():
    """A GPU box keeps the upstream "offload as much as fits" behaviour."""
    from services.hwfit.profiles import compute_serve_profiles

    system = {
        "backend": "cuda",
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_vram_gb": 24.0,
        "total_ram_gb": 64.0,
        "cpu_cores": 16,
        "has_gpu": True,
    }
    model = {
        "name": "Test-7B",
        "quant": "Q4_K_M",
        "size_gb": 4.68,
        "context_length": 32768,
        "n_layers": 29,
        "n_kv_heads": 4,
        "head_dim": 128,
        "quantizations": [{"label": "Q4_K_M", "size_gb": 4.68, "name": "Q4_K_M"}],
    }
    profiles = compute_serve_profiles(system, model)
    assert profiles
    assert all(p["n_gpu_layers"] == 999 for p in profiles)

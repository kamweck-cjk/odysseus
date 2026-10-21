"""Low-resource profile API.

Exposes the machine's detected resource tier and lets an operator pin it.
The tier is what drives the conservative limits the app applies to *local*
model servers (context window, output budget, GPU feature gating) — see
``src/low_resource.py``.

The profile is a property of the machine, not of a user, so it is stored in a
machine-level settings file and is shared by every user of the install.
"""
from fastapi import APIRouter, Depends, HTTPException

from core.middleware import require_admin


def setup_lowgpu_routes():
    router = APIRouter(prefix="/api/lowgpu", tags=["lowgpu"])

    # Sync handlers on purpose: the hardware probe shells out (nvidia-smi,
    # PowerShell) and takes a couple of seconds. A `def` route runs in
    # FastAPI's threadpool, so a slow probe never stalls the event loop.
    @router.get("/profile")
    def read_profile(fresh: bool = False):
        """Return the active low-resource profile.

        `fresh=true` re-runs the hardware probe instead of using the cached
        result (the hardware rarely changes, so the probe is cached).
        """
        import src.low_resource as low_resource

        profile = (
            low_resource.refresh_profile() if fresh else low_resource.get_profile()
        )
        return {
            "profile": profile,
            "tier_override": low_resource.get_tier_override(),
            "tiers": list(low_resource.TIERS),
        }

    @router.get("/tiers")
    def read_tiers():
        """The limit table for every tier, for rendering a picker."""
        import src.low_resource as low_resource

        return {
            "tiers": list(low_resource.TIERS),
            "limits": {
                tier: dict(low_resource._TIER_LIMITS.get(tier, {}))
                for tier in low_resource.TIERS
            },
        }

    @router.post("/profile")
    def write_profile(body: dict, _admin=Depends(require_admin)):
        """Pin the resource tier, or hand control back to detection.

        Body: {"tier": "low"} to pin a tier, {"tier": "auto"} to clear the
        override and go back to detecting from the hardware.
        """
        import src.low_resource as low_resource

        tier = (body or {}).get("tier")
        try:
            profile = low_resource.set_tier(tier)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "profile": profile,
            "tier_override": low_resource.get_tier_override(),
        }

    return router

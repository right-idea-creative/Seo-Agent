"""
SiteProfile — canonical per-website business metadata.

Stored at: profiles/{client_id}/{website_id}/site.json
This file sits alongside the existing visual_style.json in the same directory.

This is the single source of truth for business name, niche, primary service,
and target location. The generate command loads it automatically so users do not
need to pass --city, --state, or --service on every invocation.

CLI flags (--city, --state, --service) override these values per-invocation
but never modify the file.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SiteProfile(BaseModel):
    """Business metadata for a single website, keyed by client_id + website_id."""

    client_id: str = Field(description="Must match the TenantContext client_id.")
    website_id: str = Field(description="Must match the TenantContext website_id.")

    # ── Business identity ─────────────────────────────────────────────────────

    business_name: str = Field(
        description="Full business name, e.g. 'Overhead Door of Denver'."
    )
    niche: str = Field(
        description="The trade or service category, e.g. 'Garage Door'."
    )
    primary_service: str = Field(
        description="The main service this site targets, e.g. 'Garage Door Repair'."
    )
    secondary_services: list[str] = Field(
        default_factory=list,
        description=(
            "Additional services offered. Used to give the planner full service context. "
            "Example: ['Garage Door Installation', 'Spring Replacement', 'Opener Repair']."
        ),
    )

    # ── Local SEO ─────────────────────────────────────────────────────────────

    city: str = Field(
        description="Primary target city for local SEO, e.g. 'Denver'."
    )
    state: str = Field(
        description="State or province abbreviation, e.g. 'CO'."
    )
    country: str = Field(default="USA")
    region: str | None = Field(
        default=None,
        description="Service area or metro label, e.g. 'Denver Metro Area'. Optional.",
    )

    # ── Content reuse ─────────────────────────────────────────────────────────

    reuse_group: str | None = Field(
        default=None,
        description=(
            "Optional group name that enables content sharing across websites from different "
            "clients. Websites with the same non-empty reuse_group may reuse each other's "
            "draft articles. Example: 'garage-door-network'. "
            "Websites with the same client_id can always share content regardless of this field."
        ),
    )

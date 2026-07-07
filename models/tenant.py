import re

from pydantic import BaseModel, Field, field_validator


class TenantContext(BaseModel):
    """
    Identifies ownership of an article within the multi-tenant hierarchy.

    Hierarchy: Client (agency or company)
                 └── Dealer (franchise / branch — optional)
                       └── Website (WordPress site)
                             └── Article

    These are opaque string IDs. The corresponding Client, Dealer,
    and Website models will be defined in V2 when the tenant management
    module is built. Using strings (not UUIDs) allows both integer IDs
    from a CRM and slug-style identifiers like 'client-acme-co'.
    """

    client_id: str = Field(
        description="Unique identifier of the client or agency."
    )
    website_id: str = Field(
        description="Unique identifier of the WordPress site within the client."
    )
    dealer_id: str | None = Field(
        default=None,
        description="Franchise, branch, or dealer identifier (if the client uses a franchise model)."
    )

    @field_validator("client_id", "website_id")
    @classmethod
    def _validate_path_safe(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                "Only letters, digits, hyphens, and underscores are allowed. "
                "Spaces and path separators (/ \\ . :) are not permitted."
            )
        return v

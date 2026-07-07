from datetime import datetime

from pydantic import BaseModel, Field

from models.enums import PublishStatus, SEOPlugin


class PublishingOptions(BaseModel):
    """
    Publishing configuration for an article.

    Kept separate from Article intentionally: publishing concerns
    (platform, schedule, SEO plugin) are orthogonal to content.
    The same article can be re-published with different options
    without touching the content model.

    Future compatible with: WordPress REST API (V2), scheduled posts,
    Yoast/Rank Math meta injection, Google Drive export (V2).
    """

    # ── Publish state ─────────────────────────────────────────

    status: PublishStatus = Field(
        default=PublishStatus.DRAFT,
        description="Target publish state: draft, publish, or future (scheduled)."
    )
    scheduled_at: datetime | None = Field(
        default=None,
        description="UTC datetime for scheduled publishing. Required when status=FUTURE."
    )

    # ── WordPress ─────────────────────────────────────────────
    # Tags and category are sourced from SEOMetadata.suggested_tags /
    # suggested_category. The PublishingAgent (V2) reads those fields
    # directly, so we do not duplicate them here.

    wp_author_id: int | None = Field(
        default=None,
        description="WordPress user ID of the post author."
    )

    # ── SEO Plugin ────────────────────────────────────────────

    seo_plugin: SEOPlugin = Field(
        default=SEOPlugin.NONE,
        description="SEO plugin installed on WordPress: yoast, rankmath, or none."
    )

    # ── Google Drive ──────────────────────────────────────────

    drive_folder_id: str | None = Field(
        default=None,
        description="Google Drive folder ID where the article document will be saved (V2)."
    )

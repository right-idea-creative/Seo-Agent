from pydantic import BaseModel, Field

from models.enums import ImageRole


class ImageMetadata(BaseModel):
    """
    Metadata for an image associated with an article.

    Designed to support the complete SEO-Agent workflow including
    Google Drive sourcing, WordPress uploads, featured images,
    accessibility, and SEO optimization.

    Future compatible with: Google Drive (V2), WordPress Media Library (V2),
    CDN delivery, and AI-generated captions.
    """

    # ── Basic Information ──────────────────────────────────────

    original_filename: str = Field(
        description="Original filename, e.g. 'garage-door-repair-denver.jpg'"
    )
    alt_text: str = Field(
        description="SEO-friendly alternative text for accessibility."
    )
    caption: str | None = Field(
        default=None,
        description="Optional caption displayed below the image."
    )
    credits: str | None = Field(
        default=None,
        description="Photographer or image source."
    )

    # ── SEO ───────────────────────────────────────────────────

    source_keyword: str | None = Field(
        default=None,
        description="Keyword associated with this image for contextual SEO."
    )

    # ── Position inside article ───────────────────────────────

    position: int | None = Field(
        default=None,
        description="Order of appearance within the article (1-indexed)."
    )
    role: ImageRole | None = Field(
        default=None,
        description="Semantic role: featured, hero, inline, gallery, infographic."
    )
    is_featured: bool = Field(
        default=False,
        description="Whether this image becomes the WordPress Featured Image."
    )

    # ── Storage ───────────────────────────────────────────────

    url: str | None = Field(
        default=None,
        description="Public URL after upload to WordPress or CDN."
    )
    drive_file_id: str | None = Field(
        default=None,
        description="Google Drive file ID (V2)."
    )
    wordpress_media_id: int | None = Field(
        default=None,
        description="WordPress Media Library ID after upload (V2)."
    )
    uploaded_filename: str | None = Field(
        default=None,
        description="Filename assigned after upload (may differ from original)."
    )

    # ── Technical ─────────────────────────────────────────────

    width: int | None = Field(default=None, description="Image width in pixels.")
    height: int | None = Field(default=None, description="Image height in pixels.")
    mime_type: str | None = Field(
        default=None,
        description="Image MIME type, e.g. 'image/jpeg', 'image/webp'."
    )

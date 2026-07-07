from enum import Enum

from pydantic import BaseModel, Field


class ImageSource(str, Enum):
    DRIVE     = "drive"      # retrieved from Google Drive
    GENERATED = "generated"  # created by an AI image generator


class ImageAsset(BaseModel):
    """
    Pre-upload image data transferred between ImageResolverAgent and MediaService.

    Carries raw bytes along with the metadata needed to upload the image to
    WordPress and populate ImageMetadata afterward. Never serialized to disk —
    lives only in memory during a publish run.

    source_detail records provenance:
    - DRIVE: the Google Drive file_id of the original file.
    - GENERATED: the first 500 chars of the prompt used for generation.

    similarity_score, selection_reason, and drive_path are audit/observability
    fields populated by ImageResolverAgent. They are never used by the publishing
    pipeline — only by the display layer and future analysis tooling.
    """

    filename: str = Field(description="Filename to use on upload, e.g. 'garaje-denver.jpg'.")
    mime_type: str = Field(description="MIME type, e.g. 'image/jpeg'.")
    data: bytes = Field(description="Raw image bytes.")
    alt_text: str = Field(description="SEO-optimized alt text.")
    caption: str | None = Field(default=None, description="Optional display caption.")
    source: ImageSource
    source_detail: str | None = Field(
        default=None,
        description="Drive file_id (DRIVE) or truncated generation prompt (GENERATED)."
    )
    similarity_score: int | None = Field(
        default=None,
        description=(
            "Visual similarity score 0–100 returned by Claude Vision (Drive only). "
            "None for AI-generated images."
        ),
    )
    selection_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable reason this image was selected. "
            "For audit, debugging, and future algorithm improvements. "
            "Never used by the publishing pipeline."
        ),
    )
    drive_path: str | None = Field(
        default=None,
        description=(
            "Relative Drive path (folder_path/name) for Drive images. "
            "None for AI-generated images."
        ),
    )
    vision_reasoning: str | None = Field(
        default=None,
        description=(
            "Full untruncated reasoning text returned by Claude Vision when selecting "
            "this image. None for AI-generated images. Stored for audit and future "
            "algorithm calibration."
        ),
    )
    drive_candidates_evaluated: int | None = Field(
        default=None,
        description=(
            "Number of Drive thumbnails actually sent to Claude Vision for this image "
            "(after scoring and thumbnail download). None for AI-generated images."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

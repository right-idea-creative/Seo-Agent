from enum import Enum

from pydantic import BaseModel, Field


class ImageSource(str, Enum):
    DRIVE     = "drive"      # original company photo, used as-is
    EDITED    = "edited"     # original company photo with a minimal preservation edit applied
    GENERATED = "generated"  # AI-generated from scratch (validate images command only — not produced by main pipeline)


class ImageAsset(BaseModel):
    """
    Pre-upload image data transferred between ImageResolverAgent and MediaService.

    Carries raw bytes along with the metadata needed to upload the image to
    WordPress and populate ImageMetadata afterward. Never serialized to disk —
    lives only in memory during a publish run.

    Source values:
      DRIVE   — original company photo from Google Drive, published without modification.
      EDITED  — original company photo with one minimal preservation edit applied.
                The original photograph remains the source of truth. The AI only
                removes a distraction, extends canvas, or adjusts lighting in a
                bounded region. Approximately 95–99% of original pixels are preserved.
      GENERATED — produced only by `validate images` command; never by the main pipeline.

    source_detail records provenance:
    - DRIVE:    the Google Drive file_id of the original file.
    - EDITED:   the exact preservation edit prompt sent to images.edit().

    reference_file_id is set for EDITED images — the Drive file_id of the photo
    that was edited, so the caller can mark it as used.

    original_data holds the pre-edit Drive photo bytes for QA comparison.
    Both Claude Vision and OpenAI Vision receive the original alongside the edit
    and evaluate: "Does this still look like the same company photograph?"

    edit_type and edit_prompt record exactly what was done for the pipeline report.
    preservation_estimate is Claude's estimate of original pixels preserved (90–100).
    """

    filename: str = Field(description="Filename to use on upload, e.g. 'garaje-denver.jpg'.")
    mime_type: str = Field(description="MIME type, e.g. 'image/jpeg'.")
    data: bytes = Field(description="Raw image bytes (edited version for EDITED, original for DRIVE).")
    alt_text: str = Field(description="SEO-optimized alt text.")
    caption: str | None = Field(default=None, description="Optional display caption.")
    source: ImageSource
    source_detail: str | None = Field(
        default=None,
        description=(
            "Drive file_id (DRIVE/EDITED) or generation prompt (GENERATED)."
        ),
    )
    reference_file_id: str | None = Field(
        default=None,
        description=(
            "For EDITED images: the Drive file_id of the photo that was edited. "
            "None for DRIVE and GENERATED images."
        ),
    )
    ai_reason: str | None = Field(
        default=None,
        description=(
            "For EDITED images: why a preservation edit was applied instead of publishing as-is. "
            "None for DRIVE images."
        ),
    )
    similarity_score: int | None = Field(
        default=None,
        description="Visual similarity score 0–100 returned by Claude Vision during Drive search.",
    )
    selection_reason: str | None = Field(
        default=None,
        description="Human-readable reason this photo was selected or how it was edited.",
    )
    drive_path: str | None = Field(
        default=None,
        description="Relative Drive path (folder_path/name) of the source company photo.",
    )
    vision_reasoning: str | None = Field(
        default=None,
        description="Full reasoning text returned by Claude Vision during Drive search.",
    )
    drive_candidates_evaluated: int | None = Field(
        default=None,
        description="Number of Drive thumbnails evaluated by Claude Vision for this slot.",
    )
    # ── Preservation edit metadata ─────────────────────────────────────────────
    original_data: bytes | None = Field(
        default=None,
        description=(
            "For EDITED images: the unmodified Drive photo bytes. "
            "Passed to both QA reviewers so they can compare original vs. edited "
            "and evaluate identity preservation. Never uploaded to WordPress."
        ),
    )
    edit_type: str | None = Field(
        default=None,
        description=(
            "Category of edit applied: remove_object, canvas_extension, "
            "exposure_adjustment, background_cleanup, day_to_night, "
            "color_correction, remove_text, crop_adjustment, general_cleanup. "
            "None for DRIVE and GENERATED images."
        ),
    )
    edit_prompt: str | None = Field(
        default=None,
        description=(
            "The exact preservation edit prompt sent to images.edit(), "
            "including the preservation prefix. Shown in pipeline report."
        ),
    )
    preservation_estimate: int | None = Field(
        default=None,
        description=(
            "Estimated percentage of original pixels preserved (90–100). "
            "Determined by Claude when selecting the minimal edit."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

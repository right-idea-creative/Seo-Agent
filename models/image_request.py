from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ImageType(str, Enum):
    """
    What the image depicts — guides both Drive search and AI prompt construction.

    INFOGRAPHIC is recognized but MVP generation falls back to a descriptive
    photograph. When Ideogram is integrated it will activate automatically for
    this type without changing any other code.
    """
    PHOTOGRAPH    = "photograph"     # generic professional photo
    PROCESS_PHOTO = "process_photo"  # procedure or technique being performed
    PRODUCT_PHOTO = "product_photo"  # product, tool, or equipment close-up
    TEAM_PHOTO    = "team_photo"     # technicians or professionals at work
    PROBLEM_PHOTO = "problem_photo"  # visual of a problem being described
    BEFORE_AFTER  = "before_after"   # comparison or transformation
    INFOGRAPHIC   = "infographic"    # informational graphic (future: Ideogram)


class ImagePurpose(str, Enum):
    FEATURED = "featured"   # WordPress featured_media — no markdown marker
    INLINE   = "inline"     # embedded in article body via HTML marker


class ImageRequest(BaseModel):
    """
    A single image the agent has decided to place in the article.

    Produced during the planning phase. Contains enough context to:
    - Search Google Drive semantically for a matching existing image.
    - Build a photorealistic AI generation prompt as fallback.
    - Set alt text, caption, and SEO metadata after upload.

    The placement_marker property returns the HTML comment string that the
    planning phase inserts into modified_markdown for INLINE images. FEATURED
    images have no marker — they are set via WordPress's featured_media field.
    """

    id: str = Field(
        description="Unique image ID within the plan, e.g. 'img_001'."
    )
    purpose: ImagePurpose
    image_type: ImageType
    section_title: str | None = Field(
        default=None,
        description="H2/H3 section this image illustrates. None for featured images."
    )
    subject: str = Field(
        description="Precise visual description of what the image should show."
    )
    communicative_intent: str = Field(
        description="Why this image is placed here — what value it adds to the reader."
    )
    related_keyword: str | None = Field(
        default=None,
        description="SEO keyword associated with this image."
    )
    alt_text: str = Field(
        description="SEO-optimized alt text for accessibility."
    )
    caption: str | None = Field(
        default=None,
        description="Optional caption displayed below the image."
    )

    @property
    def placement_marker(self) -> str:
        """HTML comment marker inserted into modified_markdown for INLINE images."""
        return f"<!-- SEO_AGENT_IMAGE: {self.id} -->"


class ImagePlacementPlan(BaseModel):
    """
    Result of the image planning phase.

    Contains all image requests for an article plus the modified version of
    content_markdown with placement markers already inserted for INLINE images.
    The first request is always FEATURED; the rest are INLINE in document order.

    modified_markdown replaces content_markdown in the publish payload so that
    markers survive the Markdown → HTML conversion and can be replaced with
    actual <figure> tags after upload.
    """

    requests: list[ImageRequest] = Field(
        description="All image requests ordered: FEATURED first, INLINE by position."
    )
    modified_markdown: str = Field(
        description=(
            "content_markdown with <!-- SEO_AGENT_IMAGE: id --> markers inserted "
            "before each inline image position. Featured image has no marker."
        )
    )
    reasoning: str = Field(
        description="Editorial reasoning explaining all placement decisions."
    )

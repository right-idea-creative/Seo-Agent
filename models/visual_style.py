from datetime import datetime

from pydantic import BaseModel, Field


class VisualStyleProfile(BaseModel):
    """
    Brand visual identity learned from the shared Google Drive photo library.

    One global profile per business — not per client or website.
    Cached as JSON in profiles/visual_style.json.

    Analysis is purely visual: filenames, folder paths, and metadata are
    intentionally ignored. The profile is invalidated when the Drive folder
    changes by more than 5 % or the cache is older than 30 days.

    prompt_guidelines is the most critical field: direct instructions appended
    verbatim to every DALL-E generation prompt to keep AI images consistent
    with the business's real photography.
    """

    drive_folder_id: str = Field(description="Global Drive folder ID — used for cache invalidation.")
    analyzed_at: datetime
    image_count: int = Field(description="Total images found in the folder at analysis time.")
    analyzed_count: int = Field(default=0, description="Images sent to Claude Vision for analysis.")
    filtered_count: int = Field(default=0, description="Images kept after filtering non-photographs.")

    photography_style: str = Field(
        description="Overall photographic style, e.g. 'professional, natural light, outdoor jobsite'."
    )
    lighting: str = Field(
        default="",
        description="Dominant lighting conditions across the photo library."
    )
    composition: str = Field(
        default="",
        description="Typical composition patterns (wide-angle, close-up, environmental, etc.)."
    )
    typical_scenarios: list[str] = Field(
        default_factory=list,
        description="Recurring visual scenarios found in the photo library (3–8 items)."
    )
    color_palette: list[str] = Field(
        default_factory=list,
        description="Dominant colors as descriptive names or hex values (3–6 items)."
    )
    prompt_guidelines: str = Field(
        description=(
            "Direct instructions for DALL-E 3 to match this brand's visual identity. "
            "Appended verbatim to every generation prompt. Written in imperative form."
        )
    )
    style_description: str = Field(
        description="Comprehensive narrative description of the brand's visual identity (2–4 sentences)."
    )
    version: str = "1.0"

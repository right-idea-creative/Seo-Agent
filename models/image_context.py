from pydantic import BaseModel, Field

from models.enums import ArticleTone


class ImageContext(BaseModel):
    """
    Article-level context used to guide image search and generation.

    Built once per publish run from the Article object. Passed to
    VisualStyleService (for style analysis) and ImageResolverAgent
    (for per-image Drive search and prompt construction).
    """

    title: str
    focus_keyword: str
    service: str | None = None
    location: str | None = Field(
        default=None,
        description="Human-readable location, e.g. 'Denver, CO'."
    )
    category: str | None = None
    content_excerpt: str = Field(
        default="",
        description="First 300 words of content_markdown for semantic context."
    )
    tone: ArticleTone
    client_id: str
    website_id: str

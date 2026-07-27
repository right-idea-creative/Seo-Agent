import re
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from models.article_plan import PlannedImage
from models.enums import ArticleLanguage, ArticleStatus, ArticleTone
from models.location import Location
from models.media import ImageMetadata
from models.publishing import PublishingOptions
from models.tenant import TenantContext


class ArticleRequest(BaseModel):
    """
    The creative brief handed to the SEO writer agent.

    Captures everything the agent needs to generate a complete article:
    topic, geographic target, service context, and writing configuration.
    The agent may enrich or adjust SEO fields during generation.
    """

    # ── Content brief ─────────────────────────────────────────

    topic: str = Field(
        description="Main topic or title idea, e.g. 'Garage door repair in Denver'."
    )
    service: str | None = Field(
        default=None,
        description="Service or product the article supports, e.g. 'garage door repair'."
    )
    objective: str | None = Field(
        default=None,
        description="Goal: attract organic traffic, convert leads, educate customers, etc."
    )

    # ── Local SEO ─────────────────────────────────────────────

    location: Location | None = Field(
        default=None,
        description="Geographic target. Required for local SEO articles."
    )

    # ── Writing configuration ─────────────────────────────────

    language: ArticleLanguage = Field(
        default=ArticleLanguage.ES,
        description="Language of the generated article."
    )
    word_count: int = Field(
        default=850,
        ge=300,
        le=10000,
        description="Target word count for the article body. Production target: 850 words (700–1000).",
    )
    tone: ArticleTone = Field(
        default=ArticleTone.PROFESIONAL,
        description="Writing tone to apply throughout the article."
    )
    target_audience: str | None = Field(
        default=None,
        description="Who this article is written for, e.g. 'homeowners in Denver'."
    )

    # ── Website context ───────────────────────────────────────

    website_url: str | None = Field(
        default=None,
        description=(
            "WordPress site URL loaded from credentials, e.g. 'https://ohdeugene.com'. "
            "Used by the planner to infer city and service when they are not explicitly provided. "
            "Not published or stored in the article output."
        ),
    )

    # ── SEO hints ─────────────────────────────────────────────

    focus_keyword: str | None = Field(
        default=None,
        description="Suggested primary keyword. Agent may adjust based on context."
    )
    internal_links_to_include: list[str] = Field(
        default_factory=list,
        description="Internal URLs to reference within the article."
    )
    competitor_urls: list[str] = Field(
        default_factory=list,
        description="Competitor URLs for content gap analysis (V3 — Ahrefs)."
    )


class SEOMetadata(BaseModel):
    """
    Complete SEO metadata for a published article.

    Populated by the SEO writer agent via Claude tool_use,
    guaranteeing structured output without string parsing.

    Designed to support Yoast, Rank Math, Open Graph, and future
    integrations with Ahrefs keyword data and Search Console metrics.
    """

    # ── Core SEO ──────────────────────────────────────────────

    seo_title: str = Field(
        max_length=70,
        description="Title tag — 60 chars max recommended for full SERP display."
    )
    meta_description: str = Field(
        max_length=170,
        description="Meta description — 160 chars max recommended."
    )
    slug: str = Field(
        description="URL-safe slug, e.g. 'reparacion-puertas-garaje-denver'."
    )

    # ── Keywords ──────────────────────────────────────────────

    focus_keyword: str = Field(
        description="Primary keyword the article is optimized for."
    )
    secondary_keywords: list[str] = Field(
        default_factory=list,
        description="Supporting keywords to include naturally in the content."
    )

    # ── Taxonomy ──────────────────────────────────────────────

    suggested_tags: list[str] = Field(
        default_factory=list,
        description="Suggested WordPress tags."
    )
    suggested_category: str | None = Field(
        default=None,
        description="Suggested WordPress category."
    )

    # ── Canonicalization ──────────────────────────────────────

    canonical_url: str | None = Field(
        default=None,
        description="Canonical URL if this article mirrors or consolidates another."
    )

    # ── Open Graph ────────────────────────────────────────────

    og_title: str | None = Field(
        default=None,
        description="Open Graph title for social sharing. Falls back to seo_title."
    )
    og_description: str | None = Field(
        default=None,
        description="Open Graph description. Falls back to meta_description."
    )

    # ── SEO Plugin ────────────────────────────────────────────

    seo_plugin_score: int | None = Field(
        default=None,
        description="Readability/SEO score returned by Yoast or Rank Math after publish (V2)."
    )

    # V3 placeholder — do not implement yet
    # keyword_data: KeywordData | None = None


class Article(BaseModel):
    """
    The complete article object produced by the SEO writer agent.

    Represents a finished artifact: content, SEO metadata, media, and
    publishing configuration in a single portable, JSON-serializable object.

    Storage: file-based JSON (MVP) → SQLite/PostgreSQL (V2).
    Transport: REST API response (V2), n8n webhook payload.
    """

    # ── Identity ──────────────────────────────────────────────

    id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(
        default="1.0",
        description="Model version for API and migration compatibility."
    )
    status: ArticleStatus = Field(default=ArticleStatus.PENDING)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Ownership ─────────────────────────────────────────────

    tenant: TenantContext = Field(
        description="Client, website, and dealer this article belongs to."
    )

    # ── Source request ────────────────────────────────────────

    request: ArticleRequest = Field(
        description="The original brief that generated this article."
    )

    # ── Content ───────────────────────────────────────────────

    title: str = Field(description="Final H1 title of the article.")
    content_markdown: str = Field(description="Full article body in Markdown.")
    content_html: str | None = Field(
        default=None,
        description="HTML version, converted from Markdown before WordPress publish."
    )

    # ── Stats — auto-computed via model_validator ──────────────

    word_count: int = Field(
        default=0,
        description="Actual word count of content_markdown, auto-computed at creation."
    )
    reading_time_minutes: int = Field(
        default=0,
        description="Estimated reading time at 200 words/min, auto-computed at creation."
    )

    # ── SEO ───────────────────────────────────────────────────

    seo: SEOMetadata

    # ── Media ─────────────────────────────────────────────────

    featured_image: ImageMetadata | None = None
    images: list[ImageMetadata] = Field(default_factory=list)

    # ── Image plan (from ArticlePlannerService) ───────────────

    image_plans: list[PlannedImage] = Field(
        default_factory=list,
        description=(
            "Image intent plan produced by the article planner. "
            "ImageResolverAgent uses this to skip its own Claude planning call "
            "and resolve images directly from the planner's intent. "
            "Empty when planning was skipped."
        ),
    )

    # ── Publishing ────────────────────────────────────────────

    publishing: PublishingOptions = Field(default_factory=PublishingOptions)

    # ── External IDs — filled after publishing (V2) ───────────

    wp_post_id: int | None = Field(
        default=None,
        description="WordPress post ID after successful publish."
    )
    wp_post_url: str | None = Field(
        default=None,
        description="Live URL of the published WordPress post."
    )
    drive_document_id: str | None = Field(
        default=None,
        description="Google Drive document ID after export."
    )

    # ── Vector DB — filled after embedding (V3) ───────────────

    embedding_id: str | None = Field(
        default=None,
        description="Vector identifier in the semantic database (V3)."
    )

    # ── Traceability ──────────────────────────────────────────

    prompt_version: str = Field(
        default="1.0",
        description="Version of the prompts used to generate this article. Bump when prompts change significantly."
    )
    model_name: str = Field(
        default="",
        description="Claude model used for generation, e.g. 'claude-opus-4-8'."
    )
    topic_id: str | None = Field(
        default=None,
        description=(
            "Stable, location-agnostic topic identifier in kebab-case. "
            "Derived from the article topic with location words stripped. "
            "Example: 'garage-door-spring-repair'. "
            "Used by DraftReuseService for deterministic topic matching before "
            "falling back to Jaccard similarity scoring."
        ),
    )

    # ── Properties ────────────────────────────────────────────

    @property
    def content_plain_text(self) -> str:
        """
        Article body without Markdown markup.

        Computed on demand from content_markdown — never stored.
        Intended for embeddings and semantic search (V3).
        Implementation can be improved (e.g. markdown-it-py parser) without
        changing the interface or any stored data.
        """
        text = re.sub(r'```[\s\S]+?```', ' ', self.content_markdown)
        text = re.sub(r'`[^`]+`', ' ', text)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ── Validators ────────────────────────────────────────────

    @model_validator(mode='after')
    def compute_content_stats(self) -> 'Article':
        """Auto-compute word count and reading time from content_markdown."""
        if self.content_markdown and self.word_count == 0:
            words = len(self.content_markdown.split())
            self.word_count = words
            self.reading_time_minutes = max(1, words // 200)
        return self

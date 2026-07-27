from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.enums import ArticleLanguage, ArticleTone


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-opus-4-8"

    # Per-stage model routing — override individual stages without touching generation quality.
    # Article generation stays on Opus; everything else defaults to Sonnet or Haiku.
    planner_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for article technical planning.",
    )
    seo_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for SEO metadata generation.",
    )
    qa_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for QA review and article revision.",
    )
    link_enricher_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for internal link enrichment.",
    )
    topic_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for topic suggestion.",
    )
    image_eval_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for Drive photo relevance scoring.",
    )
    edit_prompt_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for one-sentence preservation edit instructions.",
    )

    # Article defaults
    default_language: ArticleLanguage = ArticleLanguage.EN
    default_tone: ArticleTone = ArticleTone.PROFESIONAL
    default_word_count: int = 850
    min_article_words: int = Field(
        default=700,
        ge=100,
        description="Minimum article word count required for publication readiness.",
    )

    # Output
    output_dir: Path = Path("output/articles")

    # SEO QA
    seo_qa_min_score: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Minimum SEO quality score required to publish (0-100).",
    )

    # Credentials
    credentials_dir: Path = Path("credentials")

    # CLI defaults
    default_client_id: str | None = None
    default_website_id: str | None = None

    # Google Drive (global)
    # Single shared folder containing all business reference photos.
    # Filenames and folder structure are ignored — analysis is purely visual.
    drive_folder_id: str | None = None

    # Budget
    claude_monthly_budget_usd: float = 1000
    openai_monthly_budget_usd: float = 50.0
    budget_dir: Path = Path("budget")
    max_monthly_cost_usd: float = Field(
        default=50,
        ge=0,
        description="Hard cap on combined monthly API spend (Claude + OpenAI). Pipeline stops when reached.",
    )
    max_article_cost_usd: float = Field(
        default=0.55,
        ge=0,
        description="Target per-article API cost (USD). A warning is printed when exceeded.",
    )
    enable_draft_reuse: bool = Field(
        default=True,
        description=(
            "Search the local article repository before generating. "
            "When a topic-similar article is found, reuse it instead of making API calls."
        ),
    )

    # Image resolution
    openai_api_key: str | None = None
    image_generator_provider: str = "openai"
    image_drive_search_limit: int = 50
    image_style_analysis_limit: int = 30

    # Drive image resolution thresholds (Claude Vision similarity score 0–100)
    # DRIVE_EXACT_SCORE:   score ≥ this → use original Drive photo as-is (Priority 1)
    # DRIVE_PARTIAL_SCORE: score ≥ this → label variation as "strong reference" vs "weak"
    #                      Any score > 0 triggers a variation; this only affects reporting.
    # MAX_OPENAI_IMAGES:   per-article AI budget (variations + generated combined)
    drive_exact_score: int = Field(
        default=75,
        ge=0,
        le=100,
        description="Minimum Claude Vision score to use a Drive photo as-is (Priority 1).",
    )
    drive_partial_score: int = Field(
        default=40,
        ge=0,
        le=100,
        description=(
            "Score threshold that labels a Drive variation as 'strong' (≥ this) "
            "vs 'weak' (< this, but still used as reference). Any score > 0 triggers "
            "a variation instead of generating from scratch."
        ),
    )
    max_openai_images_per_article: int = Field(
        default=1,
        ge=0,
        description=(
            "Maximum AI images per article (variations + generated combined). "
            "Set to 0 to disable AI generation entirely and always reuse Drive photos."
        ),
    )

    # Google Service Account
    google_sa_json_path: Path | None = None

    # Drive image index
    # Local SQLite index of all Drive images. Re-synced only when stale,
    # eliminating redundant Drive API traversals on every publish run.
    drive_index_path: Path = Path("index/drive_images.db")
    drive_sync_max_age_hours: int = 168  # 7 days

    editorial_history_path: Path = Path("output/editorial/image_usage.json")

    # Profile cache
    profiles_dir: Path = Path("profiles")

    # SEO plugin (per-site; "auto" detects via WP REST namespace registry)
    seo_plugin: str = Field(
        default="auto",
        description="SEO plugin: auto | yoast | rankmath | none",
    )

    # V3: Ahrefs (optional)
    ahrefs_api_key: str | None = None

    # Dual QA (production quality gate)
    qa_enabled: bool = Field(
        default=True,
        description=(
            "Enable the dual-reviewer QA gate (Claude + OpenAI) before every publish. "
            "Set to False for fast testing. Never disable in production."
        ),
    )
    qa_max_cycles: int = Field(
        default=3, ge=1, le=10,
        description="Maximum article revision cycles before aborting with a QA failure report.",
    )
    qa_min_seo: int = Field(default=90, ge=0, le=100, description="Minimum Claude SEO score to pass.")
    qa_min_editorial: int = Field(default=90, ge=0, le=100, description="Minimum Claude editorial score to pass.")
    qa_min_writing: int = Field(default=90, ge=0, le=100, description="Minimum OpenAI human writing score to pass.")
    qa_min_authenticity: int = Field(default=90, ge=0, le=100, description="Minimum OpenAI authenticity score to pass.")
    qa_min_vision_claude: int = Field(default=90, ge=0, le=100, description="Minimum Claude Vision score for AI images.")
    qa_min_vision_openai: int = Field(default=90, ge=0, le=100, description="Minimum OpenAI Vision score for AI images.")
    openai_text_review_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for article text review.",
    )
    openai_vision_review_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for image vision review.",
    )
    qa_compliance_check: bool = Field(
        default=False,
        description=(
            "Run revision compliance checker after each QA cycle (reporting only — "
            "does not affect pipeline decisions). Costs 12k-22k tokens per revision. "
            "Enable only when debugging revision quality."
        ),
    )


settings = Settings()

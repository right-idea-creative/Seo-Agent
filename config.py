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

    # Article defaults
    default_language: ArticleLanguage = ArticleLanguage.EN
    default_tone: ArticleTone = ArticleTone.PROFESIONAL
    default_word_count: int = 1000

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
    claude_monthly_budget_usd: float = 10.0
    openai_monthly_budget_usd: float = 10.0
    budget_dir: Path = Path("budget")

    # Image resolution
    openai_api_key: str | None = None
    image_generator_provider: str = "openai"
    image_drive_search_limit: int = 50
    image_style_analysis_limit: int = 30

    # Google Service Account
    google_sa_json_path: Path | None = None

    # Drive image index
    # Local SQLite index of all Drive images. Re-synced only when stale,
    # eliminating redundant Drive API traversals on every publish run.
    drive_index_path: Path = Path("index/drive_images.db")
    drive_sync_max_age_hours: int = 168  # 7 days

    # Profile cache
    profiles_dir: Path = Path("profiles")

    # SEO plugin (per-site; "auto" detects via WP REST namespace registry)
    seo_plugin: str = Field(
        default="auto",
        description="SEO plugin: auto | yoast | rankmath | none",
    )

    # V3: Ahrefs (optional)
    ahrefs_api_key: str | None = None


settings = Settings()

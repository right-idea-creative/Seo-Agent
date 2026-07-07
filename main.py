import logging
import re
import time
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any

import httpx

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents import DryRunReport, ImageResolverAgent, ImageResolverError, PublisherAgent, article_agent, link_enricher
from agents.publisher_agent import SEOQualityError, PublisherAgent as _PA
from config import settings
from models import ArticleRequest, Location, TenantContext
from models.article import Article, SEOMetadata
from services.seo_qa_service import SEOQAService
from models.enums import ArticleLanguage, ArticleTone, PublishStatus, SEOPlugin
from services import ClaudeAPIError, ClaudeRateLimitError, MediaService, WordPressService
from services import budget, GoogleDriveService, OpenAIImageGenerator, VisualStyleService
from services import DriveImageIndex
from services import claude
from services.credential_store import CredentialError, CredentialNotFoundError, CredentialStore
from services.wordpress_service import WordPressAuthError, WordPressError
from models.seo_report import IssueSeverity, SEOReport

@dataclass
class _PipelineState:
    """Collects data from each publish stage to feed _display_pipeline_report()."""
    # Stage 1: WordPress
    conn_ok: bool = False
    conn_error: str | None = None
    auth_ok: bool = False
    auth_user: str | None = None
    auth_error: str | None = None

    # Stage 2: SEO plugin
    seo_plugin: SEOPlugin = SEOPlugin.NONE
    meta_check: dict[str, str] = _dc_field(default_factory=dict)

    # Stage 3: Images
    images_active: bool = False       # whether image resolution ran at all
    images_skip_reason: str = ""
    drive_indexed: int = 0
    img_requested: int = 0
    img_from_drive: int = 0
    img_from_openai: int = 0
    img_uploaded: int = 0
    img_featured: bool = False
    img_errors: list[str] = _dc_field(default_factory=list)

    # Stage 4: HTML
    html_tables: int = 0
    html_callouts: int = 0
    html_faq: bool = False
    html_internal_links: int = 0
    html_external_links: int = 0

    # Stage 4b: SEO QA
    qa_score: int = 0

    # Stage 5: Publish result
    post_id: int | None = None
    post_url: str | None = None
    post_status: str = ""
    post_slug: str = ""

    # Stage 6: Timing (seconds)
    t_images: float = 0.0
    t_upload: float = 0.0
    t_publish: float = 0.0
    t_total: float = 0.0

    # Stage 7: Costs
    claude_input_tokens: int = 0
    claude_output_tokens: int = 0
    claude_cost_usd: float = 0.0
    openai_images_generated: int = 0
    openai_cost_usd: float = 0.0


app = typer.Typer(
    name="seo-agent",
    help="Generate SEO-optimized articles powered by Claude.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def generate(
    topic: str = typer.Option(..., "--topic", "-t", help="Main topic of the article."),
    service: str | None = typer.Option(None, "--service", "-s", help="Service or product the article supports."),
    city: str | None = typer.Option(None, "--city", help="Target city for local SEO."),
    state: str | None = typer.Option(None, "--state", help="Target state or province (required with --city)."),
    words: int = typer.Option(settings.default_word_count, "--words", "-w", min=300, max=10000, help="Target word count."),
    tone: ArticleTone = typer.Option(settings.default_tone, "--tone", help="Writing tone."),
    language: ArticleLanguage = typer.Option(ArticleLanguage.EN, "--language", "-l", help="Article language (default: English)."),
    keyword: str | None = typer.Option(None, "--keyword", "-k", help="Primary keyword hint for the agent."),
    client_id: str | None = typer.Option(None, "--client-id", help="Client identifier (overrides DEFAULT_CLIENT_ID in .env)."),
    website_id: str | None = typer.Option(None, "--website-id", help="Website identifier (overrides DEFAULT_WEBSITE_ID in .env)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory (overrides OUTPUT_DIR in .env)."),
) -> None:
    """Generate a complete SEO article and save it to disk."""

    # ── Resolve tenant ──────────────────────────────────────────
    resolved_client = client_id or settings.default_client_id
    resolved_website = website_id or settings.default_website_id

    if not resolved_client or not resolved_website:
        console.print(
            "[red]Error:[/red] Client ID and Website ID are required.\n"
            "Pass [bold]--client-id[/bold] / [bold]--website-id[/bold], "
            "or set [bold]DEFAULT_CLIENT_ID[/bold] / [bold]DEFAULT_WEBSITE_ID[/bold] in .env"
        )
        raise typer.Exit(code=1)

    tenant = TenantContext(client_id=resolved_client, website_id=resolved_website)

    # ── Validate and build location ─────────────────────────────
    if bool(city) != bool(state):
        console.print("[red]Error:[/red] --city and --state must be provided together.")
        raise typer.Exit(code=1)

    location = Location(city=city, state=state) if city and state else None

    # ── Build request ───────────────────────────────────────────
    request = ArticleRequest(
        topic=topic,
        service=service,
        location=location,
        word_count=words,
        tone=tone,
        language=language,
        focus_keyword=keyword,
    )

    _execute_generation(request, tenant, output or settings.output_dir)


@app.command()
def interactive() -> None:
    """Guided wizard: answer prompts step by step to generate a blog article."""
    console.print()
    console.print(Panel(
        "[bold]SEO Agent — Interactive Mode[/bold]\n"
        "[dim]Press Enter to accept defaults. Leave optional fields blank to skip.[/dim]",
        expand=False,
    ))
    console.print()

    # ── Tenant ─────────────────────────────────────────────────
    client_id = settings.default_client_id or typer.prompt("Client ID")
    website_id = settings.default_website_id or typer.prompt("Website ID")
    tenant = TenantContext(client_id=client_id, website_id=website_id)

    # ── Topic ──────────────────────────────────────────────────
    topic = typer.prompt("Article topic")

    # ── Service ────────────────────────────────────────────────
    svc_raw = typer.prompt("Service  [optional, Enter to skip]", default="")
    service: str | None = svc_raw.strip() or None

    # ── Location ───────────────────────────────────────────────
    city_raw = typer.prompt("City     [optional, Enter to skip]", default="")
    city: str | None = city_raw.strip() or None
    state: str | None = None
    if city:
        state_raw = typer.prompt(f"State / Province for {city}")
        state = state_raw.strip() or None
        if not state:
            city = None  # location requires both city and state

    # ── Keyword ────────────────────────────────────────────────
    kw_raw = typer.prompt("Focus keyword  [optional, Enter to skip]", default="")
    keyword: str | None = kw_raw.strip() or None

    # ── Word count ─────────────────────────────────────────────
    words_raw = typer.prompt("Word count", default=str(settings.default_word_count))
    try:
        words = max(300, min(10000, int(words_raw)))
    except ValueError:
        words = settings.default_word_count

    console.print()

    location = Location(city=city, state=state) if city and state else None
    request = ArticleRequest(
        topic=topic,
        service=service,
        location=location,
        word_count=words,
        tone=settings.default_tone,
        language=ArticleLanguage.EN,
        focus_keyword=keyword,
    )

    _execute_generation(request, tenant, settings.output_dir)


# ── Private helpers ───────────────────────────────────────────────────────────

def _save_article(article: Article, base_dir: Path) -> Path:
    article_dir = (
        base_dir
        / article.tenant.client_id
        / article.tenant.website_id
        / article.seo.slug
    )
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "article.json").write_text(
        article.model_dump_json(indent=2), encoding="utf-8"
    )
    (article_dir / "article.md").write_text(
        article.content_markdown, encoding="utf-8"
    )
    return article_dir


def _save_checkpoint(content: str, checkpoint_dir: Path, topic: str) -> None:
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(topic[:50])
        checkpoint_file = checkpoint_dir / f"{safe_name}.md"
        checkpoint_file.write_text(content, encoding="utf-8")
        logger.debug("Content checkpoint saved: %s", checkpoint_file)
    except OSError:
        logger.warning("Could not save content checkpoint to %s", checkpoint_dir)


def _safe_filename(text: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower())
    return re.sub(r'-+', '-', slug).strip('-') or 'checkpoint'


def _display_result(article: Article, article_dir: Path, elapsed: float) -> None:
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=16)
    table.add_column("Value", overflow="fold")

    table.add_row("Title", article.title)
    table.add_row("Slug", article.seo.slug)
    table.add_row("Words", f"{article.word_count:,}")
    table.add_row("Reading time", f"{article.reading_time_minutes} min")
    table.add_row("Focus keyword", article.seo.focus_keyword)
    table.add_row("SEO title", article.seo.seo_title)
    if article.seo.suggested_category:
        table.add_row("Category", article.seo.suggested_category)
    table.add_row("Model", article.model_name)
    table.add_row("Generated in", f"{elapsed:.1f}s")
    table.add_row("Saved to", str(article_dir))

    console.print()
    console.print(
        Panel(table, title="[bold green]Article Generated[/bold green]", expand=False)
    )


def _execute_generation(
    request: "ArticleRequest",
    tenant: "TenantContext",
    base_dir: Path,
) -> Path:
    """
    Run Claude generation, save to disk, and display results.

    Shared by the `generate` and `interactive` commands so the generation
    logic is never duplicated. Returns the path to the saved article.json.
    """
    checkpoint_dir = base_dir / tenant.client_id / tenant.website_id / ".checkpoints"
    start = time.perf_counter()

    with console.status("[bold green]Generating content...", spinner="dots") as status:

        def on_content_ready(content: str) -> None:
            status.update("[bold green]Generating SEO metadata...")
            _save_checkpoint(content, checkpoint_dir, request.topic)

        try:
            article = article_agent.generate(
                request=request,
                tenant=tenant,
                on_content_ready=on_content_ready,
            )
        except ClaudeRateLimitError:
            console.print(
                "\n[red]Error:[/red] Rate limit exceeded after retries. "
                f"Content checkpoint saved to [dim]{checkpoint_dir}[/dim] if generation had started."
            )
            raise typer.Exit(code=1)
        except ClaudeAPIError as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    elapsed = time.perf_counter() - start

    try:
        article_dir = _save_article(article, base_dir)
    except OSError as exc:
        console.print(
            f"\n[yellow]Warning:[/yellow] Article generated but could not be saved to disk.\n"
            f"Error: {exc}\n"
            f"Content checkpoint available at: [dim]{checkpoint_dir}[/dim]"
        )
        raise typer.Exit(code=1)

    _display_result(article, article_dir, elapsed)
    return article_dir / "article.json"


# ── Publish pipeline helper ───────────────────────────────────────────────────

def _run_publish_flow(
    article_path: Path,
    article: Article,
    *,
    status: str = "publish",
    min_score: int | None = None,
    no_image: bool = False,
    no_links: bool = False,
    post_id: int | None = None,
    show_pipeline_report: bool = True,
) -> tuple[Article, _PipelineState]:
    """
    Full publish pipeline: credentials → image resolution → WordPress publish.

    Shared by `publish` and `autopublish` — the pipeline logic lives here once.
    Returns (updated_article, pipeline_state) so callers can display a custom
    summary after the standard pipeline report.
    Raises typer.Exit(code=1) on unrecoverable errors.
    """
    effective_min_score = min_score if min_score is not None else settings.seo_qa_min_score
    t_total_start = time.perf_counter()
    budget_before = budget.status()

    # Load credentials from the article's tenant context
    try:
        creds = CredentialStore(settings.credentials_dir).load(
            article.tenant.client_id, article.tenant.website_id
        )
    except CredentialNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except CredentialError as exc:
        console.print(f"[red]Error:[/red] Invalid credential file: {exc}")
        raise typer.Exit(code=1)

    # Image resolution (before WP connection — no WP calls needed)
    image_plan = None
    resolved_images: list = []
    drive_count = 0
    resolve_elapsed = 0.0

    if not no_image:
        resolve_start = time.perf_counter()
        image_plan, resolved_images, drive_count = _resolve_images(article)
        resolve_elapsed = time.perf_counter() - resolve_start

    with WordPressService(creds) as wp:
        agent = PublisherAgent(wp)
        state = _PipelineState()

        # ── Stage 1: WordPress connection + auth ─────────────────
        try:
            wp.check_connection()
            state.conn_ok = True
        except Exception as exc:
            state.conn_error = str(exc)
            console.print(f"\n[red]WordPress connection failed:[/red] {exc}")
            raise typer.Exit(code=1)

        try:
            state.auth_user = wp.check_auth()
            state.auth_ok = True
        except Exception as exc:
            state.auth_error = str(exc)
            console.print(f"\n[red]WordPress authentication failed:[/red] {exc}")
            raise typer.Exit(code=1)

        # ── Stage 2: SEO plugin detection ────────────────────────
        state.seo_plugin = agent.resolve_seo_plugin(article.publishing.seo_plugin)

        # ── Apply requested publish status ───────────────────────
        try:
            article = article.model_copy(update={
                "publishing": article.publishing.model_copy(
                    update={"status": PublishStatus(status)}
                )
            })
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid --status '{status}'. Use 'publish' or 'draft'.")
            raise typer.Exit(code=1)

        # ── Pre-flight validation ────────────────────────────────
        issues = agent.validate(article)
        if issues:
            console.print("[red]Error:[/red] Article is not ready to publish:")
            for issue in issues:
                console.print(f"  [dim]•[/dim] {issue}")
            raise typer.Exit(code=1)

        # ── Stage 3: Image state from resolution phase ───────────
        from models.image_asset import ImageSource
        from models.image_request import ImagePurpose

        if no_image:
            state.images_skip_reason = "--no-image flag"
        elif image_plan is None:
            state.images_skip_reason = "Drive and OpenAI not configured (or setup failed)"
        else:
            state.images_active    = True
            state.drive_indexed    = drive_count
            state.img_requested    = len(image_plan.requests)
            state.img_from_drive   = sum(1 for _, a in resolved_images if a.source == ImageSource.DRIVE)
            state.img_from_openai  = sum(1 for _, a in resolved_images if a.source == ImageSource.GENERATED)

        state.t_images = resolve_elapsed

        # ── Upload resolved images to WP Media Library ───────────
        uploaded_images = None
        t_upload_start = time.perf_counter()
        if resolved_images:
            uploaded_images = []
            media_svc = MediaService(wp)
            with console.status(
                f"[bold green]Uploading {len(resolved_images)} image(s) to WordPress...",
                spinner="dots",
            ):
                for req, asset in resolved_images:
                    try:
                        meta = media_svc.upload(asset)
                        uploaded_images.append((req, meta))
                    except Exception as exc:
                        state.img_errors.append(f"{req.id}: {exc}")
                        console.print(f"\n[yellow]Warning:[/yellow] Could not upload image {req.id}: {exc}")
        state.t_upload = time.perf_counter() - t_upload_start

        if uploaded_images:
            state.img_uploaded = len(uploaded_images)
            state.img_featured = any(
                req.purpose == ImagePurpose.FEATURED and bool(meta.wordpress_media_id)
                for req, meta in uploaded_images
            )

        # ── Stage 4b: SEO QA score (captured before publish raises on failure) ─
        state.qa_score = SEOQAService().analyze(article).score

        # ── Publish ──────────────────────────────────────────────
        t_publish_start = time.perf_counter()
        try:
            with console.status("[bold green]Publishing to WordPress...", spinner="dots"):
                updated = agent.publish(
                    article,
                    min_score=effective_min_score,
                    image_plan=image_plan,
                    uploaded_images=uploaded_images or None,
                    update_post_id=post_id,
                    link_enricher=None if no_links else link_enricher,
                )
        except SEOQualityError as exc:
            _display_qa_report(exc.report, exc.min_score)
            raise typer.Exit(code=1)
        except WordPressAuthError as exc:
            console.print(f"\n[red]Auth error:[/red] {exc}")
            raise typer.Exit(code=1)
        except WordPressError as exc:
            console.print(f"\n[red]WordPress error:[/red] {exc}")
            raise typer.Exit(code=1)

        state.t_publish = time.perf_counter() - t_publish_start
        state.post_id     = updated.wp_post_id
        state.post_url    = updated.wp_post_url
        state.post_status = updated.publishing.status.value
        state.post_slug   = updated.seo.slug

        # ── Post-publish: verify SEO meta acceptance ─────────────
        if state.seo_plugin in (SEOPlugin.YOAST, SEOPlugin.RANKMATH):
            if state.seo_plugin == SEOPlugin.YOAST:
                keys = list(PublisherAgent.YOAST_META_KEYS)
                if article.seo.canonical_url:
                    keys.append("_yoast_wpseo_canonical")
                if state.img_featured:
                    keys += ["_yoast_wpseo_opengraph-image", "_yoast_wpseo_twitter-image"]
            else:
                keys = list(PublisherAgent.RANKMATH_META_KEYS)
                if article.seo.canonical_url:
                    keys.append("rank_math_canonical_url")
                if state.img_featured:
                    keys += ["rank_math_og_image_url", "rank_math_twitter_image_url"]
            state.meta_check = wp.verify_meta_accepted(updated.wp_post_id, keys)

        # ── Post-publish: HTML analysis ──────────────────────────
        if updated.content_html:
            html_stats = PublisherAgent.analyze_html(updated.content_html, creds.url)
            state.html_tables         = html_stats["tables"]
            state.html_callouts       = html_stats["callouts"]
            state.html_faq            = html_stats["faq"]
            state.html_internal_links = html_stats["internal_links"]
            state.html_external_links = html_stats["external_links"]

        # ── Save updated article.json ────────────────────────────
        try:
            article_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            console.print(
                f"\n[yellow]Warning:[/yellow] Post published but could not update article.json: {exc}\n"
                f"Post ID: {updated.wp_post_id} — URL: {updated.wp_post_url}"
            )

        # ── Save image report (silent) ───────────────────────────
        if image_plan is not None:
            try:
                _save_image_report(
                    article_path, image_plan, resolved_images, uploaded_images,
                    drive_count, resolve_elapsed, article,
                )
            except Exception as exc:
                logger.warning("Could not save image resolution report: %s", exc)

        # ── Timing and costs ─────────────────────────────────────
        state.t_total = time.perf_counter() - t_total_start
        budget_after = budget.status()
        state.claude_input_tokens  = budget_after["claude"]["input_tokens"]  - budget_before["claude"]["input_tokens"]
        state.claude_output_tokens = budget_after["claude"]["output_tokens"] - budget_before["claude"]["output_tokens"]
        state.claude_cost_usd      = round(budget_after["claude"]["usd"]     - budget_before["claude"]["usd"], 6)
        state.openai_images_generated = budget_after["openai"]["images"] - budget_before["openai"]["images"]
        state.openai_cost_usd      = round(budget_after["openai"]["usd"]  - budget_before["openai"]["usd"], 6)

        if show_pipeline_report:
            _display_pipeline_report(state, article, updated)

        return updated, state


@app.command()
def publish(
    input: Path = typer.Option(..., "--input", "-i", help="Path to article.json to publish."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without creating a post."),
    min_score: int = typer.Option(settings.seo_qa_min_score, "--min-score", min=0, max=100, help="Minimum SEO quality score required to publish."),
    no_image: bool = typer.Option(False, "--no-image", help="Skip automatic image resolution and upload."),
    no_links: bool = typer.Option(False, "--no-links", help="Skip automatic internal link enrichment."),
    status: str = typer.Option("publish", "--status", help="WordPress post status: publish or draft."),
    post_id: int = typer.Option(None, "--post-id", help="Update an existing WordPress post by ID instead of creating a new one."),
) -> None:
    """Publish a generated article to WordPress. Creates a new post by default; use --post-id to update an existing one."""

    if not input.exists():
        console.print(f"[red]Error:[/red] File not found: {input}")
        raise typer.Exit(code=1)

    try:
        article = Article.model_validate_json(input.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error:[/red] Could not parse article.json: {exc}")
        raise typer.Exit(code=1)

    if dry_run:
        try:
            creds = CredentialStore(settings.credentials_dir).load(
                article.tenant.client_id, article.tenant.website_id
            )
        except CredentialNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)
        except CredentialError as exc:
            console.print(f"[red]Error:[/red] Invalid credential file: {exc}")
            raise typer.Exit(code=1)

        with WordPressService(creds) as wp:
            agent = PublisherAgent(wp)
            with console.status("[bold green]Running dry-run checks...", spinner="dots"):
                report = agent.dry_run(article, update_post_id=post_id)
        _display_dry_run(article, report, min_score)
        raise typer.Exit(code=0 if report.is_ready else 1)

    _run_publish_flow(
        input, article,
        status=status,
        min_score=min_score,
        no_image=no_image,
        no_links=no_links,
        post_id=post_id,
        show_pipeline_report=True,
    )


# ── Image resolution helper ───────────────────────────────────────────────────

def _resolve_images(article: Article) -> tuple:
    """
    Plan and resolve images for an article.

    Drive images are served from a local SQLite index (DriveImageIndex) that
    is synced from the global DRIVE_FOLDER_ID folder. The index is only
    refreshed when stale (age > DRIVE_SYNC_MAX_AGE_HOURS, default 7 days),
    so most publish runs incur zero Drive API traversal cost.

    Returns (ImagePlacementPlan | None, list[tuple[ImageRequest, ImageAsset]], int).
    The third element is the number of Drive candidates available from the index.
    Returns (None, [], 0) if neither Drive nor OpenAI is configured or on error.
    Errors are caught and reported as warnings so publishing can continue.
    """
    folder_id = settings.drive_folder_id
    drive_svc = None
    style_svc = None
    generator = None
    drive_candidates = []

    if settings.google_sa_json_path and settings.google_sa_json_path.exists() and folder_id:
        try:
            drive_svc = GoogleDriveService(settings.google_sa_json_path)
            style_svc = VisualStyleService(
                settings.profiles_dir,
                drive_svc,
                claude,
                settings.image_style_analysis_limit,
            )

            # Sync the Drive index if stale, then load all candidates locally.
            index = DriveImageIndex(settings.drive_index_path)
            if index.needs_sync(folder_id, settings.drive_sync_max_age_hours):
                with console.status("[bold green]Syncing Drive image index...", spinner="dots"):
                    sync_stats = index.sync(drive_svc, folder_id)
                console.print(
                    f"  [dim]Drive sync:[/dim] {sync_stats.images_found} images, "
                    f"{sync_stats.folders_scanned} folders "
                    f"({sync_stats.duration_seconds:.1f}s)"
                )

            drive_candidates = index.list_all()
            logger.info("%d Drive candidates loaded from index.", len(drive_candidates))

        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Drive setup failed: {exc}")

    if settings.openai_api_key:
        try:
            generator = OpenAIImageGenerator(settings.openai_api_key, budget=budget)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Image generator setup failed: {exc}")

    n_candidates = len(drive_candidates)

    if not drive_candidates and generator is None:
        return None, [], 0

    resolver = ImageResolverAgent(claude=claude, drive=drive_svc, generator=generator)

    # Phase 1: plan
    try:
        with console.status("[bold green]Planning image placement...", spinner="dots"):
            plan = resolver.plan(article)
        console.print(
            f"  [dim]Image plan:[/dim] {len(plan.requests)} image(s) — "
            + ", ".join(f"{r.id}({r.image_type.value})" for r in plan.requests)
        )
    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] Image planning failed — publishing without images: {exc}")
        return None, [], n_candidates

    # Load visual style profile (shared across all clients/websites)
    style_profile = None
    if style_svc and folder_id:
        try:
            with console.status("[bold green]Loading visual style profile...", spinner="dots"):
                style_profile = style_svc.get_profile(folder_id)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Visual style profile unavailable: {exc}")

    # Phase 2: resolve — pass pre-fetched candidates from index
    try:
        with console.status(
            f"[bold green]Resolving {len(plan.requests)} image(s)...", spinner="dots"
        ):
            resolved = resolver.resolve_all(plan, style_profile, drive_candidates=drive_candidates)
    except ImageResolverError as exc:
        console.print(f"[yellow]Warning:[/yellow] Image resolution failed — publishing without images: {exc}")
        return None, [], n_candidates
    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] Unexpected error during image resolution: {exc}")
        return None, [], n_candidates

    return plan, resolved, n_candidates


# ── Display helpers ───────────────────────────────────────────────────────────

def _display_dry_run(article: Article, report: DryRunReport, min_score: int) -> None:
    status_icon = "[green]✓[/green]" if report.is_ready else "[red]✗[/red]"
    title = f"{status_icon} Dry Run — {'Ready to publish' if report.is_ready else 'Issues found'}"

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Check", style="dim", min_width=20)
    table.add_column("Result", overflow="fold")

    table.add_row("Article", article.title[:80])
    table.add_row("Slug", article.seo.slug)
    table.add_row(
        "Connection",
        "[green]OK[/green]" if report.connection_ok else "[red]FAILED[/red]"
    )
    table.add_row(
        "Authentication",
        f"[green]OK[/green] ({report.auth_user})" if report.auth_ok else "[red]FAILED[/red]"
    )
    table.add_row(
        "Markdown → HTML",
        f"[green]OK[/green] ({report.html_chars:,} chars)" if report.html_chars else "[red]FAILED[/red]"
    )
    if report.post_action:
        action_style = "yellow" if report.post_action.startswith("UPDATE") else "green"
        table.add_row("Post action", f"[{action_style}]{report.post_action}[/{action_style}]")

    if report.category_name:
        if report.category_found:
            cat_status = f"[green]Found[/green] (ID {report.category_id})"
        elif report.category_id:
            cat_status = f"[yellow]Not found — using default ID {report.category_id}[/yellow]"
        else:
            cat_status = "[yellow]Not found — will publish without category[/yellow]"
        table.add_row(f"Category '{report.category_name}'", cat_status)

    if report.tags_existing:
        table.add_row("Tags (exist)", ", ".join(report.tags_existing))
    if report.tags_to_create:
        table.add_row("Tags (will create)", ", ".join(report.tags_to_create))

    if report.validation_issues:
        table.add_row("Validation", "[red]" + " | ".join(report.validation_issues) + "[/red]")
    else:
        table.add_row("Validation", "[green]OK[/green]")

    if report.errors:
        for err in report.errors:
            table.add_row("[red]Error[/red]", err)

    console.print()
    console.print(Panel(table, title=f"[bold]{title}[/bold]", expand=False))

    if report.qa_report is not None:
        _display_qa_report(report.qa_report, min_score)


_SEVERITY_ICON = {
    IssueSeverity.CRITICAL: "[red]✗ CRITICAL[/red]",
    IssueSeverity.ERROR:    "[red]✗ ERROR   [/red]",
    IssueSeverity.WARNING:  "[yellow]⚠ WARNING [/yellow]",
    IssueSeverity.INFO:     "[dim]ℹ INFO     [/dim]",
}


def _display_qa_report(report: SEOReport, min_score: int) -> None:
    blocked = report.summary.critical > 0 or report.score < min_score
    score_color = "green" if not blocked else "red"
    header = f"[{score_color}]{report.score}/100[/{score_color}]  (minimum: {min_score})"

    summary_parts = []
    if report.summary.critical:
        summary_parts.append(f"[red]{report.summary.critical} critical[/red]")
    if report.summary.errors:
        summary_parts.append(f"[red]{report.summary.errors} error{'s' if report.summary.errors > 1 else ''}[/red]")
    if report.summary.warnings:
        summary_parts.append(f"[yellow]{report.summary.warnings} warning{'s' if report.summary.warnings > 1 else ''}[/yellow]")
    if report.summary.info:
        summary_parts.append(f"[dim]{report.summary.info} info[/dim]")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("", min_width=14)
    table.add_column("Message", overflow="fold")
    table.add_column("Detail", style="dim", overflow="fold")

    table.add_row("Score", header, "")
    if summary_parts:
        table.add_row("", "  ·  ".join(summary_parts), "")
    table.add_row("", "", "")

    for issue in report.issues:
        table.add_row(
            _SEVERITY_ICON[issue.severity],
            issue.message,
            issue.detail or "",
        )

    qa_title = "[bold red]SEO Quality — Blocked[/bold red]" if blocked else "[bold green]SEO Quality — Passed[/bold green]"
    console.print()
    console.print(Panel(table, title=qa_title, expand=False))


def _display_image_resolution(
    plan: Any,
    resolved: list,
    uploaded: list | None,
    drive_count: int,
) -> None:
    """
    Display a summary panel of the image resolution phase.

    Shows per-image: source (Drive/OpenAI), similarity score, Drive path,
    and selection reason. Totals reflect what was actually uploaded to WordPress.

    Called only when image_plan is not None (i.e. image resolution ran).
    """
    from models.image_asset import ImageSource
    from models.image_request import ImagePurpose

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=28)
    table.add_column("Value", overflow="fold")

    table.add_row("Drive indexed images", str(drive_count))

    resolved_map = {req.id: (req, asset) for req, asset in resolved}

    for req in plan.requests:
        table.add_row("", "")

        section_label = (
            "Featured Image"
            if req.purpose == ImagePurpose.FEATURED
            else req.section_title or req.id
        )

        pair = resolved_map.get(req.id)
        if pair is None:
            table.add_row(section_label, "[red]✗ Not resolved[/red]")
            continue

        _, asset = pair
        if asset.source == ImageSource.DRIVE:
            table.add_row(section_label, "[green]✓ Found[/green]")
            if asset.similarity_score is not None:
                table.add_row("  Similarity", f"{asset.similarity_score}%")
            if asset.drive_path:
                table.add_row("  Path", asset.drive_path)
            if asset.selection_reason:
                table.add_row("  Reason", asset.selection_reason[:120])
        else:
            table.add_row(section_label, "[yellow]↑ OpenAI generated[/yellow]")
            if asset.selection_reason:
                table.add_row("  Reason", asset.selection_reason)

    table.add_row("", "")

    drive_used = sum(1 for _, a in resolved if a.source == ImageSource.DRIVE)
    openai_used = sum(1 for _, a in resolved if a.source == ImageSource.GENERATED)
    uploaded_count = len(uploaded) if uploaded else 0
    inline_count = sum(
        1 for req, _ in (uploaded or [])
        if req.purpose == ImagePurpose.INLINE
    )
    featured_ok = any(
        req.purpose == ImagePurpose.FEATURED and meta.wordpress_media_id
        for req, meta in (uploaded or [])
    )

    table.add_row("OpenAI Images generated", str(openai_used))
    table.add_row("Images uploaded to WordPress", str(uploaded_count))
    table.add_row("Images inserted into article", str(inline_count))
    table.add_row(
        "Featured Image assigned",
        "[green]Yes[/green]" if featured_ok else "[dim]No[/dim]",
    )

    console.print()
    console.print(Panel(table, title="[bold]Image Resolution[/bold]", expand=False))


def _save_image_report(
    article_path: Path,
    plan: Any,
    resolved: list,
    uploaded: list | None,
    drive_count: int,
    elapsed_seconds: float,
    article: Article,
) -> None:
    """
    Persist image_resolution_report.json alongside article.json.

    Silent by design — never raises to the caller. Failures are caught and
    logged at WARNING level so they never interrupt the publish flow.

    The report captures the full provenance of every image decision:
    why each image was selected, where it came from, what Claude Vision
    said about it, and what happened after upload. It is not read by
    any other part of the pipeline — it exists purely for audit,
    debugging, and future algorithm improvements.
    """
    import json
    from datetime import datetime, timezone
    from models.image_asset import ImageSource
    from models.image_request import ImagePurpose

    resolved_map = {req.id: (req, asset) for req, asset in resolved}
    uploaded_map = {req.id: meta for req, meta in (uploaded or [])}

    # ── Per-image records ─────────────────────────────────────────────────────
    images: list[dict] = []
    for position, req in enumerate(plan.requests):
        pair = resolved_map.get(req.id)
        meta = uploaded_map.get(req.id)

        base: dict = {
            "id":            req.id,
            "purpose":       req.purpose.value,
            "image_type":    req.image_type.value,
            "section_title": req.section_title,
            "subject":       req.subject,
            "alt_text":      req.alt_text,
            "is_featured":   req.purpose == ImagePurpose.FEATURED,
            "position":      position,
        }

        if pair is None:
            images.append({**base, "error": "not_resolved"})
            continue

        _, asset = pair
        is_drive = asset.source == ImageSource.DRIVE

        images.append({
            **base,
            "source":                     asset.source.value,
            "drive_file_id":              asset.source_detail if is_drive else None,
            "drive_path":                 asset.drive_path,
            "similarity_score":           asset.similarity_score,
            "drive_candidates_evaluated": asset.drive_candidates_evaluated,
            "selection_reason":           asset.selection_reason,
            "vision_reasoning":           asset.vision_reasoning,
            "openai_prompt":              asset.source_detail if not is_drive else None,
            "wordpress_media_id":         meta.wordpress_media_id if meta else None,
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    drive_assets = [a for _, a in resolved if a.source == ImageSource.DRIVE]
    ai_assets    = [a for _, a in resolved if a.source == ImageSource.GENERATED]

    similarity_scores = [
        a.similarity_score for a in drive_assets if a.similarity_score is not None
    ]
    avg_similarity = (
        round(sum(similarity_scores) / len(similarity_scores), 1)
        if similarity_scores else None
    )

    featured_ok = any(
        req.purpose == ImagePurpose.FEATURED and bool(meta.wordpress_media_id)
        for req, meta in (uploaded or [])
    )

    # ── Folder usage (Drive images only) ─────────────────────────────────────
    folder_usage: dict[str, int] = {}
    for _, asset in resolved:
        if asset.drive_path:
            parts = asset.drive_path.split("/")
            folder = "/".join(parts[:-1])   # strip filename
            if folder:
                folder_usage[folder] = folder_usage.get(folder, 0) + 1

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "article_title":   article.title,
        "article_slug":    article.seo.slug,
        "article_id":      str(article.id),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "summary": {
            "images_requested":           len(plan.requests),
            "images_resolved":            len(resolved),
            "drive_selected":             len(drive_assets),
            "openai_generated":           len(ai_assets),
            "drive_candidates_available": drive_count,
            "avg_similarity_score":       avg_similarity,
            "featured_assigned":          featured_ok,
            "openai_cost_estimate_usd":   round(len(ai_assets) * 0.12, 4),
        },
        "images":       images,
        "folder_usage": dict(
            sorted(folder_usage.items(), key=lambda x: x[1], reverse=True)
        ),
    }

    report_path = article_path.parent / "image_resolution_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.debug("Image resolution report saved: %s", report_path)


def _display_pipeline_report(
    state: _PipelineState,
    article: Article,
    updated: Article,
) -> None:
    """Full pipeline observability report shown after every successful publish."""
    from rich.rule import Rule

    def _ok(label: str) -> None:
        console.print(f"   [green]✓[/green] {label}")

    def _fail(label: str, detail: str = "") -> None:
        suffix = f"  [dim red]— {detail}[/dim red]" if detail else ""
        console.print(f"   [red]✗[/red] {label}{suffix}")

    def _section(title: str) -> None:
        console.print(f"[bold]{title}[/bold]")

    def _row(label: str, value: str, width: int = 28) -> None:
        console.print(f"   {label:<{width}}{value}")

    console.print()
    console.print(Rule("[bold]PUBLISH PIPELINE[/bold]"))
    console.print()

    # ── 1. WordPress ──────────────────────────────────────────────
    _section("1. WordPress")
    if state.conn_ok:
        _ok("Connected")
    else:
        _fail("Connection failed", state.conn_error or "")
    if state.auth_ok:
        _ok(f"Authenticated  [dim]({state.auth_user})[/dim]")
    else:
        _fail("Authentication failed", state.auth_error or "")
    console.print()

    # ── 2. SEO Plugin ─────────────────────────────────────────────
    _section("2. SEO Plugin")
    plugin_label = {
        SEOPlugin.YOAST:    "[green]Yoast SEO[/green]",
        SEOPlugin.RANKMATH: "[green]Rank Math[/green]",
        SEOPlugin.NONE:     "[dim]None — metadata not sent[/dim]",
    }.get(state.seo_plugin, str(state.seo_plugin.value))
    _row("Detected:", plugin_label)

    if state.seo_plugin in (SEOPlugin.YOAST, SEOPlugin.RANKMATH):
        labels = (
            PublisherAgent.YOAST_META_LABELS
            if state.seo_plugin == SEOPlugin.YOAST
            else PublisherAgent.RANKMATH_META_LABELS
        )
        console.print("   SEO metadata:")
        for key, verdict in state.meta_check.items():
            label = labels.get(key, key)
            if verdict == "accepted":
                console.print(f"     [green]✓[/green] {label}  [dim]accepted[/dim]")
            elif verdict == "empty":
                console.print(f"     [yellow]⚠[/yellow] {label}  [yellow]registered but empty[/yellow]")
            elif verdict == "not_registered":
                console.print(
                    f"     [red]✗[/red] {label}  "
                    f"[red]not saved — show_in_rest not registered in WordPress[/red]"
                )
            else:
                console.print(f"     [dim]?[/dim] {label}  [dim]could not verify[/dim]")

        if state.meta_check:
            accepted = sum(1 for v in state.meta_check.values() if v == "accepted")
            total    = len(state.meta_check)
            if accepted == total:
                _row("WordPress response:", "[green]All accepted[/green]")
            elif accepted > 0:
                _row("WordPress response:", f"[yellow]Partial ({accepted}/{total} fields accepted)[/yellow]")
            else:
                _row(
                    "WordPress response:",
                    "[red]Not saved — register meta keys with show_in_rest=True[/red]",
                )
    console.print()

    # ── 3. Image Resolution ───────────────────────────────────────
    _section("3. Image Resolution")
    if not state.images_active:
        reason = state.images_skip_reason or "skipped"
        _row("Status:", f"[dim]Skipped — {reason}[/dim]")
    else:
        _row("Drive images indexed:", str(state.drive_indexed))
        _row("Images requested:",     str(state.img_requested))
        _row("Resolved from Drive:",  str(state.img_from_drive))
        _row("Generated by OpenAI:",  str(state.img_from_openai))
        _row("Uploaded to WordPress:", str(state.img_uploaded))
        _row(
            "Featured Image:",
            "[green]Assigned[/green]" if state.img_featured else "[dim]Not assigned[/dim]",
        )
        for err in state.img_errors:
            console.print(f"   [red]✗ Upload error:[/red] {err}")
    console.print()

    # ── 4. HTML Content ───────────────────────────────────────────
    _section("4. HTML Content")
    _row("Tables:",        f"[green]✓[/green] ({state.html_tables})" if state.html_tables else f"[dim]none[/dim]")
    _row("Callouts:",      f"[green]✓[/green] ({state.html_callouts})" if state.html_callouts else f"[dim]none[/dim]")
    _row("FAQ section:",   "[green]✓[/green]" if state.html_faq else "[dim]not found[/dim]")
    _row("Internal links:", str(state.html_internal_links))
    _row("External links:", str(state.html_external_links))
    console.print()

    # ── 5. Published ──────────────────────────────────────────────
    _section("5. Published")
    _row("Status:", f"[green]{state.post_status}[/green]")
    _row("Post ID:", str(state.post_id))
    _row("Slug:",    state.post_slug)
    if state.post_url:
        _row("URL:", state.post_url)
    console.print()

    # ── 6. Timing ─────────────────────────────────────────────────
    _section("6. Timing")
    def _fmt_s(s: float) -> str:
        return f"{s:.1f}s" if s >= 0.1 else "[dim]—[/dim]"
    _row("Image resolution:", _fmt_s(state.t_images))
    _row("Upload:",           _fmt_s(state.t_upload))
    _row("WordPress publish:", _fmt_s(state.t_publish))
    _row("Total pipeline:",   f"[bold]{_fmt_s(state.t_total)}[/bold]")
    console.print()

    # ── 7. Costs ──────────────────────────────────────────────────
    _section("7. Costs")
    _row("Claude input tokens:",  f"{state.claude_input_tokens:,}")
    _row("Claude output tokens:", f"{state.claude_output_tokens:,}")
    _row("Claude cost:",          f"[cyan]${state.claude_cost_usd:.4f}[/cyan]")
    if state.openai_images_generated:
        _row("OpenAI images:",    str(state.openai_images_generated))
        _row("OpenAI cost:",      f"[cyan]${state.openai_cost_usd:.4f}[/cyan]")
    total_cost = state.claude_cost_usd + state.openai_cost_usd
    _row("Total run cost:",       f"[bold cyan]${total_cost:.4f}[/bold cyan]")
    console.print()
    console.print(Rule())


# ── Validate sub-commands ─────────────────────────────────────────────────────

validate_app = typer.Typer(
    name="validate",
    help="Validate individual integrations in isolation before running the full pipeline.",
    no_args_is_help=True,
)
app.add_typer(validate_app, name="validate")


@validate_app.command("claude")
def validate_claude(
    topic: str = typer.Option(
        "Reparación de puertas de garaje en Miami",
        "--topic", "-t",
        help="Topic for the test article.",
    ),
    words: int = typer.Option(400, "--words", "-w", min=300, max=800),
    language: ArticleLanguage = typer.Option(ArticleLanguage.EN, "--language", "-l"),
    client_id: str = typer.Option("demo", "--client-id"),
    website_id: str = typer.Option("demo", "--website-id"),
    save: bool = typer.Option(False, "--save", help="Save article.json and article.md to disk."),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Validate Claude: generate a minimal article and verify JSON serialization."""
    request = ArticleRequest(
        topic=topic,
        language=language,
        word_count=words,
        tone=settings.default_tone,
    )
    tenant = TenantContext(client_id=client_id, website_id=website_id)

    console.print(f"[dim]Topic:[/dim] {topic}")
    console.print(f"[dim]Language:[/dim] {language.value}  [dim]Words:[/dim] {words}\n")

    t0 = time.perf_counter()
    try:
        with console.status("[bold blue]Generating article...", spinner="dots"):
            article = article_agent.generate(request=request, tenant=tenant)
    except ClaudeAPIError as exc:
        console.print(f"[red]✗ Claude API error:[/red] {exc}")
        raise typer.Exit(code=1)

    elapsed = time.perf_counter() - t0

    try:
        raw_json = article.model_dump_json(indent=2)
        Article.model_validate_json(raw_json)
        ser_status = f"[green]✓[/green] {len(raw_json.encode()):,} bytes"
    except Exception as exc:
        ser_status = f"[red]✗ {exc}[/red]"

    qa = SEOQAService().analyze(article)
    score_color = "green" if qa.score >= 70 else "yellow" if qa.score >= 50 else "red"

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=18)
    table.add_column("Value", overflow="fold")
    table.add_row("Status", "[green]✓ Generated[/green]")
    table.add_row("Time", f"{elapsed:.1f}s")
    table.add_row("Title", article.title)
    table.add_row("Words", f"{article.word_count:,}")
    table.add_row("Reading time", f"{article.reading_time_minutes} min")
    table.add_row("Focus keyword", article.seo.focus_keyword)
    table.add_row("SEO title", article.seo.seo_title)
    table.add_row("Slug", article.seo.slug)
    table.add_row("Meta description", article.seo.meta_description[:80] + "…")
    table.add_row(
        "SEO score",
        f"[{score_color}]{qa.score}/100[/{score_color}]"
        f"  {qa.summary.critical} critical · {qa.summary.errors} errors · {qa.summary.warnings} warnings",
    )
    table.add_row("JSON serialization", ser_status)
    table.add_row("Model", article.model_name)

    console.print(Panel(table, title="[bold green]✓ Claude Validated[/bold green]", expand=False))

    console.print("\n[dim]Markdown preview (first 400 chars):[/dim]")
    console.print(article.content_markdown[:400])

    if save:
        out_dir = (output or settings.output_dir) / "validate" / client_id / website_id / article.seo.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "article.json").write_text(raw_json, encoding="utf-8")
        (out_dir / "article.md").write_text(article.content_markdown, encoding="utf-8")
        console.print(f"\n[dim]Saved to:[/dim] {out_dir}")


@validate_app.command("wordpress")
def validate_wordpress(
    client_id: str = typer.Option(..., "--client-id"),
    website_id: str = typer.Option(..., "--website-id"),
    input: Path | None = typer.Option(None, "--input", "-i", help="Existing article.json to test with (optional)."),
) -> None:
    """Validate WordPress: authenticate, check taxonomy, dry-run. Does not publish."""
    try:
        creds = CredentialStore(settings.credentials_dir).load(client_id, website_id)
    except CredentialNotFoundError as exc:
        console.print(f"[red]✗ Credentials not found:[/red] {exc}")
        raise typer.Exit(code=1)
    except CredentialError as exc:
        console.print(f"[red]✗ Invalid credentials:[/red] {exc}")
        raise typer.Exit(code=1)

    if input is not None:
        if not input.exists():
            console.print(f"[red]✗ File not found:[/red] {input}")
            raise typer.Exit(code=1)
        try:
            article = Article.model_validate_json(input.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]✗ Cannot parse article.json:[/red] {exc}")
            raise typer.Exit(code=1)
        console.print(f"[dim]Using article:[/dim] {article.title}\n")
    else:
        article = Article(
            title="[SEO Agent] WordPress Validation Test",
            content_markdown=(
                "## Acerca de esta prueba\n\n"
                "Este es un artículo sintético generado por `seo-agent validate wordpress`.\n"
                "Se usa únicamente para verificar la conectividad y los permisos de WordPress.\n"
                "**No se publicará.**\n\n"
                + ("Párrafo de prueba. " * 30 + "\n\n") * 6
            ),
            tenant=TenantContext(client_id=client_id, website_id=website_id),
            request=ArticleRequest(
                topic="WordPress connection validation",
                language=ArticleLanguage.ES,
            ),
            seo=SEOMetadata(
                seo_title="SEO Agent — WordPress Validation Test",
                meta_description="Artículo sintético para validar la integración con WordPress antes de producción.",
                slug="seo-agent-validate-wordpress",
                focus_keyword="seo agent validación",
                suggested_category="Blog",
                suggested_tags=["test", "seo-agent"],
            ),
            model_name="validate",
        )
        console.print("[dim]Sin --input: usando artículo sintético (no se publicará).[/dim]\n")

    with WordPressService(creds) as wp:
        agent = PublisherAgent(wp)
        with console.status("[bold blue]Conectando a WordPress...", spinner="dots"):
            report = agent.dry_run(article)

    _display_dry_run(article, report, min_score=0)

    if report.connection_ok and report.auth_ok:
        console.print("\n[green]✓ WordPress validado correctamente.[/green]")
    else:
        console.print("\n[red]✗ Fallo en la conexión o autenticación de WordPress.[/red]")
        raise typer.Exit(code=1)


@validate_app.command("drive")
def validate_drive(
    sample: int = typer.Option(10, "--sample", "-n", help="Number of image paths to show as examples."),
) -> None:
    """Validate Google Drive: sync the image index and report what was found."""
    if not settings.google_sa_json_path:
        console.print(
            "[red]✗ GOOGLE_SA_JSON_PATH not set.[/red]\n"
            "Add it to .env:  [bold]GOOGLE_SA_JSON_PATH=/path/to/service-account.json[/bold]"
        )
        raise typer.Exit(code=1)

    if not settings.google_sa_json_path.exists():
        console.print(f"[red]✗ Service account file not found:[/red] {settings.google_sa_json_path}")
        raise typer.Exit(code=1)

    if not settings.drive_folder_id:
        console.print(
            "[red]✗ DRIVE_FOLDER_ID not set.[/red]\n"
            "Add it to .env:  [bold]DRIVE_FOLDER_ID=1ABC...[/bold]"
        )
        raise typer.Exit(code=1)

    console.print(f"[dim]Service account:[/dim] {settings.google_sa_json_path}")
    console.print(f"[dim]Folder ID:[/dim]       {settings.drive_folder_id}")
    console.print(f"[dim]Index path:[/dim]      {settings.drive_index_path}\n")

    try:
        drive = GoogleDriveService(settings.google_sa_json_path)
        index = DriveImageIndex(settings.drive_index_path)

        with console.status("[bold blue]Syncing Drive image index (full traversal)...", spinner="dots"):
            stats = index.sync(drive, settings.drive_folder_id)

    except Exception as exc:
        console.print(f"[red]✗ Drive sync failed:[/red] {exc}")
        raise typer.Exit(code=1)

    images = index.list_all()

    # Thumbnail probe
    thumb_status = "[dim]No images with thumbnails[/dim]"
    if images:
        probe = next((img for img in images if img.thumbnail_link), None)
        if probe:
            try:
                with console.status(f"[bold blue]Testing thumbnail download...", spinner="dots"):
                    resp = httpx.get(probe.thumbnail_link, follow_redirects=True, timeout=15)
                    resp.raise_for_status()
                thumb_status = f"[green]✓ OK[/green] ({len(resp.content):,} bytes — {probe.name[:40]})"
            except Exception as exc:
                thumb_status = f"[yellow]⚠ Failed:[/yellow] {exc}"
        else:
            thumb_status = "[yellow]⚠ No thumbnail links in index[/yellow]"

    # Summary table
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=22)
    table.add_column("Value", overflow="fold")

    table.add_row("Sync type", "Full sync")
    table.add_row("Duration", f"{stats.duration_seconds:.1f}s")
    table.add_row("Folders scanned", str(stats.folders_scanned))
    table.add_row("Images indexed", str(stats.images_found))
    table.add_row(
        "Changes",
        f"+{stats.images_added} added  ~{stats.images_updated} updated  -{stats.images_removed} removed"
        if stats.images_added or stats.images_updated or stats.images_removed
        else "No changes",
    )
    table.add_row("Ignored files", str(stats.ignored_files))
    table.add_row("Thumbnail test", thumb_status)

    console.print(Panel(table, title="[bold green]Google Drive Validated[/bold green]", expand=False))

    if not images:
        console.print("\n[yellow]⚠ No images found. Check that the service account has Viewer access to the folder.[/yellow]")
        raise typer.Exit(code=1)

    # Sample paths
    console.print(f"\n[dim]Sample image paths (first {min(sample, len(images))}):[/dim]")
    for img in images[:sample]:
        folder = img.folder_path.strip("/")
        display_path = f"{folder}/{img.name}" if folder else img.name
        size_str = f"  [dim]{img.size // 1024:,} KB[/dim]" if img.size else ""
        console.print(f"  {display_path}{size_str}")

    console.print(f"\n[green]✓ Google Drive validado — {stats.images_found} imágenes en {stats.folders_scanned} carpetas.[/green]")


@validate_app.command("images")
def validate_images(
    prompt: str = typer.Option(
        "A professional garage door installation in a suburban home, natural daylight photography. No text.",
        "--prompt", "-p",
    ),
    output: Path = typer.Option(Path("validate-image.png"), "--output", "-o", help="Path to save the generated image."),
) -> None:
    """Validate AI image generation: generate one test image and save it."""
    if not settings.openai_api_key:
        console.print(
            "[red]✗ OPENAI_API_KEY not set.[/red]\n"
            "Add it to .env:  [bold]OPENAI_API_KEY=sk-...[/bold]"
        )
        raise typer.Exit(code=1)

    from services.image_generators import ImageGenerationRequest

    console.print(f"[dim]Prompt:[/dim] {prompt[:100]}\n")

    t0 = time.perf_counter()
    try:
        generator = OpenAIImageGenerator(settings.openai_api_key)
        with console.status("[bold blue]Generating image via DALL-E 3...", spinner="dots"):
            asset = generator.generate(ImageGenerationRequest(
                prompt=prompt,
                alt_text="Validation test image",
                size="1792x1024",
            ))
    except Exception as exc:
        console.print(f"[red]✗ Image generation failed:[/red] {exc}")
        raise typer.Exit(code=1)

    elapsed = time.perf_counter() - t0

    out_path = output if output.is_absolute() else Path.cwd() / output
    try:
        out_path.write_bytes(asset.data)
    except OSError as exc:
        console.print(f"[red]✗ Could not save image:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=18)
    table.add_column("Value", overflow="fold")
    table.add_row("Status", "[green]✓ Generated[/green]")
    table.add_row("Time", f"{elapsed:.1f}s")
    table.add_row("File size", f"{len(asset.data):,} bytes ({len(asset.data) // 1024:,} KB)")
    table.add_row("Format", asset.mime_type)
    table.add_row("Saved to", str(out_path))
    if asset.source_detail:
        table.add_row("Revised prompt", asset.source_detail[:100] + ("…" if len(asset.source_detail) > 100 else ""))

    console.print(Panel(table, title="[bold green]✓ Image Generation Validated[/bold green]", expand=False))


# ── Topic suggestion helper ───────────────────────────────────────────────────

def _suggest_topics(
    client_id: str,
    website_id: str,
    *,
    service: str | None = None,
    city: str | None = None,
    language: ArticleLanguage = ArticleLanguage.EN,
    n: int = 10,
    base_dir: Path | None = None,
) -> list[str]:
    """
    Generate blog topic suggestions for a client/website, excluding already-published titles.

    Shared by `suggest` (interactive display) and `autopublish` (automated pick).
    Returns a list of topic strings parsed from the Claude response.
    Raises ClaudeAPIError on generation failure.
    """
    base_dir = base_dir or settings.output_dir
    client_dir = base_dir / client_id / website_id

    existing_titles: list[str] = []
    if client_dir.exists():
        for article_file in client_dir.rglob("article.json"):
            try:
                data = __import__("json").loads(article_file.read_text(encoding="utf-8"))
                title = data.get("title") or data.get("seo", {}).get("meta_title") or ""
                if title:
                    existing_titles.append(title)
            except Exception:
                pass

    context_lines: list[str] = []
    if service:
        context_lines.append(f"Service/niche: {service}")
    if city:
        context_lines.append(f"Target location: {city}")
    context_lines.append(f"Language: {language.value}")
    context_lines.append(f"Request: generate exactly {n} unique blog topic ideas for a local service business.")
    context_lines.append("Each idea should target a specific customer question, pain point, or local search intent.")
    context_lines.append("Ideas must be practical, SEO-friendly, and genuinely useful to readers.")
    context_lines.append(
        "All ideas must be evergreen — do NOT include years (2024, 2025, 2026, etc.) in any title. "
        "Avoid patterns like 'Best X for 2025' or 'Top trends in 2026'. "
        "Each idea should remain relevant and accurate for years after publication."
    )
    if existing_titles:
        context_lines.append(
            f"\nAlready published ({len(existing_titles)} article(s)) — do NOT repeat or closely paraphrase these:\n"
            + "\n".join(f"  - {t}" for t in existing_titles)
        )
    context_lines.append(
        "\nReturn ONLY a numbered list of topic ideas, one per line. "
        "No preamble, no explanations, no markdown formatting beyond the numbers."
    )

    system = (
        "You are an expert SEO content strategist for local service businesses. "
        "Generate fresh, unique blog topic ideas that have strong local search intent "
        "and have not been covered in the existing articles listed. "
        "All topics must be evergreen — avoid year-specific titles or angles. "
        "Write ideas whose relevance does not expire."
    )

    raw = claude.generate(system, [{"role": "user", "content": "\n".join(context_lines)}], thinking=False)
    return re.findall(r'^\d+\.\s+(.+)$', raw.strip(), flags=re.MULTILINE)


@app.command()
def suggest(
    client_id: str = typer.Option(None, "--client-id", help="Client identifier (overrides DEFAULT_CLIENT_ID in .env)."),
    website_id: str = typer.Option(None, "--website-id", help="Website identifier (overrides DEFAULT_WEBSITE_ID in .env)."),
    n: int = typer.Option(10, "--n", "-n", min=1, max=50, help="Number of topic ideas to generate."),
    service: str = typer.Option(None, "--service", "-s", help="Service or product focus (e.g. 'garage door repair')."),
    city: str = typer.Option(None, "--city", help="Target city for local SEO."),
    language: ArticleLanguage = typer.Option(ArticleLanguage.EN, "--language", "-l", help="Language for topic ideas (default: English)."),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory (overrides OUTPUT_DIR in .env)."),
) -> None:
    """Suggest unique blog topic ideas, avoiding topics already published for this client."""
    resolved_client = client_id or settings.default_client_id
    resolved_website = website_id or settings.default_website_id

    if not resolved_client or not resolved_website:
        console.print(
            "[red]Error:[/red] Client ID and Website ID are required.\n"
            "Pass [bold]--client-id[/bold] / [bold]--website-id[/bold], "
            "or set [bold]DEFAULT_CLIENT_ID[/bold] / [bold]DEFAULT_WEBSITE_ID[/bold] in .env"
        )
        raise typer.Exit(code=1)

    base_dir = output or settings.output_dir

    with console.status(f"[bold green]Generating {n} topic ideas...", spinner="dots"):
        try:
            topics = _suggest_topics(
                resolved_client, resolved_website,
                service=service,
                city=city,
                language=language,
                n=n,
                base_dir=base_dir,
            )
        except ClaudeAPIError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    if not topics:
        console.print("[yellow]Warning:[/yellow] No topics were parsed from the response.")
        return

    client_dir = base_dir / resolved_client / resolved_website
    existing_count = sum(1 for _ in client_dir.rglob("article.json")) if client_dir.exists() else 0
    already_label = f"  [dim]({existing_count} existing article(s) excluded)[/dim]" if existing_count else ""
    title_line = f"[bold green]{n} Blog Topic Ideas — {resolved_website}[/bold green]{already_label}"
    raw_display = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(topics))

    console.print()
    console.print(Panel(raw_display, title=title_line, expand=False))

    console.print()
    pick = typer.prompt(
        "Enter a number to generate that article now, or press Enter to exit",
        default="",
    )
    if not pick.strip():
        return

    try:
        idx = int(pick.strip()) - 1
    except ValueError:
        console.print("[yellow]Invalid input.[/yellow]")
        return

    if not (0 <= idx < len(topics)):
        console.print("[yellow]Number out of range.[/yellow]")
        return

    selected_topic = topics[idx]
    console.print(f"\n[dim]Selected:[/dim] {selected_topic}\n")

    gen_city: str | None = city
    gen_state: str | None = None
    if gen_city:
        state_raw = typer.prompt(f"State / Province for {gen_city} (Enter to skip)", default="")
        gen_state = state_raw.strip() or None
        if not gen_state:
            gen_city = None

    kw_raw = typer.prompt("Focus keyword  [optional, Enter to skip]", default="")
    gen_keyword: str | None = kw_raw.strip() or None

    tenant = TenantContext(client_id=resolved_client, website_id=resolved_website)
    gen_location = Location(city=gen_city, state=gen_state) if gen_city and gen_state else None
    gen_request = ArticleRequest(
        topic=selected_topic,
        service=service,
        location=gen_location,
        word_count=settings.default_word_count,
        tone=settings.default_tone,
        language=ArticleLanguage.EN,
        focus_keyword=gen_keyword,
    )
    _execute_generation(gen_request, tenant, base_dir)


# ── Autopublish summary ───────────────────────────────────────────────────────

def _display_autopublish_summary(
    topic: str,
    article: "Article",
    updated: "Article",
    state: "_PipelineState",
) -> None:
    from rich.table import Table

    lines: list[str] = []

    lines.append(f"[bold]Topic[/bold]     {topic}")
    lines.append(f"[bold]Article[/bold]   [green]✓ Generated[/green]  {len(article.content_markdown.split())} words")
    kw = article.seo.focus_keyword or ""
    if kw:
        lines.append(f"[bold]Keyword[/bold]   {kw}")

    qa_label = f"{state.qa_score}/100"
    qa_color = "green" if state.qa_score >= 80 else ("yellow" if state.qa_score >= 60 else "red")
    lines.append(f"[bold]SEO QA[/bold]    [{qa_color}]{qa_label}[/{qa_color}]")

    img_parts: list[str] = []
    if state.img_from_drive:
        img_parts.append(f"{state.img_from_drive} from Drive")
    if state.img_from_openai:
        img_parts.append(f"{state.img_from_openai} from OpenAI")
    if not img_parts:
        img_parts.append("none")
    lines.append(f"[bold]Images[/bold]    {', '.join(img_parts)}")

    wp_status = state.post_status or updated.status.value
    post_label = f"[green]✓ {wp_status}[/green]" if state.post_id else "[red]✗ Failed[/red]"
    lines.append(f"[bold]WordPress[/bold] {post_label}")
    if state.post_id:
        lines.append(f"[bold]Post ID[/bold]   {state.post_id}")
    if state.post_url:
        lines.append(f"[bold]URL[/bold]       {state.post_url}")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="[bold green]Auto Publish — Complete[/bold green]",
        expand=False,
    ))


@app.command()
def autopublish(
    client_id: str | None = typer.Option(None, "--client-id", help="Client identifier (overrides DEFAULT_CLIENT_ID in .env)."),
    website_id: str | None = typer.Option(None, "--website-id", help="Website identifier (overrides DEFAULT_WEBSITE_ID in .env)."),
    service: str | None = typer.Option(None, "--service", "-s", help="Service or product niche (e.g. 'garage door repair')."),
    city: str | None = typer.Option(None, "--city", help="Target city for local SEO."),
    state_opt: str | None = typer.Option(None, "--state", help="Target state (required with --city)."),
    keyword: str | None = typer.Option(None, "--keyword", "-k", help="Primary keyword hint for generation."),
    words: int = typer.Option(settings.default_word_count, "--words", "-w", min=300, max=10000, help="Target word count."),
    language: ArticleLanguage = typer.Option(ArticleLanguage.EN, "--language", "-l", help="Article language (always English)."),
    status: str = typer.Option("draft", "--status", help="WordPress post status: draft or publish."),
    min_score: int = typer.Option(settings.seo_qa_min_score, "--min-score", min=0, max=100, help="Minimum SEO quality score."),
    no_image: bool = typer.Option(False, "--no-image", help="Skip image resolution and upload."),
    no_links: bool = typer.Option(False, "--no-links", help="Skip internal link enrichment."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory (overrides OUTPUT_DIR in .env)."),
    suggest_n: int = typer.Option(10, "--suggest-n", help="How many topic ideas to generate before picking the first."),
) -> None:
    """Full auto-pilot: suggest a topic → generate article → publish to WordPress."""

    resolved_client = client_id or settings.default_client_id
    resolved_website = website_id or settings.default_website_id

    if not resolved_client or not resolved_website:
        console.print(
            "[red]Error:[/red] Client ID and Website ID are required.\n"
            "Pass [bold]--client-id[/bold] / [bold]--website-id[/bold], "
            "or set [bold]DEFAULT_CLIENT_ID[/bold] / [bold]DEFAULT_WEBSITE_ID[/bold] in .env"
        )
        raise typer.Exit(code=1)

    if bool(city) != bool(state_opt):
        console.print("[red]Error:[/red] --city and --state must be provided together.")
        raise typer.Exit(code=1)

    base_dir = output or settings.output_dir
    tenant = TenantContext(client_id=resolved_client, website_id=resolved_website)
    location = Location(city=city, state=state_opt) if city and state_opt else None

    console.print()
    console.print(Panel(
        f"[bold]SEO Agent — Auto Publish[/bold]\n"
        f"[dim]{resolved_client} / {resolved_website}[/dim]",
        expand=False,
    ))
    console.print()

    # ── Step 1: Suggest topics ────────────────────────────────────
    with console.status(f"[bold green]Generating {suggest_n} topic ideas...", spinner="dots"):
        try:
            topics = _suggest_topics(
                resolved_client, resolved_website,
                service=service,
                city=city,
                language=language,
                n=suggest_n,
                base_dir=base_dir,
            )
        except ClaudeAPIError as exc:
            console.print(f"[red]Error:[/red] Topic suggestion failed: {exc}")
            raise typer.Exit(code=1)

    if not topics:
        console.print("[red]Error:[/red] No topics were generated. Try again.")
        raise typer.Exit(code=1)

    selected_topic = topics[0]
    console.print(f"[dim]Generated {len(topics)} ideas. Selected:[/dim]")
    console.print(f"[bold cyan]  {selected_topic}[/bold cyan]")
    console.print()

    # ── Step 2: Generate article ──────────────────────────────────
    request = ArticleRequest(
        topic=selected_topic,
        service=service,
        location=location,
        word_count=words,
        tone=settings.default_tone,
        language=language,
        focus_keyword=keyword,
    )

    article_path = _execute_generation(request, tenant, base_dir)
    article = Article.model_validate_json(article_path.read_text(encoding="utf-8"))

    # ── Steps 3–6: SEO QA + Image resolution + Publish ───────────
    updated, state = _run_publish_flow(
        article_path, article,
        status=status,
        min_score=min_score,
        no_image=no_image,
        no_links=no_links,
        show_pipeline_report=True,
    )

    # ── Summary ───────────────────────────────────────────────────
    _display_autopublish_summary(selected_topic, article, updated, state)


if __name__ == "__main__":
    app()

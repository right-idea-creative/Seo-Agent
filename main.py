import csv
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

from agents import ArticleValidationError, DryRunReport, DualQAAgent, DualQAFailedError, ImageResolverAgent, ImageResolverError, PublisherAgent, SEOQualityError, article_agent, link_enricher
from services.budget_service import BudgetExceededError
from services.draft_reuse_service import DraftMatch, DraftReuseService
from services.business_context_resolver import BusinessContextResolver
from config import settings
from models import ArticleRequest, Location, TenantContext
from models.site_profile import SiteProfile
from models.article import Article, SEOMetadata
from services.seo_qa_service import SEOQAService
from models.enums import ArticleLanguage, ArticleTone, PublishStatus, SEOPlugin
from services import ClaudeAPIError, ClaudeRateLimitError, LLMAllProvidersFailedError, MediaService, SiteProfileService, WordPressService
from services import budget, GoogleDriveService, OpenAIImageGenerator, OpenAIReviewService
from services import EditorialHistoryService
from services import DriveImageIndex
from services import claude
from services import ContentSanitizationService, SanitizationResult
from services import PublicationReadinessService, ReadinessResult
from services import PublicationCertificationService, CertificationReport
from services.credential_store import CredentialError, CredentialNotFoundError, CredentialStore, WordPressCredentials
from services.wordpress_service import SiteValidationResult, WordPressAuthError, WordPressError
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
    drive_semantic_candidates: int = 0   # Drive photos with keyword overlap > 0 (before Vision)
    img_requested: int = 0
    img_from_drive: int = 0         # P1: original Drive photos published as-is
    img_from_edited: int = 0        # P2: original Drive photos with minimal preservation edit
    img_uploaded: int = 0
    img_featured: bool = False
    img_errors: list[str] = _dc_field(default_factory=list)
    ai_reasons: list[str] = _dc_field(default_factory=list)  # edit reasons for EDITED images
    openai_budget_total: int = 0
    openai_budget_remaining: int = 0
    edited_photos: list = _dc_field(default_factory=list)    # per-edit audit details

    # Stage 4: HTML
    html_tables: int = 0
    html_callouts: int = 0
    html_faq: bool = False
    html_internal_links: int = 0
    html_external_links: int = 0

    # Stage 4: Dual QA
    dual_qa_enabled: bool = False
    dual_qa_passed: bool = False
    dual_qa_iterations: int = 0
    dual_qa_seo_score: int = 0
    dual_qa_editorial_score: int = 0
    dual_qa_writing_score: int = 0
    dual_qa_authenticity_score: int = 0
    dual_qa_combined_score: float = 0.0
    dual_qa_images_reviewed: int = 0
    dual_qa_images_passed: int = 0
    dual_qa_images_failed: int = 0
    dual_qa_rejection_reasons: list[str] = _dc_field(default_factory=list)
    dual_qa_image_results: list = _dc_field(default_factory=list)
    # Publication readiness + authenticity
    dual_qa_publication_readiness: float = 0.0
    dual_qa_article_authenticity: float = 0.0
    dual_qa_image_authenticity: float | None = None
    dual_qa_overall_authenticity: float = 0.0
    dual_qa_authenticity_label: str = ""
    dual_qa_authenticity_narrative: str = ""
    # QA cost breakdown (USD)
    dual_qa_claude_review_cost: float = 0.0
    dual_qa_openai_review_cost: float = 0.0
    dual_qa_revision_cost: float = 0.0
    dual_qa_vision_claude_cost: float = 0.0
    dual_qa_vision_openai_cost: float = 0.0
    dual_qa_total_cost: float = 0.0
    # QA timing
    dual_qa_elapsed_seconds: float = 0.0
    dual_qa_avg_cycle_seconds: float = 0.0

    # Stage 4b: SEO QA (rule-based structural check — backstop)
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

_LAST_ARTICLE_PATH = Path("output/last_article.json")


@app.command()
def generate(
    topic: str = typer.Option(..., "--topic", "-t", help="Main topic of the article."),
    service: str | None = typer.Option(None, "--service", "-s", help="Service or product the article supports."),
    city: str | None = typer.Option(None, "--city", help="Target city for local SEO."),
    state: str | None = typer.Option(None, "--state", help="Target state or province (required with --city)."),
    words: int = typer.Option(settings.default_word_count, "--words", "-w", min=300, max=950, help="Target word count (700–900; default 800, hard cap 950)."),
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

    # ── Validate CLI flags ──────────────────────────────────────
    if bool(city) != bool(state):
        console.print("[red]Error:[/red] --city and --state must be provided together.")
        raise typer.Exit(code=1)

    # ── Load credential URL (for URL-domain extraction in resolver) ─
    website_url: str | None = None
    try:
        _creds = CredentialStore(settings.credentials_dir).load(resolved_client, resolved_website)
        website_url = _creds.url
    except Exception:
        pass

    # ── Build initial request ───────────────────────────────────
    location = Location(city=city, state=state) if city and state else None
    request = ArticleRequest(
        topic=topic,
        service=service,
        location=location,
        word_count=words,
        tone=tone,
        language=language,
        focus_keyword=keyword,
        website_url=website_url,
    )

    # ── Resolve business context before planning ────────────────
    # Tries: CLI flags → SiteProfile → WP API → homepage HTML → domain heuristic.
    # If location cannot be resolved, generation continues as a non-local article.
    request = BusinessContextResolver(settings.profiles_dir).resolve(
        resolved_client, resolved_website, request
    )
    if not request.location:
        console.print(
            "[yellow]Warning:[/yellow] Location could not be resolved — "
            "generating without geographic targeting.\n"
            "  Pass [bold]--city[/bold] / [bold]--state[/bold], or run "
            f"[bold]seo profile create --client {resolved_client} "
            f"--website {resolved_website}[/bold] to persist site context."
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

    # ── Load credential URL (for URL-domain extraction in resolver) ─
    website_url: str | None = None
    try:
        _creds = CredentialStore(settings.credentials_dir).load(client_id, website_id)
        website_url = _creds.url
    except Exception:
        pass

    # ── Peek at SiteProfile for wizard defaults ─────────────────
    profile = SiteProfileService(settings.profiles_dir).load(client_id, website_id)
    profile_city: str | None = profile.city if profile else None
    profile_state: str | None = profile.state if profile else None
    profile_service: str | None = profile.primary_service if profile else None

    if profile:
        console.print(
            f"[dim]Site profile:[/dim] {profile.business_name} — "
            f"{profile.primary_service}, {profile.city}, {profile.state}"
        )
    elif website_url:
        console.print(f"[dim]Credential URL:[/dim] {website_url}")

    # ── Topic ──────────────────────────────────────────────────
    topic = typer.prompt("Article topic")

    # ── Service ────────────────────────────────────────────────
    svc_raw = typer.prompt(
        f"Service  [Enter to use: '{profile_service}']" if profile_service else "Service  [optional, Enter to skip]",
        default=profile_service or "",
    )
    service: str | None = svc_raw.strip() or None

    # ── Location ───────────────────────────────────────────────
    city_raw = typer.prompt(
        f"City     [Enter to use: '{profile_city}']" if profile_city else "City     [optional — resolver will try URL]",
        default=profile_city or "",
    )
    city: str | None = city_raw.strip() or None
    state: str | None = None
    if city:
        state_raw = typer.prompt(
            f"State    [Enter to use: '{profile_state}']" if profile_state else f"State / Province for {city}",
            default=profile_state or "",
        )
        state = state_raw.strip() or None
        if not state:
            city = None

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
        website_url=website_url,
    )

    # ── Resolve business context before planning ────────────────
    request = BusinessContextResolver(settings.profiles_dir).resolve(
        client_id, website_id, request
    )
    if request.location:
        console.print(
            f"[dim]Resolved:[/dim] {request.service or '(service unknown)'} — "
            f"{request.location.city}, {request.location.state}"
        )
    else:
        console.print(
            "[yellow]Warning:[/yellow] Location could not be resolved — "
            "generating without geographic targeting.\n"
            "  Create a site profile to persist context: "
            f"[bold]seo profile create --client {client_id} --website {website_id}[/bold]"
        )

    _execute_generation(request, tenant, settings.output_dir)


# ── Private helpers ───────────────────────────────────────────────────────────

def _save_article(article: Article, base_dir: Path) -> Path:
    safe_slug = re.sub(r'[^a-z0-9-]', '-', article.seo.slug.lower()).strip('-') or "article"
    article_dir = (
        base_dir
        / article.tenant.client_id
        / article.tenant.website_id
        / safe_slug
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


def _save_last_article(article: "Article") -> None:
    """Persist article to output/last_article.json for the republish command."""
    try:
        _LAST_ARTICLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LAST_ARTICLE_PATH.write_text(article.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Last article saved: %s", _LAST_ARTICLE_PATH)
    except OSError as exc:
        logger.warning("Could not save last article to %s: %s", _LAST_ARTICLE_PATH, exc)


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
    Generate an article (or reuse an existing draft), save to disk, and display results.

    Shared by the `generate`, `interactive`, and `autopublish` commands.
    Returns the path to the saved article.json.

    Execution order (budget-aware):
      1. Draft Pool lookup  — free, no API calls, runs regardless of budget state.
      2a. Reuse path        — adapts the matched draft; minor LLM calls (SEO regen,
                              location rewrite) fall back gracefully if budget is tight.
      2b. Budget check      — runs ONLY when no reusable draft exists.
      3.  Fresh generation  — full article generation, blocked if budget exceeded.

    This order guarantees that an exhausted monthly budget never prevents reuse of
    an existing draft.
    """
    from services.draft_pool_service import DraftPoolService
    from services.location_adaptation_service import LocationAdaptationService
    from services.reuse_stats_service import ReuseStatsService
    from services.seo_cache_service import SEOCacheService
    from services.topic_normalization import normalize_topic_id

    budget_before = budget.status()
    checkpoint_dir = base_dir / tenant.client_id / tenant.website_id / ".checkpoints"
    start = time.perf_counter()
    reused = False
    location_adapted = False
    seo_from_cache = False
    reuse_match: DraftMatch | None = None
    article = None

    stats = ReuseStatsService(base_dir)

    # ── Canonical client enrichment ───────────────────────────────────────────
    # Best-effort: load canonical_client from the site profile so generated and
    # reused articles carry the normalized identity from the start.  Missing
    # canonical_client is NOT an error here — the publish gate validates it.
    try:
        _cc_prof = SiteProfileService(settings.profiles_dir).load(
            tenant.client_id, tenant.website_id
        )
        if _cc_prof and _cc_prof.canonical_client:
            tenant = tenant.model_copy(update={"canonical_client": _cc_prof.canonical_client})
    except Exception:
        pass

    # ── STEP 1: Draft Pool lookup — runs before any budget check ─────────────
    # All operations here are free (no API calls):
    #   pool index load, topic_id normalization, Jaccard scoring, article.json read.
    if settings.enable_draft_reuse:
        try:
            _profile = SiteProfileService(settings.profiles_dir).load(
                tenant.client_id, tenant.website_id
            )
            _req_reuse_group = _profile.reuse_group if _profile else None
            if _req_reuse_group is not None:
                tenant = tenant.model_copy(update={"reuse_group": _req_reuse_group})

            # Pass 1: fast in-memory pool lookup (no filesystem scan)
            pool = DraftPoolService(base_dir)
            pool.build_or_load()

            pool_match = pool.find_match(request, tenant, _req_reuse_group)
            if pool_match:
                _full = pool.load_article(pool_match, base_dir)
                if _full:
                    reuse_match = DraftMatch(
                        article=_full,
                        source_path=base_dir / pool_match.entry.article_path,
                        similarity=pool_match.similarity,
                        same_website=pool_match.same_website,
                        matched_by_topic_id=pool_match.matched_by_topic_id,
                    )

            # Pass 2: fallback filesystem scan if pool had no match
            if reuse_match is None:
                reuse_svc = DraftReuseService(base_dir)
                reuse_match = reuse_svc.find_match(request, tenant, req_reuse_group=_req_reuse_group)

        except Exception as exc:
            logger.warning("Draft reuse search failed (non-blocking): %s", exc)
            reuse_match = None

        if reuse_match is not None:
            reused = True

            # ── Adapt tenant/request (free, no API) ───────────────────────
            from models.enums import ArticleStatus
            from models.publishing import PublishingOptions
            article = reuse_match.article.model_copy(update={
                "tenant": tenant,
                "request": request,
                "publishing": PublishingOptions(),
                "status": ArticleStatus.REVIEW,
                "wp_post_id": None,
                "wp_post_url": None,
                "drive_document_id": None,
            })

            match_label = (
                "topic_id" if reuse_match.matched_by_topic_id
                else f"{reuse_match.similarity:.0%} similarity"
            )
            console.print(
                f"  [dim]Draft reuse:[/dim] [green]Matched[/green] "
                f"[bold]{reuse_match.article.title[:70]}[/bold]  "
                f"({match_label})  "
                f"[dim]{reuse_match.source_path}[/dim]"
            )

            # ── Location adaptation (free direct replacement; targeted LLM
            #    only for residual refs — falls back gracefully if budget tight)
            original_location = (
                reuse_match.article.request.location
                if reuse_match.article.request else None
            )
            target_location = request.location
            locations_differ = (
                original_location is not None
                and target_location is not None
                and original_location.city.lower() != target_location.city.lower()
            )

            if locations_differ:
                try:
                    loc_svc = LocationAdaptationService(claude)
                    article, loc_report = loc_svc.adapt(article, original_location, target_location)
                    location_adapted = True
                    _display_location_scan(loc_report)
                    stats.record_location_adapted()
                    if loc_report.sections_llm_budget_skipped:
                        stats.record_location_refinement_skipped(loc_report.sections_llm_budget_skipped)
                except Exception as exc:
                    logger.warning("Location adaptation failed (non-blocking): %s", exc)
                    console.print(
                        f"  [yellow]Warning:[/yellow] Location adaptation failed — body may contain "
                        f"references to {original_location.city}."
                    )

            # ── SEO cache check (free) / regeneration (minor LLM call) ───
            # Budget-aware: if the monthly limit is exceeded, skip SEO regen and
            # reuse the original metadata — never block the entire reuse path.
            req_topic_id = normalize_topic_id(request.topic, request.location)
            seo_cache = SEOCacheService(base_dir, tenant.client_id, tenant.website_id)
            cached_seo = seo_cache.get(req_topic_id, request.focus_keyword)

            if cached_seo is not None:
                article = article.model_copy(update={"seo": cached_seo})
                seo_from_cache = True
                stats.record_seo_cache_hit()
                console.print("  [dim]SEO metadata loaded from cache (no API call).[/dim]")
            else:
                _seo_budget_ok = True
                try:
                    budget.check_monthly_total(settings.max_monthly_cost_usd)
                except BudgetExceededError:
                    _seo_budget_ok = False
                    logger.warning(
                        "SEO regeneration skipped (monthly budget exceeded). "
                        "Reusing existing SEO."
                    )
                    console.print(
                        "  [dim]SEO regeneration skipped (monthly budget exceeded). "
                        "Reusing existing SEO.[/dim]"
                    )
                    stats.record_seo_regen_skipped()

                if _seo_budget_ok:
                    try:
                        with console.status(
                            "[bold green]Regenerating SEO metadata for this website...",
                            spinner="dots",
                        ):
                            new_seo = article_agent._generate_seo(request, article.content_markdown)
                        article = article.model_copy(update={"seo": new_seo})
                        seo_cache.put(req_topic_id, new_seo, request.focus_keyword)
                        console.print("  [dim]SEO metadata regenerated and cached.[/dim]")
                    except Exception as exc:
                        logger.warning("SEO regeneration after reuse failed (non-blocking): %s", exc)
                        console.print(
                            "  [yellow]Warning:[/yellow] SEO regeneration failed — "
                            "reusing original SEO metadata."
                        )

    # ── STEP 2: Budget check — only reached when no draft was found ───────────
    if not reused:
        try:
            budget.check_monthly_total(settings.max_monthly_cost_usd)
        except BudgetExceededError as exc:
            stats.record_budget_block_generation()
            try:
                stats.save()
            except Exception:
                pass
            console.print(
                f"\n[red]No reusable draft found and monthly generation budget has been exceeded.[/red]\n"
                f"[dim]{exc}[/dim]"
            )
            raise typer.Exit(code=1)

    # ── STEP 3: Fresh generation (only when no reusable draft exists) ─────────
    if not reused:
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
            except ArticleValidationError as exc:
                console.print(f"\n[red]Validation error:[/red] {exc}")
                raise typer.Exit(code=1)
            except LLMAllProvidersFailedError:
                raise typer.Exit(code=1)
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

    # ── Update draft pool with newly saved article ────────────────────────────
    try:
        pool = DraftPoolService(base_dir)
        pool.build_or_load()
        pool.add_entry(article, article_dir / "article.json")
        pool.save()
    except Exception as exc:
        logger.warning("Draft pool update failed (non-blocking): %s", exc)

    _display_result(article, article_dir, elapsed)

    # ── Cost report ───────────────────────────────────────────────────────────
    budget_after = budget.status()
    article_cost = round(
        (budget_after["claude"]["usd"] - budget_before["claude"]["usd"])
        + (budget_after["openai"]["usd"] - budget_before["openai"]["usd"]),
        6,
    )
    _display_cost_report(
        claude_cost=round(budget_after["claude"]["usd"] - budget_before["claude"]["usd"], 6),
        openai_cost=round(budget_after["openai"]["usd"] - budget_before["openai"]["usd"], 6),
        article_cost=article_cost,
        monthly_total=budget_after["claude"]["usd"] + budget_after["openai"]["usd"],
        monthly_limit=settings.max_monthly_cost_usd,
        article_limit=settings.max_article_cost_usd,
        reused=reused,
        reuse_match=reuse_match,
    )

    # ── Reuse stats: record + display ─────────────────────────────────────────
    req_topic_id_for_stats = normalize_topic_id(request.topic, request.location)
    if reused:
        api_calls_avoided = 1  # article generation skipped
        if seo_from_cache:
            api_calls_avoided += 1  # SEO regen also skipped
        stats.record_reuse(req_topic_id_for_stats, savings_usd=settings.max_article_cost_usd)
        stats.record_api_calls_avoided(api_calls_avoided)
    else:
        stats.record_generation(cost_usd=article_cost)

    try:
        stats.save()
    except Exception as exc:
        logger.warning("Stats save failed (non-blocking): %s", exc)

    _display_reuse_stats(stats.monthly_report())

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
    no_qa: bool = False,
    post_id: int | None = None,
    show_pipeline_report: bool = True,
    event_type: str = "publish",
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

    # ── Monthly budget guard ──────────────────────────────────────────────────
    try:
        budget.check_monthly_total(settings.max_monthly_cost_usd)
    except BudgetExceededError as exc:
        console.print(f"\n[red]Monthly budget exceeded:[/red] {exc}")
        raise typer.Exit(code=1)

    budget_before = budget.status()

    # ── Canonical client validation ───────────────────────────────────────────
    # Required before any BQ write. Legacy article.json files (generated before
    # Request #5) will have tenant.canonical_client = None; resolve from profile.
    if not article.tenant.canonical_client:
        try:
            _val_prof = SiteProfileService(settings.profiles_dir).load(
                article.tenant.client_id, article.tenant.website_id
            )
        except Exception:
            _val_prof = None
        if _val_prof and _val_prof.canonical_client:
            article = article.model_copy(update={
                "tenant": article.tenant.model_copy(
                    update={"canonical_client": _val_prof.canonical_client}
                )
            })
        else:
            _val_path = (
                settings.profiles_dir
                / article.tenant.client_id
                / article.tenant.website_id
                / "site.json"
            )
            console.print(f"\n[bold red]Missing canonical_client in:[/bold red]")
            console.print(f"  {_val_path}")
            console.print(
                "[dim]  Add: \"canonical_client\": \"<your-cortex-id>\" to site.json, "
                "then re-run.[/dim]"
            )
            raise typer.Exit(code=1)

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

    # Content sanitization — remove UI artifacts before any pipeline stage sees the content
    san_result = ContentSanitizationService().sanitize(article.content_markdown)
    if san_result.changed:
        _san_words = len(san_result.markdown.split())
        article = article.model_copy(update={
            "content_markdown": san_result.markdown,
            "word_count": _san_words,
            "reading_time_minutes": max(1, _san_words // 200),
        })
        _display_sanitization(san_result)

    # Image resolution (before WP connection — no WP calls needed)
    image_plan = None
    resolved_images: list = []
    drive_count = 0
    resolve_elapsed = 0.0
    img_stats: dict = {}

    editorial_history = EditorialHistoryService(
        settings.editorial_history_path.parent
        / article.tenant.client_id
        / article.tenant.website_id
        / "image_usage.json"
    )
    if not no_image:
        resolve_start = time.perf_counter()
        image_plan, resolved_images, drive_count, img_stats = _resolve_images(
            article, editorial_history=editorial_history
        )
        resolve_elapsed = time.perf_counter() - resolve_start

    # Merge image markers into the canonical markdown before QA and publishing.
    # From this point on article.content_markdown is the single source of truth:
    # QA sees and preserves the markers; the publisher reads them from there directly.
    if image_plan is not None:
        article = article.model_copy(update={
            "content_markdown": image_plan.modified_markdown
        })
        try:
            article_path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
            (article_path.parent / "article.md").write_text(
                article.content_markdown, encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not persist marker-bearing markdown to disk: %s", exc)

    # Dual QA (before WordPress — no wasted API calls if article fails review).
    # DualQAAgent.run() also guarantees full marker integrity when image_plan is supplied.
    qa_report = None
    if settings.qa_enabled and not no_qa:
        article, resolved_images, qa_report = _run_dual_qa(
            article, resolved_images, article_path,
            image_plan=image_plan,
        )

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
            state.images_active              = True
            state.drive_indexed              = drive_count
            state.drive_semantic_candidates  = img_stats.get("drive_semantic_candidates", 0)
            state.img_requested              = len(image_plan.requests)
            state.img_from_drive             = sum(1 for _, a in resolved_images if a.source == ImageSource.DRIVE)
            state.img_from_edited            = sum(1 for _, a in resolved_images if a.source == ImageSource.EDITED)
            state.ai_reasons                 = [
                a.ai_reason for _, a in resolved_images
                if a.ai_reason and a.source == ImageSource.EDITED
            ]
            state.openai_budget_total     = img_stats.get("edit_budget_total", 0)
            state.openai_budget_remaining = img_stats.get("edit_budget_remaining", 0)
            state.edited_photos           = img_stats.get("edited_photos", [])

        state.t_images = resolve_elapsed

        # ── Stage 4: Dual QA state ───────────────────────────────
        if qa_report is not None:
            from models.qa_report import DualQAReport as _DualQAReport
            state.dual_qa_enabled = True
            state.dual_qa_passed = qa_report.article_review_passed
            state.dual_qa_iterations = qa_report.iterations_used
            if qa_report.final_article_review is not None:
                r = qa_report.final_article_review
                state.dual_qa_seo_score       = r.seo_score
                state.dual_qa_editorial_score = r.editorial_score
                state.dual_qa_writing_score   = r.writing_score
                state.dual_qa_authenticity_score = r.authenticity_score
                state.dual_qa_combined_score  = round(r.combined_score, 1)
            reviewed_images = qa_report.image_results
            state.dual_qa_images_reviewed = len(reviewed_images)
            state.dual_qa_images_passed   = sum(1 for r in reviewed_images if r.approved)
            state.dual_qa_images_failed   = sum(1 for r in reviewed_images if not r.approved)
            state.dual_qa_rejection_reasons = qa_report.rejection_reasons
            state.dual_qa_image_results = qa_report.image_results
            # Publication readiness + authenticity
            state.dual_qa_publication_readiness = qa_report.publication_readiness_score
            state.dual_qa_article_authenticity  = qa_report.article_authenticity
            state.dual_qa_image_authenticity    = qa_report.image_authenticity
            state.dual_qa_overall_authenticity  = qa_report.overall_authenticity
            state.dual_qa_authenticity_label    = qa_report.authenticity_label
            state.dual_qa_authenticity_narrative = qa_report.authenticity_narrative
            # QA cost breakdown
            state.dual_qa_claude_review_cost  = qa_report.claude_review_cost_usd
            state.dual_qa_openai_review_cost  = qa_report.openai_review_cost_usd
            state.dual_qa_revision_cost       = qa_report.revision_cost_usd
            state.dual_qa_vision_claude_cost  = qa_report.vision_claude_cost_usd
            state.dual_qa_vision_openai_cost  = qa_report.vision_openai_cost_usd
            state.dual_qa_total_cost          = qa_report.total_qa_cost_usd
            # QA timing
            state.dual_qa_elapsed_seconds     = qa_report.qa_elapsed_seconds
            state.dual_qa_avg_cycle_seconds   = qa_report.avg_cycle_seconds

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

        # ── Link enrichment (upstream of readiness gate) ─────────
        links_added = 0
        if not no_links:
            try:
                posts = wp.list_posts()
                enriched_md = link_enricher.enrich(article, posts, article.content_markdown)
                links_added = link_enricher.last_links_added
                _enrich_words = len(enriched_md.split())
                article = article.model_copy(update={
                    "content_markdown": enriched_md,
                    "word_count": _enrich_words,
                    "reading_time_minutes": max(1, _enrich_words // 200),
                })
            except Exception as exc:
                logger.warning("Link enrichment failed (non-blocking): %s", exc)

        # ── SEO QA for readiness gate ────────────────────────────
        gate_seo_report = None
        try:
            gate_seo_report = SEOQAService().analyze(article)
        except Exception as exc:
            logger.warning("Pre-gate SEO QA failed: %s", exc)

        # ── Publication readiness gate ───────────────────────────
        dual_qa_passed_bool: bool | None = None
        if state.dual_qa_enabled:
            dual_qa_passed_bool = state.dual_qa_passed
        readiness = PublicationReadinessService().validate(
            article=article,
            image_plan=image_plan,
            resolved_images=resolved_images,
            uploaded_images=uploaded_images,
            links_added=links_added,
            no_links=no_links,
            seo_qa_report=gate_seo_report,
            min_seo_score=effective_min_score,
            dual_qa_passed=dual_qa_passed_bool,
            min_word_count=settings.min_article_words,
        )
        _display_readiness_gate(readiness)
        if not readiness.ready:
            warn_suffix = (
                f"  ({len(readiness.warnings)} warning(s) noted but non-blocking.)"
                if readiness.warnings else ""
            )
            console.print(
                "\n[red]Publication blocked:[/red] "
                f"{len(readiness.failures)} readiness check(s) failed. "
                f"No WordPress post was created.{warn_suffix}"
            )
            raise typer.Exit(code=1)

        # ── Publish ──────────────────────────────────────────────
        # link_enricher=None because enrichment already ran upstream
        t_publish_start = time.perf_counter()
        try:
            with console.status("[bold green]Publishing to WordPress...", spinner="dots"):
                updated = agent.publish(
                    article,
                    min_score=effective_min_score,
                    uploaded_images=uploaded_images or None,
                    update_post_id=post_id,
                    link_enricher=None,
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

        # ── Stage 4b: SEO QA score ───────────────────────────────
        state.qa_score = agent.last_qa_report.score if agent.last_qa_report else (gate_seo_report.score if gate_seo_report else 0)

        state.t_publish = time.perf_counter() - t_publish_start
        state.post_id     = updated.wp_post_id
        state.post_url    = updated.wp_post_url
        state.post_status = updated.publishing.status.value
        state.post_slug   = updated.seo.slug

        # Update editorial image usage history — only for genuinely published articles.
        # Drafts may read existing history for diversity scoring but must never write it.
        if image_plan is not None and uploaded_images and state.post_status == "publish":
            _update_editorial_history(
                editorial_history,
                resolved_images=resolved_images,
                uploaded_images=uploaded_images,
                slug=updated.seo.slug,
                post_id=updated.wp_post_id,
            )
            console.print(
                "[dim]Editorial Diversity History:[/dim] Updated — published article recorded"
            )
        else:
            console.print(
                f"[dim]Editorial Diversity History:[/dim] Not updated — "
                f"WordPress status is {state.post_status}"
            )

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

        # ── Evict from draft pool (published articles are never reusable) ──
        try:
            from services.draft_pool_service import DraftPoolService
            _pool = DraftPoolService(settings.output_dir)
            _pool.build_or_load()
            if _pool.remove_entry(article_path):
                _pool.save()
        except Exception as _pool_exc:
            logger.debug("Draft pool eviction skipped: %s", _pool_exc)

        # ── Save image report (silent) ───────────────────────────
        if image_plan is not None:
            try:
                _save_image_report(
                    article_path, image_plan, resolved_images, uploaded_images,
                    drive_count, resolve_elapsed, article,
                )
            except Exception as exc:
                logger.warning("Could not save image resolution report: %s", exc)

        # ── Publication certification ────────────────────────────
        try:
            wp_post_data = wp.get_post(updated.wp_post_id) or {} if updated.wp_post_id else {}
            cert_report = PublicationCertificationService().certify(
                article=updated,
                wp_post=wp_post_data,
                uploaded_images=uploaded_images,
                seo_qa_report=gate_seo_report,
                links_added=links_added,
                no_links=no_links,
                wp_service=wp,
                editorial_history=editorial_history if image_plan is not None else None,
                min_seo_score=effective_min_score,
                min_word_count=settings.min_article_words,
            )
            _display_certification(cert_report)
        except Exception as exc:
            logger.warning("Publication certification failed (non-blocking): %s", exc)

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
            _display_diversity_report(img_stats)

        # ── BigQuery analytics sink (fire-and-forget) ────────────────────────
        # Runs after every successful WP publish — errors are swallowed by
        # BqSinkService; publication is never blocked or interrupted.
        import services.call_tracer as _bq_ct
        from services.bq_sink_service import BqSinkService
        _bq_tracer = _bq_ct.get()
        _bq_article_id = str(updated.id)
        _bq_canonical = updated.tenant.canonical_client
        _bq = BqSinkService()
        _bq.insert_article(updated, qa_report, _bq_tracer, state.t_total, event_type=event_type)
        if qa_report is not None:
            _bq.insert_qa_results(_bq_article_id, qa_report, canonical_client=_bq_canonical)
        if _bq_tracer:
            _bq.insert_llm_costs(
                "seo-agent", _bq_tracer,
                article_id=_bq_article_id, event_type=event_type,
                canonical_client=_bq_canonical,
            )

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

    import services.call_tracer as _call_tracer
    _call_tracer.start()

    _run_publish_flow(
        input, article,
        status=status,
        min_score=min_score,
        no_image=no_image,
        no_links=no_links,
        post_id=post_id,
        show_pipeline_report=True,
        event_type="publish",
    )

    _tracer = _call_tracer.get()
    if _tracer and _tracer.records:
        console.print(_tracer.summary())


# ── Image resolution helper ───────────────────────────────────────────────────

def _resolve_images(article: Article, *, editorial_history: EditorialHistoryService | None = None) -> tuple:
    """
    Plan and resolve images for an article using the Photo Preservation Pipeline.

    Drive images are served from a local SQLite index (DriveImageIndex) that
    is synced from the global DRIVE_FOLDER_ID folder. The index is only
    refreshed when stale (age > DRIVE_SYNC_MAX_AGE_HOURS, default 7 days),
    so most publish runs incur zero Drive API traversal cost.

    Returns:
      (plan, resolved, n_candidates, img_stats)

    Returns (None, [], 0, {}) if Drive is not configured or setup fails.
    Errors are caught and reported as warnings so publishing can continue.
    The pipeline NEVER generates images from scratch — OpenAI is only used for
    minimal preservation edits on Drive photos.
    """
    folder_id = settings.drive_folder_id
    drive_svc = None
    generator = None
    drive_candidates = []

    if settings.google_sa_json_path and settings.google_sa_json_path.exists() and folder_id:
        try:
            drive_svc = GoogleDriveService(settings.google_sa_json_path)

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

    # OpenAI is only used for P2 preservation edits — not for generation.
    if settings.openai_api_key:
        try:
            generator = OpenAIImageGenerator(settings.openai_api_key, budget=budget)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Image generator setup failed: {exc}")

    n_candidates = len(drive_candidates)

    if not drive_candidates:
        logger.info("No Drive candidates — image slots will be skipped.")
        return None, [], 0, {}

    resolver = ImageResolverAgent(
        claude=claude,
        drive=drive_svc,
        generator=generator,
        exact_score=settings.drive_exact_score,
        partial_score=settings.drive_partial_score,
        max_ai=settings.max_openai_images_per_article,
        editorial_history=editorial_history,
    )

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
        return None, [], n_candidates, {}

    # Phase 2: resolve using the preservation pipeline
    try:
        with console.status(
            f"[bold green]Resolving {len(plan.requests)} image(s)...", spinner="dots"
        ):
            resolved = resolver.resolve_all(plan, drive_candidates=drive_candidates)
    except ImageResolverError as exc:
        console.print(f"[yellow]Warning:[/yellow] Image resolution failed — publishing without images: {exc}")
        return None, [], n_candidates, {}
    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] Unexpected error during image resolution: {exc}")
        return None, [], n_candidates, {}

    return plan, resolved, n_candidates, resolver.last_run_stats


# ── Dual QA helper ────────────────────────────────────────────────────────────

def _run_dual_qa(
    article: Article,
    resolved_images: list,
    article_path: Path,
    *,
    image_plan: Any = None,
) -> tuple[Article, list, Any]:
    """
    Run the dual QA pipeline (Claude + OpenAI) on the article and any edited images.

    Returns (article, resolved_images, qa_report).
    article may be a revised version. Edited photos (ImageSource.EDITED) that fail
    identity preservation QA are reverted to their original Drive photo.

    If image_plan is supplied, DualQAAgent guarantees full marker integrity before
    returning — no marker repair is needed in the caller.

    On DualQAFailedError: saves QA report to disk, prints the failure report, and
    raises typer.Exit(code=1) — the article is NOT published.
    """
    from models.qa_report import DualQAReport

    openai_reviewer = None
    if settings.openai_api_key:
        try:
            openai_reviewer = OpenAIReviewService(
                api_key=settings.openai_api_key,
                text_model=settings.openai_text_review_model,
                vision_model=settings.openai_vision_review_model,
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] OpenAI reviewer setup failed — Claude-only QA: {exc}")
    else:
        console.print("[yellow]Warning:[/yellow] OPENAI_API_KEY not set — running Claude-only dual QA.")

    qa_agent = DualQAAgent(
        claude=claude,
        openai_reviewer=openai_reviewer,
        min_seo=settings.qa_min_seo,
        min_editorial=settings.qa_min_editorial,
        min_writing=settings.qa_min_writing,
        min_authenticity=settings.qa_min_authenticity,
        min_vision_claude=settings.qa_min_vision_claude,
        min_vision_openai=settings.qa_min_vision_openai,
        max_cycles=settings.qa_max_cycles,
        enable_rescue=settings.qa_rescue_enabled,
    )

    try:
        with console.status(
            f"[bold green]Dual QA review (max {settings.qa_max_cycles} cycle(s))...",
            spinner="dots",
        ):
            approved_article, approved_images, report = qa_agent.run(
                article, resolved_images,
                image_plan=image_plan,
            )

        # Show brief QA summary
        r = report.final_article_review
        if r:
            console.print(
                f"  [dim]Dual QA:[/dim] [green]PASS[/green]  "
                f"SEO={r.seo_score} Editorial={r.editorial_score} "
                f"Writing={r.writing_score} Authenticity={r.authenticity_score}  "
                f"({report.iterations_used} cycle(s))"
            )
        removed = len(resolved_images) - len(approved_images)
        if removed:
            console.print(f"  [dim]Images:[/dim] {removed} AI image(s) excluded by vision review.")

        _save_qa_report(article_path, report)
        return approved_article, approved_images, report

    except DualQAFailedError as exc:
        report = exc.report
        _save_qa_report(article_path, report)
        _display_qa_failure(report, article_path)
        import services.call_tracer as _ct
        _ct_inst = _ct.get()
        if _ct_inst and _ct_inst.records:
            console.print(_ct_inst.summary())
        raise typer.Exit(code=1)


def _update_editorial_history(
    history: Any,
    *,
    resolved_images: list,
    uploaded_images: list,
    slug: str,
    post_id: int | None,
) -> None:
    """Record published images in the editorial diversity database."""
    from models.image_asset import ImageSource
    from models.image_request import ImagePurpose
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    uploaded_ids = {req.id for req, _ in uploaded_images}
    resolved_map = {req.id: (req, asset) for req, asset in resolved_images}

    for img_id in uploaded_ids:
        pair = resolved_map.get(img_id)
        if pair is None:
            continue
        req, asset = pair
        file_id = None
        if asset.source == ImageSource.DRIVE:
            file_id = asset.source_detail
        elif asset.source == ImageSource.EDITED:
            file_id = asset.reference_file_id
        if not file_id:
            continue
        history.record_publication(
            file_id=file_id,
            filename=asset.filename or "",
            slug=slug,
            post_id=post_id,
            purpose="featured" if req.purpose == ImagePurpose.FEATURED else "inline",
            date=today,
        )

    history.finalize_article(slug)
    history.save()


def _display_diversity_report(img_stats: dict) -> None:
    div = img_stats.get("diversity_report")
    if not div or not div.get("selections"):
        return

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=28)
    table.add_column("Value", overflow="fold")

    table.add_row("Candidates evaluated by Vision", str(div.get("candidates_considered", 0)))
    table.add_row("Excluded by recency", str(div.get("excluded_recent", 0)))
    table.add_row("Excluded by folder duplicate", str(div.get("excluded_folder_duplicate", 0)))
    table.add_row("Previously unused images selected", str(div.get("previously_unused", 0)))

    for sel in div.get("selections", []):
        table.add_row("", "")
        slot_label = "Featured Image" if sel.get("purpose") == "featured" else sel.get("slot", "?")
        vision_color = "green" if sel.get("vision_score", 0) >= 75 else "yellow"
        ed_score = sel.get("editorial_score", 0)
        ed_color = "green" if ed_score >= 80 else "yellow" if ed_score >= 60 else "dim"
        table.add_row(slot_label, "")
        table.add_row("  Vision score", f"[{vision_color}]{sel.get('vision_score', '?')}[/{vision_color}]")
        table.add_row("  Editorial score", f"[{ed_color}]{ed_score:.1f}[/{ed_color}]")
        table.add_row("  Times used", str(sel.get("times_used", "?")))
        table.add_row("  Reason", str(sel.get("reason", ""))[:100])

    console.print()
    console.print(Panel(table, title="[bold]Image Diversity Report[/bold]", expand=False))


def _save_qa_report(article_path: Path, report: Any) -> None:
    """Save the full QA report + per-cycle JSONs inside a qa/ subdirectory."""
    import json as _json
    try:
        qa_dir = article_path.parent / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)

        # Full consolidated report
        (qa_dir / "report.json").write_text(
            _json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Per-cycle files — one per article review iteration
        for iteration in report.article_iterations:
            cycle_path = qa_dir / f"cycle-{iteration.iteration}.json"
            cycle_path.write_text(
                _json.dumps(iteration.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("Could not save QA report: %s", exc)


def _display_qa_failure(report: Any, article_path: Path) -> None:
    """Display the QA failure report when max cycles are exhausted."""
    from rich.rule import Rule

    console.print()
    console.print(Rule("[bold red]DUAL QA FAILED — Article Not Published[/bold red]"))
    console.print()
    console.print(f"  The article failed quality review after {report.iterations_used} revision cycle(s).")
    console.print(f"  The best revision has been saved to: [dim]{article_path}[/dim]")
    console.print(f"  Full QA report: [dim]{article_path.parent / 'qa' / 'report.json'}[/dim]")
    console.print()

    final = report.final_article_review
    if final:
        # ── Score summary table ───────────────────────────────────────────────
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Reviewer", style="bold", min_width=20)
        table.add_column("Dimension", min_width=16)
        table.add_column("Score", min_width=8)
        table.add_column("Required", min_width=8)
        table.add_column("Priority", min_width=8)
        table.add_column("", min_width=6)

        def _pass_fail(score: int, required: int) -> str:
            return "[green]PASS[/green]" if score >= required else "[red]FAIL[/red]"

        def _priority_fmt(p: str) -> str:
            colors = {"High": "red", "Medium": "yellow", "Low": "dim"}
            c = colors.get(p, "dim")
            return f"[{c}]{p}[/{c}]" if p else "—"

        table.add_row(
            "Claude (SEO Editor)", "SEO Quality",
            str(final.seo_score), "≥ 90",
            _priority_fmt(final.seo_detail.priority),
            _pass_fail(final.seo_score, 90),
        )
        table.add_row(
            "", "Editorial Quality",
            str(final.editorial_score), "≥ 90",
            _priority_fmt(final.editorial_detail.priority),
            _pass_fail(final.editorial_score, 90),
        )
        table.add_row(
            "OpenAI (Authenticity)", "Human Writing",
            str(final.writing_score), "≥ 90",
            _priority_fmt(final.writing_detail.priority),
            _pass_fail(final.writing_score, 90),
        )
        table.add_row(
            "", "Authenticity",
            str(final.authenticity_score), "≥ 90",
            _priority_fmt(final.authenticity_detail.priority),
            _pass_fail(final.authenticity_score, 90),
        )
        console.print(table)

        # ── Per-dimension explanation for every failed dimension ──────────────
        failed_dims = final.failed_dimensions()
        if failed_dims:
            console.print()
            console.print("  [bold]Why the article failed:[/bold]")
            console.print()
            for label, score, detail in failed_dims:
                console.print(f"  [bold red]{label}[/bold red]  [dim]{score}/100[/dim]")
                if detail.reasoning:
                    console.print(f"  [dim]{detail.reasoning}[/dim]")
                if detail.weaknesses:
                    console.print()
                    console.print("  [bold]Reasons:[/bold]")
                    for w in detail.weaknesses:
                        console.print(f"    [dim]•[/dim] {w}")
                if detail.improvements:
                    console.print()
                    console.print("  [bold]Improvements:[/bold]")
                    for imp in detail.improvements:
                        console.print(f"    [dim]→[/dim] {imp}")
                console.print()

    # ── Revision compliance history ───────────────────────────────────────────
    all_iterations = getattr(report, "article_iterations", [])
    compliance_iters = [it for it in all_iterations if it.revision_attempts]
    if compliance_iters:
        console.print()
        console.print("  [bold]Revision compliance history:[/bold]")
        console.print()
        for it in compliance_iters:
            attempts = it.revision_attempts
            evaluable = [a for a in attempts if a.evaluable]
            not_evaluable = [a for a in attempts if not a.evaluable]
            applied_count = sum(1 for a in evaluable if a.applied)
            rate_pct = int(100 * applied_count / len(evaluable)) if evaluable else 0
            rate_color = "green" if rate_pct >= 80 else "yellow" if rate_pct >= 50 else "red"

            summary = (
                f"[{rate_color}]{applied_count}/{len(evaluable)} applied ({rate_pct}%)[/{rate_color}]"
            )
            if not_evaluable:
                summary += f"  [dim]{len(not_evaluable)} not evaluable[/dim]"
            console.print(f"  Cycle {it.iteration} revision — {summary}")

            # Instructions not applied (these explain the plateau)
            not_applied = [a for a in evaluable if not a.applied]
            if not_applied:
                console.print("    Instructions [red]not applied[/red]:")
                for a in not_applied:
                    pri = f"[red]{a.priority}[/red]" if a.priority == "High" else f"[dim]{a.priority}[/dim]"
                    console.print(f"      [{pri}] {a.instruction}")
                    if a.evidence:
                        console.print(f"             [dim]{a.evidence}[/dim]")

            # Not-evaluable instructions listed separately, not as failures
            if not_evaluable:
                console.print("    Instructions [dim]not evaluable[/dim] (excluded from score):")
                for a in not_evaluable:
                    console.print(f"      [dim]—[/dim] {a.instruction}")
                    if a.evidence:
                        console.print(f"             [dim]{a.evidence}[/dim]")
        console.print()

    if report.rejection_reasons:
        console.print("  [bold]Rejection summary:[/bold]")
        for reason in report.rejection_reasons:
            console.print(f"    [dim]•[/dim] {reason}")
        console.print()


# ── Display helpers ───────────────────────────────────────────────────────────

def _display_site_validation(result: SiteValidationResult, site_url: str) -> None:
    status_colors = {"READY": "green", "READY_WITH_WARNINGS": "yellow", "FAILED": "red"}
    status_icons = {"READY": "✓", "READY_WITH_WARNINGS": "⚠", "FAILED": "✗"}
    color = status_colors[result.status]
    icon = status_icons[result.status]

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Check", style="dim", min_width=22)
    table.add_column("Result", overflow="fold")

    table.add_row("Site", site_url)
    table.add_row(
        "REST API",
        "[green]Reachable[/green]" if result.rest_api_reachable else "[red]Unreachable[/red]",
    )
    table.add_row(
        "Authentication",
        f"[green]OK[/green] ({result.auth_user})" if result.auth_ok else "[red]FAILED[/red]",
    )
    seo_label = result.seo_plugin.value if result.seo_plugin != SEOPlugin.NONE else "None detected"
    seo_style = "green" if result.seo_plugin != SEOPlugin.NONE else "yellow"
    table.add_row("SEO Plugin", f"[{seo_style}]{seo_label}[/{seo_style}]")
    table.add_row(
        "seo-agent.php",
        "[green]Installed[/green]" if result.agent_plugin_installed else "[yellow]Not detected[/yellow]",
    )

    if result.errors:
        for err in result.errors:
            table.add_row(f"[{color}]Issue[/{color}]", err)

    title = f"[bold][{color}]{icon} Site Validation — {result.status}[/{color}][/bold]"
    console.print()
    console.print(Panel(table, title=title, expand=False))


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


def _display_location_scan(report: "ScanReport") -> None:
    """Print a compact summary of the location adaptation pass."""
    if report.sections_with_refs == 0:
        console.print(
            f"  [dim]Location scan: No '{report.original_city}' references found — body clean.[/dim]"
        )
        return

    summary = (
        f"  [dim]Location scan:[/dim] {report.sections_with_refs} section(s) "
        f"'{report.original_city}' → '{report.target_city}'"
    )
    if report.sections_llm_rewritten:
        summary += f"  [dim]({report.sections_llm_rewritten} AI-refined)[/dim]"
    if report.sections_llm_budget_skipped:
        summary += (
            f"  [dim]{report.sections_llm_budget_skipped} AI refinement(s) skipped "
            f"(monthly budget exceeded — direct adaptation retained)[/dim]"
        )
    console.print(summary)


def _display_reuse_stats(report: dict) -> None:
    """Print the monthly reuse statistics panel after every generate/autopublish run."""
    from rich.table import Table

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Label", style="dim", min_width=26)
    table.add_column("Value", overflow="fold")

    month = report["month"]
    total = report["total_articles"]
    generated = report["articles_generated"]
    reused = report["articles_reused"]
    reuse_pct = report["reuse_percentage"]
    avoided = report["api_calls_avoided"]
    saved = report["dollars_saved"]
    avg_cost = report["average_article_cost"]
    monthly_cost = report["total_cost_usd"]
    seo_hits = report["seo_cache_hits"]
    seo_skipped = report["seo_regens_skipped"]
    pool_hits = report["pool_hits"]
    location_adapted = report["location_adapted"]
    loc_skipped = report["location_refinements_skipped"]
    budget_gen_blocks = report["budget_blocks_generation"]
    budget_minor_blocks = report["budget_blocks_minor"]
    top_topics = report["most_reused_topics"]

    reuse_color = "green" if reuse_pct >= 50 else "yellow" if reuse_pct >= 20 else "dim"
    table.add_row("Month", month)
    table.add_row(
        "Articles this month",
        f"{total} total  "
        f"([green]{generated} generated[/green] / [{reuse_color}]{reused} reused[/{reuse_color}])",
    )
    table.add_row(
        "Reuse rate",
        f"[{reuse_color}]{reuse_pct:.1f}%[/{reuse_color}]",
    )
    table.add_row("API calls avoided", str(avoided))
    table.add_row(
        "Estimated savings",
        f"[green]~${saved:.2f}[/green]",
    )
    table.add_row(
        "Avg article cost",
        f"${avg_cost:.4f}" if avg_cost else "[dim]$0.0000[/dim]",
    )
    table.add_row(
        "Monthly API spend",
        f"${monthly_cost:.4f}",
    )
    if seo_hits:
        table.add_row("SEO cache hits", str(seo_hits))
    if pool_hits:
        table.add_row("Pool lookups (fast)", str(pool_hits))
    if location_adapted:
        table.add_row("Location-adapted", str(location_adapted))
    # ── Budget-skip counters (only shown when non-zero) ───────────────────────
    if seo_skipped:
        table.add_row(
            "SEO regens skipped",
            f"[dim]{seo_skipped} (budget limit — existing SEO reused)[/dim]",
        )
    if loc_skipped:
        table.add_row(
            "Location AI skipped",
            f"[dim]{loc_skipped} section(s) (budget limit — direct replacement retained)[/dim]",
        )
    if budget_gen_blocks:
        table.add_row(
            "Generation blocked",
            f"[yellow]{budget_gen_blocks} (budget limit — no draft available)[/yellow]",
        )
    if budget_minor_blocks and not seo_skipped and not loc_skipped:
        table.add_row(
            "Minor adaptations skipped",
            f"[dim]{budget_minor_blocks} (budget limit)[/dim]",
        )
    if top_topics:
        topic_str = "  ".join(f"{t} ×{n}" for t, n in top_topics[:3])
        table.add_row("Most reused topics", f"[dim]{topic_str}[/dim]")

    console.print()
    console.print(Panel(table, title=f"[bold]Reuse Stats — {month}[/bold]", expand=False))


def _display_cost_report(
    *,
    claude_cost: float,
    openai_cost: float,
    article_cost: float,
    monthly_total: float,
    monthly_limit: float,
    article_limit: float,
    reused: bool,
    reuse_match: "DraftMatch | None",
) -> None:
    """Print a concise cost summary after every generate / autopublish run."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Label", style="dim", min_width=24)
    table.add_column("Value", overflow="fold")

    if reused:
        table.add_row("Article", "[green]Reused (no API calls)[/green]")
        if reuse_match:
            savings_est = round(article_limit, 4)
            table.add_row("Estimated savings", f"[green]~${savings_est:.2f}[/green]")
        claude_cost_str = "[dim]$0.000000[/dim]"
        openai_cost_str = "[dim]$0.000000[/dim]"
        article_cost_str = "[dim]$0.000000[/dim]"
    else:
        table.add_row("Article", "Generated")
        over = article_cost > article_limit
        cost_color = "red" if over else "green"
        article_cost_str = f"[{cost_color}]${article_cost:.6f}[/{cost_color}]"
        if over:
            table.add_row(
                "Per-article target",
                f"[red]${article_cost:.4f} > ${article_limit:.2f} limit[/red]",
            )
        claude_cost_str = f"${claude_cost:.6f}"
        openai_cost_str = f"${openai_cost:.6f}"

    table.add_row("Claude cost", claude_cost_str)
    table.add_row("OpenAI cost", openai_cost_str)
    table.add_row("This article total", article_cost_str if not reused else "[dim]$0.000000[/dim]")

    monthly_pct = min(100.0, 100 * monthly_total / monthly_limit) if monthly_limit > 0 else 0.0
    monthly_color = "red" if monthly_pct >= 90 else "yellow" if monthly_pct >= 70 else "green"
    table.add_row(
        "Monthly spend",
        f"[{monthly_color}]${monthly_total:.4f} / ${monthly_limit:.2f}[/{monthly_color}]"
        f"  [dim]({monthly_pct:.1f}% used)[/dim]",
    )

    console.print()
    console.print(Panel(table, title="[bold]Cost Report[/bold]", expand=False))


def _display_sanitization(result: SanitizationResult) -> None:
    if not result.changed:
        return
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("", min_width=14)
    table.add_column("Removed", overflow="fold")
    for label in result.removed:
        table.add_row("[dim]artifact[/dim]", label)
    console.print()
    console.print(Panel(table, title="[bold yellow]Content Sanitization — Artifacts Removed[/bold yellow]", expand=False))


def _display_readiness_gate(readiness: ReadinessResult) -> None:
    title = (
        "[bold green]Publication Readiness — READY[/bold green]"
        if readiness.ready
        else "[bold red]Publication Readiness — NOT READY[/bold red]"
    )
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("", min_width=4)
    table.add_column("Check", min_width=22)
    table.add_column("Detail", overflow="fold", style="dim")
    for check in readiness.checks:
        if check.passed:
            icon = "[green]✓[/green]"
        elif check.blocking:
            icon = "[red]✗[/red]"
        else:
            icon = "[yellow]⚠[/yellow]"
        table.add_row(icon, check.name, check.detail)
    console.print()
    console.print(Panel(table, title=title, expand=False))


def _display_certification(report: CertificationReport) -> None:
    title = (
        "[bold green]Publication Certification — CERTIFIED[/bold green]"
        if report.certified
        else "[bold red]Publication Certification — NOT CERTIFIED[/bold red]"
    )
    current_section = ""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("", min_width=4)
    table.add_column("Check", min_width=26)
    table.add_column("Detail", overflow="fold", style="dim")
    for item in report.items:
        if item.section != current_section:
            table.add_row("", f"[bold]{item.section}[/bold]", "")
            current_section = item.section
        icon = "[green]✓[/green]" if item.passed else "[red]✗[/red]"
        table.add_row(icon, item.name, item.detail)
    console.print()
    console.print(Panel(table, title=title, expand=False))


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
            table.add_row(section_label, "[green]✓ Drive original[/green]")
            if asset.similarity_score is not None:
                table.add_row("  Similarity", f"{asset.similarity_score}%")
            if asset.drive_path:
                table.add_row("  Path", asset.drive_path)
            if asset.selection_reason:
                table.add_row("  Reason", asset.selection_reason[:120])
        elif asset.source == ImageSource.EDITED:
            edit_desc = asset.edit_type or "preservation edit"
            preserved = f" ({asset.preservation_estimate}% preserved)" if asset.preservation_estimate else ""
            table.add_row(section_label, f"[yellow]~ Drive photo with {edit_desc}{preserved}[/yellow]")
            if asset.drive_path:
                table.add_row("  Original", asset.drive_path)
            if asset.ai_reason:
                table.add_row("  Reason", asset.ai_reason[:120])
        else:
            table.add_row(section_label, "[dim]Not resolved[/dim]")
            if asset.selection_reason:
                table.add_row("  Reason", asset.selection_reason[:120])

    table.add_row("", "")

    drive_used   = sum(1 for _, a in resolved if a.source == ImageSource.DRIVE)
    edited_count = sum(1 for _, a in resolved if a.source == ImageSource.EDITED)
    uploaded_count = len(uploaded) if uploaded else 0
    inline_count = sum(
        1 for req, _ in (uploaded or [])
        if req.purpose == ImagePurpose.INLINE
    )
    featured_ok = any(
        req.purpose == ImagePurpose.FEATURED and meta.wordpress_media_id
        for req, meta in (uploaded or [])
    )

    table.add_row("Preservation edits", str(edited_count))
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
        is_edited = asset.source == ImageSource.EDITED

        images.append({
            **base,
            "source":                     asset.source.value,
            "drive_file_id":              asset.source_detail if is_drive else None,
            "reference_file_id":          asset.reference_file_id if is_edited else None,
            "drive_path":                 asset.drive_path,
            "similarity_score":           asset.similarity_score,
            "drive_candidates_evaluated": asset.drive_candidates_evaluated,
            "selection_reason":           asset.selection_reason,
            "vision_reasoning":           asset.vision_reasoning,
            "ai_reason":                  asset.ai_reason,
            "edit_type":                  asset.edit_type if is_edited else None,
            "edit_prompt":                asset.edit_prompt if is_edited else None,
            "preservation_estimate":      asset.preservation_estimate if is_edited else None,
            "edit_prompt_openai":         asset.source_detail if is_edited else None,
            "wordpress_media_id":         meta.wordpress_media_id if meta else None,
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    drive_assets  = [a for _, a in resolved if a.source == ImageSource.DRIVE]
    edited_assets = [a for _, a in resolved if a.source == ImageSource.EDITED]

    similarity_scores = [
        a.similarity_score for a in drive_assets + edited_assets
        if a.similarity_score is not None
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
            "drive_originals_used":       len(drive_assets),
            "preservation_edits":         len(edited_assets),
            "drive_candidates_available": drive_count,
            "avg_similarity_score":       avg_similarity,
            "featured_assigned":          featured_ok,
            "openai_edit_cost_estimate_usd": round(len(edited_assets) * 0.04, 4),
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
        _row("Drive images indexed:",       str(state.drive_indexed))
        _row("Semantic candidates:",        str(state.drive_semantic_candidates))
        _row("Images requested:",          str(state.img_requested))
        _row("Drive originals (P1):",      f"[green]{state.img_from_drive}[/green]" if state.img_from_drive else "[dim]0[/dim]")
        _row("Preservation edits (P2):",   f"[yellow]{state.img_from_edited}[/yellow]" if state.img_from_edited else "[dim]0[/dim]")
        _row(
            "Edit budget:",
            f"{state.img_from_edited}/{state.openai_budget_total} used  "
            f"([dim]{state.openai_budget_remaining} remaining[/dim])",
        )
        _row("Uploaded to WordPress:",     str(state.img_uploaded))
        _row(
            "Featured Image:",
            "[green]Assigned[/green]" if state.img_featured else "[dim]Not assigned[/dim]",
        )
        if state.edited_photos:
            console.print("   [dim]Preservation edits:[/dim]")
            for ref in state.edited_photos:
                score_str = f"score={ref.get('score', '?')}"
                drive_path = ref.get("drive_path") or "unknown"
                edit_type = ref.get("edit_type") or "edit"
                preserved = ref.get("preserved")
                preserved_str = f"  [dim]{preserved}% preserved[/dim]" if preserved else ""
                console.print(
                    f"     [yellow]~[/yellow] {ref.get('id', '?')}  "
                    f"[dim]src:[/dim] {drive_path}  [dim]({score_str}, {edit_type})[/dim]{preserved_str}"
                )
        if state.ai_reasons and not state.edited_photos:
            console.print("   [dim]Edit reasons:[/dim]")
            for reason_text in state.ai_reasons:
                console.print(f"     [dim]• {reason_text}[/dim]")
        for err in state.img_errors:
            console.print(f"   [red]✗ Upload error:[/red] {err}")
    console.print()

    # ── 4. Dual QA ────────────────────────────────────────────────
    _section("4. Dual QA")
    if not state.dual_qa_enabled:
        _row("Status:", "[dim]Skipped (QA_ENABLED=False)[/dim]")
    else:
        overall = "[green]PASS[/green]" if state.dual_qa_passed else "[red]FAIL[/red]"
        _row("Overall decision:", overall)
        _row("Review cycles:", str(state.dual_qa_iterations))
        console.print()
        console.print("   [bold dim]Article[/bold dim]")

        def _score_fmt(score: int, required: int = 90) -> str:
            color = "green" if score >= required else "red"
            return f"[{color}]{score}/100[/{color}]"

        _row("  Claude SEO:",       _score_fmt(state.dual_qa_seo_score))
        _row("  Claude Editorial:", _score_fmt(state.dual_qa_editorial_score))
        _row("  OpenAI Writing:",   _score_fmt(state.dual_qa_writing_score))
        _row("  OpenAI Authenticity:", _score_fmt(state.dual_qa_authenticity_score))
        _row("  Combined:",         f"{state.dual_qa_combined_score:.1f}/100")

        if state.dual_qa_images_reviewed > 0:
            console.print()
            console.print("   [bold dim]Preservation Edits[/bold dim]")
            _row("  Reviewed:", str(state.dual_qa_images_reviewed))
            _row(
                "  Passed:",
                f"[green]{state.dual_qa_images_passed}[/green]"
                if state.dual_qa_images_passed else "[dim]0[/dim]",
            )
            if state.dual_qa_images_failed:
                _row("  Failed:", f"[red]{state.dual_qa_images_failed}[/red]")
                for img_r in state.dual_qa_image_results:
                    if not img_r.approved:
                        console.print(
                            f"     [red]✗[/red] {img_r.image_id}  "
                            f"Claude={img_r.claude_vision_score}  "
                            f"OpenAI={img_r.openai_vision_score}"
                        )

        if state.dual_qa_rejection_reasons:
            console.print()
            console.print("   [dim]Rejection reasons:[/dim]")
            for reason in state.dual_qa_rejection_reasons:
                console.print(f"     [dim]• {reason}[/dim]")
    console.print()

    # ── 5. Authenticity Report ────────────────────────────────────
    if state.dual_qa_enabled:
        _section("5. Authenticity Report")
        auth_label = state.dual_qa_authenticity_label or "—"
        auth_color = {
            "Excellent":          "green",
            "Very Good":          "green",
            "Good":               "yellow",
            "Fair":               "yellow",
            "Needs Improvement":  "red",
        }.get(auth_label, "dim")
        _row("Overall authenticity:", f"[{auth_color}]{state.dual_qa_overall_authenticity:.1f}/100  ({auth_label})[/{auth_color}]")
        _row("Article authenticity:", f"{state.dual_qa_article_authenticity:.1f}/100")
        if state.dual_qa_image_authenticity is not None:
            _row("Image authenticity:",  f"{state.dual_qa_image_authenticity:.1f}/100")
        if state.dual_qa_authenticity_narrative:
            console.print(f"   [dim]{state.dual_qa_authenticity_narrative}[/dim]")
        console.print()

    # ── 6. QA Cost Report ─────────────────────────────────────────
    if state.dual_qa_enabled:
        _section("6. QA Cost Report")
        _row("Claude review cost:",     f"[cyan]${state.dual_qa_claude_review_cost:.4f}[/cyan]")
        _row("OpenAI review cost:",     f"[cyan]${state.dual_qa_openai_review_cost:.4f}[/cyan]")
        if state.dual_qa_revision_cost > 0:
            _row("Revision cost:",      f"[cyan]${state.dual_qa_revision_cost:.4f}[/cyan]")
        vision_total = state.dual_qa_vision_claude_cost + state.dual_qa_vision_openai_cost
        if vision_total > 0:
            _row("Vision review cost:", f"[cyan]${vision_total:.4f}[/cyan]"
                 f"  [dim](Claude ${state.dual_qa_vision_claude_cost:.4f} + OpenAI ${state.dual_qa_vision_openai_cost:.4f})[/dim]")
        _row("Total QA cost:",          f"[bold cyan]${state.dual_qa_total_cost:.4f}[/bold cyan]")
        def _fmt_s_qa(s: float) -> str:
            return f"{s:.1f}s" if s >= 0.1 else "[dim]—[/dim]"
        _row("QA time:",                _fmt_s_qa(state.dual_qa_elapsed_seconds))
        _row("Review cycles:",          str(state.dual_qa_iterations))
        if state.dual_qa_avg_cycle_seconds > 0:
            _row("Avg cycle time:",     _fmt_s_qa(state.dual_qa_avg_cycle_seconds))
        console.print()

    # ── 7. HTML Content ───────────────────────────────────────────
    _section("7. HTML Content")
    _row("Tables:",        f"[green]✓[/green] ({state.html_tables})" if state.html_tables else f"[dim]none[/dim]")
    _row("Callouts:",      f"[green]✓[/green] ({state.html_callouts})" if state.html_callouts else f"[dim]none[/dim]")
    _row("FAQ section:",   "[green]✓[/green]" if state.html_faq else "[dim]not found[/dim]")
    _row("Internal links:", str(state.html_internal_links))
    _row("External links:", str(state.html_external_links))
    console.print()

    # ── 8. Published ─────────────────────────────────────────────
    _section("8. Published")
    _row("Status:", f"[green]{state.post_status}[/green]")
    _row("Post ID:", str(state.post_id))
    _row("Slug:",    state.post_slug)
    if state.post_url:
        _row("URL:", state.post_url)
    console.print()

    # ── 9. Timing ────────────────────────────────────────────────
    _section("9. Timing")
    def _fmt_s(s: float) -> str:
        return f"{s:.1f}s" if s >= 0.1 else "[dim]—[/dim]"
    _row("Image resolution:", _fmt_s(state.t_images))
    _row("Upload:",           _fmt_s(state.t_upload))
    _row("WordPress publish:", _fmt_s(state.t_publish))
    _row("Total pipeline:",   f"[bold]{_fmt_s(state.t_total)}[/bold]")
    console.print()

    # ── 10. Costs ────────────────────────────────────────────────
    _section("10. Costs")
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

    # ── Final Summary Panel ───────────────────────────────────────
    console.print()
    summary_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    summary_table.add_column("", style="dim", min_width=24)
    summary_table.add_column("", overflow="fold")

    decision_label = f"[bold green]PUBLISHED ({state.post_status.upper()})[/bold green]"
    summary_table.add_row("Decision", decision_label)

    if state.dual_qa_enabled:
        pr_score = state.dual_qa_publication_readiness
        pr_color = "green" if pr_score >= 90 else "yellow" if pr_score >= 80 else "red"
        summary_table.add_row(
            "Publication Readiness",
            f"[bold {pr_color}]{pr_score:.1f}/100[/bold {pr_color}]",
        )
        auth_lbl = state.dual_qa_authenticity_label or "—"
        auth_col = {
            "Excellent": "green", "Very Good": "green",
            "Good": "yellow", "Fair": "yellow", "Needs Improvement": "red",
        }.get(auth_lbl, "dim")
        summary_table.add_row(
            "Authenticity",
            f"[{auth_col}]{state.dual_qa_overall_authenticity:.1f}/100  ({auth_lbl})[/{auth_col}]",
        )
        summary_table.add_row(
            "QA Cost",
            f"${state.dual_qa_total_cost:.4f}",
        )
        summary_table.add_row(
            "QA Time",
            _fmt_s(state.dual_qa_elapsed_seconds),
        )
        summary_table.add_row(
            "Review Cycles",
            str(state.dual_qa_iterations),
        )

    summary_table.add_row("Total Run Cost", f"${total_cost:.4f}")
    summary_table.add_row("Total Time",     _fmt_s(state.t_total))
    if state.post_url:
        summary_table.add_row("Published URL", state.post_url)

    console.print(Panel(
        summary_table,
        title="[bold green]FINAL SUMMARY[/bold green]",
        expand=False,
    ))
    console.print()


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
    except ArticleValidationError as exc:
        console.print(f"[red]✗ Validation error:[/red] {exc}")
        raise typer.Exit(code=1)
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
    input: Path | None = typer.Option(None, "--input", "-i", help="article.json to use for taxonomy dry-run (optional)."),
) -> None:
    """Validate WordPress: REST API, credentials, plugins. Optionally dry-run an article."""
    try:
        creds = CredentialStore(settings.credentials_dir).load(client_id, website_id)
    except CredentialNotFoundError as exc:
        console.print(f"[red]✗ Credentials not found:[/red] {exc}")
        raise typer.Exit(code=1)
    except CredentialError as exc:
        console.print(f"[red]✗ Invalid credentials:[/red] {exc}")
        raise typer.Exit(code=1)

    with WordPressService(creds) as wp:
        with console.status("[bold blue]Validating site...", spinner="dots"):
            result = wp.validate_site()

    _display_site_validation(result, creds.url)

    if input is not None and result.rest_api_reachable and result.auth_ok:
        if not input.exists():
            console.print(f"[red]✗ File not found:[/red] {input}")
            raise typer.Exit(code=1)
        try:
            article = Article.model_validate_json(input.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]✗ Cannot parse article.json:[/red] {exc}")
            raise typer.Exit(code=1)
        console.print(f"\n[dim]Running taxonomy dry-run for:[/dim] {article.title}")
        with WordPressService(creds) as wp2:
            agent = PublisherAgent(wp2)
            with console.status("[bold blue]Running dry-run...", spinner="dots"):
                report = agent.dry_run(article)
        _display_dry_run(article, report, min_score=0)

    if not result.ready:
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
        generator = OpenAIImageGenerator(settings.openai_api_key, budget=budget)
        with console.status("[bold blue]Generating image via gpt-image-1...", spinner="dots"):
            asset = generator.generate(ImageGenerationRequest(
                prompt=prompt,
                alt_text="Validation test image",
                size="1536x1024",
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


@validate_app.command("edit-photo")
def validate_edit_photo(
    drive_file_id: str | None = typer.Option(
        None, "--drive-file-id", "-d",
        help="Google Drive file ID of a specific photo to edit. "
             "Omit to auto-select a real company photograph from Drive.",
    ),
    image: Path | None = typer.Option(
        None, "--image", "-i",
        help="Local image file to edit (alternative to Drive).",
    ),
    edit: str = typer.Option(
        ..., "--edit", "-e",
        help="Minimal edit description sent verbatim to images.edit().",
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o",
        help="Directory to save original.jpg, edited.jpg, comparison.jpg.",
    ),
) -> None:
    """
    Test OpenAI images.edit() on a real company photograph with a minimal edit.

    Source priority:
      1. --image PATH         use a local file
      2. --drive-file-id ID   use a specific Drive photo
      3. (default)            auto-select a real company photograph from Drive

    The auto-selection searches the connected Drive, filters out logos, screenshots,
    and documents, and picks one genuine company photograph at random.

    Saves three files to --output-dir:
      original.jpg    — the unmodified source photo
      edited.jpg      — the result from images.edit()
      comparison.jpg  — side-by-side for visual evaluation (requires Pillow)

    Examples:
      python3 main.py validate edit-photo \\
        --edit "Convert this exact photograph to nighttime. Preserve absolutely \\
                everything except the sky, ambient lighting and shadows."

      python3 main.py validate edit-photo \\
        --drive-file-id 1ABC... \\
        --edit "Remove the red sedan parked at the right edge of the driveway."
    """
    import io as _io
    import base64 as _base64
    import random as _random

    # ── Validate prerequisites ─────────────────────────────────────────────────
    if not settings.openai_api_key:
        console.print("[red]✗ OPENAI_API_KEY not set.[/red]  Add it to .env: OPENAI_API_KEY=sk-...")
        raise typer.Exit(code=1)

    output_dir = output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Acquire source image ───────────────────────────────────────────────────
    src_bytes: bytes
    src_name: str

    if image:
        # Source 1: local file
        if not image.exists():
            console.print(f"[red]✗ File not found:[/red] {image}")
            raise typer.Exit(code=1)
        src_bytes = image.read_bytes()
        src_name = image.name
        console.print(f"[dim]Source:[/dim] {image}  ({len(src_bytes):,} bytes)")

    else:
        # Source 2 or 3: Drive (specific file ID or auto-select)
        if not (settings.google_sa_json_path and settings.google_sa_json_path.exists()):
            console.print(
                "[red]✗ Google Drive not configured.[/red]\n"
                "Set GOOGLE_SA_JSON_PATH in .env to use Drive photos.\n\n"
                "Alternatively, pass a local file with --image PATH."
            )
            raise typer.Exit(code=1)

        drive_svc = GoogleDriveService(settings.google_sa_json_path)

        if drive_file_id:
            # Source 2: specific file requested
            console.print(f"[dim]Drive file ID:[/dim] {drive_file_id}")
            try:
                with console.status("[bold blue]Downloading from Google Drive...", spinner="dots"):
                    src_bytes = drive_svc.download(drive_file_id)
                src_name = f"{drive_file_id}.jpg"
                console.print(f"[dim]Downloaded:[/dim] {len(src_bytes):,} bytes")
            except Exception as exc:
                console.print(f"[red]✗ Drive download failed:[/red] {exc}")
                raise typer.Exit(code=1)

        else:
            # Source 3: auto-select from the Drive index
            if not settings.drive_folder_id:
                console.print(
                    "[red]✗ DRIVE_FOLDER_ID not set.[/red]\n"
                    "Add it to .env, or pass --drive-file-id or --image."
                )
                raise typer.Exit(code=1)

            try:
                with console.status("[bold blue]Loading Drive index...", spinner="dots"):
                    index = DriveImageIndex(settings.drive_index_path)
                    if index.needs_sync(settings.drive_folder_id, settings.drive_sync_max_age_hours):
                        sync_stats = index.sync(drive_svc, settings.drive_folder_id)
                        console.print(
                            f"  [dim]Drive sync:[/dim] {sync_stats.images_found} images, "
                            f"{sync_stats.folders_scanned} folders "
                            f"({sync_stats.duration_seconds:.1f}s)"
                        )
                    all_candidates = index.list_all()
            except Exception as exc:
                console.print(f"[red]✗ Drive index failed:[/red] {exc}")
                raise typer.Exit(code=1)

            # Filter to genuine company photographs only.
            # Exclude: logos, screenshots, documents, non-photo mime types.
            _EXCLUDE_KEYWORDS = frozenset({
                "logo", "logos", "screenshot", "screenshots", "icon", "icons",
                "banner", "banners", "document", "documents", "doc", "flyer",
                "flyers", "brochure", "brochures", "graphic", "graphics",
                "template", "templates", "thumbnail", "thumbnails",
            })
            _PHOTO_MIME_TYPES = {"image/jpeg", "image/png", "image/heic", "image/heif", "image/webp"}

            def _is_real_photo(f) -> bool:
                if f.mime_type not in _PHOTO_MIME_TYPES:
                    return False
                combined = (f.name + " " + f.folder_path).lower()
                return not any(kw in combined for kw in _EXCLUDE_KEYWORDS)

            real_photos = [f for f in all_candidates if _is_real_photo(f)]

            if not real_photos:
                console.print(
                    "[red]✗ No suitable company photographs found in Drive.[/red]\n"
                    "Pass --drive-file-id or --image to specify one manually."
                )
                raise typer.Exit(code=1)

            selected = _random.choice(real_photos)
            console.print()
            console.print(f"[bold]Selected Drive photo:[/bold]")
            console.print(f"  {selected.name}")
            console.print()
            console.print(f"[bold]Drive File ID:[/bold]")
            console.print(f"  {selected.file_id}")
            console.print()

            try:
                with console.status("[bold blue]Downloading from Google Drive...", spinner="dots"):
                    src_bytes = drive_svc.download(selected.file_id)
                src_name = selected.name
                drive_file_id = selected.file_id
                folder = selected.folder_path.strip("/")
                drive_path = f"{folder}/{selected.name}" if folder else selected.name
                console.print(f"  [dim]Path:[/dim] {drive_path}")
                console.print(f"  [dim]Size:[/dim] {len(src_bytes):,} bytes")
            except Exception as exc:
                console.print(f"[red]✗ Drive download failed:[/red] {exc}")
                raise typer.Exit(code=1)

    # Save original
    original_path = output_dir / "original.jpg"
    original_path.write_bytes(src_bytes)

    # ── Prepare image for images.edit() ───────────────────────────────────────
    # Always use an explicit (filename, bytes, content_type) 3-tuple.
    # Raw bytes → application/octet-stream (rejected by API).
    # BytesIO with .name is unreliable — BytesIO does not officially support .name.
    from services.image_generators.openai_generator import _detect_content_type, _as_file_tuple

    try:
        from PIL import Image as _PILImage
        pil_img = _PILImage.open(_io.BytesIO(src_bytes))
        original_wh = pil_img.size
        # Resize to ≤1536×1024 if needed to stay within API limits.
        max_w, max_h = 1536, 1024
        if pil_img.width > max_w or pil_img.height > max_h:
            pil_img.thumbnail((max_w, max_h), _PILImage.LANCZOS)
            resized_bytes_buf = _io.BytesIO()
            pil_img.save(resized_bytes_buf, format="PNG")
            upload_bytes = resized_bytes_buf.getvalue()
        else:
            upload_bytes = src_bytes
        resized_wh = pil_img.size
        pil_available = True
    except ImportError:
        upload_bytes = src_bytes
        original_wh = ("?", "?")
        resized_wh = ("?", "?")
        pil_available = False

    # Detect MIME type from actual bytes (never trust extension alone).
    detected_mime, detected_ext = _detect_content_type(upload_bytes)
    upload_filename = (
        Path(src_name).stem + detected_ext if src_name else f"photo{detected_ext}"
    )
    # Explicit 3-tuple guarantees the correct Content-Type header.
    edit_tuple = _as_file_tuple(upload_bytes, upload_filename)

    # ── Build full prompt (preservation prefix + user edit) ───────────────────
    from agents.image_resolver_agent import ImageResolverAgent as _IRA
    full_prompt = _IRA._PRESERVATION_PROMPT_PREFIX + edit

    # ── Print what we're sending ───────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[dim]{_IRA._PRESERVATION_PROMPT_PREFIX}[/dim][bold]{edit}[/bold]",
        title="[dim]Full prompt sent to images.edit()[/dim]",
        border_style="dim",
        expand=False,
    ))
    console.print(
        f"  [dim]Original size:[/dim] {original_wh[0]}×{original_wh[1]}"
        + (f"  →  [dim]Sent:[/dim] {resized_wh[0]}×{resized_wh[1]}" if pil_available and resized_wh != original_wh else "")
    )
    console.print(
        f"  [dim]Upload filename:[/dim] {upload_filename}  "
        f"[dim]Content-Type:[/dim] {detected_mime}  "
        f"[dim]Model:[/dim] gpt-image-1  [dim]Quality:[/dim] high"
    )
    console.print()

    # ── Call images.edit() ─────────────────────────────────────────────────────
    import openai as _openai
    _client = _openai.OpenAI(api_key=settings.openai_api_key)

    _INPUT_PER_M = 2.50   # gpt-image-1 approximate text token rate
    _OUTPUT_PER_M = 10.00

    t0 = time.perf_counter()
    try:
        with console.status("[bold blue]Calling images.edit()...", spinner="dots"):
            response = _client.images.edit(
                image=edit_tuple,       # (filename, bytes, content_type) — never octet-stream
                prompt=full_prompt,
                model="gpt-image-1",
                size="1536x1024",
                quality="high",
                n=1,
            )
    except Exception as exc:
        console.print(f"[red]✗ images.edit() failed:[/red] {exc}")
        raise typer.Exit(code=1)

    elapsed = time.perf_counter() - t0

    # ── Decode result ──────────────────────────────────────────────────────────
    result_b64 = response.data[0].b64_json
    if not result_b64:
        console.print("[red]✗ No image data in response[/red]")
        raise typer.Exit(code=1)

    edited_bytes = _base64.b64decode(result_b64)

    # ── Compute cost ───────────────────────────────────────────────────────────
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else 0
    cost_usd = round(
        input_tokens * _INPUT_PER_M / 1_000_000
        + output_tokens * _OUTPUT_PER_M / 1_000_000,
        6,
    )

    # ── Save outputs ───────────────────────────────────────────────────────────
    edited_path = output_dir / "edited.jpg"
    edited_path.write_bytes(edited_bytes)

    comparison_path = output_dir / "comparison.jpg"
    if pil_available:
        try:
            orig_pil = _PILImage.open(original_path).convert("RGB")
            edit_pil = _PILImage.open(_io.BytesIO(edited_bytes)).convert("RGB")
            # Resize both to 768×512 for side-by-side
            thumb_w, thumb_h = 768, 512
            orig_thumb = orig_pil.resize((thumb_w, thumb_h), _PILImage.LANCZOS)
            edit_thumb = edit_pil.resize((thumb_w, thumb_h), _PILImage.LANCZOS)
            comparison = _PILImage.new("RGB", (thumb_w * 2 + 12, thumb_h + 40), (30, 30, 30))
            comparison.paste(orig_thumb, (0, 40))
            comparison.paste(edit_thumb, (thumb_w + 12, 40))
            # Add simple text labels via PIL ImageDraw
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 10), "ORIGINAL", fill=(200, 200, 200), font=font)
            draw.text((thumb_w + 22, 10), "EDITED", fill=(200, 200, 200), font=font)
            comparison.save(comparison_path, "JPEG", quality=92)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] comparison.jpg not generated: {exc}")
            comparison_path = None
    else:
        comparison_path = None

    # ── Report ─────────────────────────────────────────────────────────────────
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Field", style="dim", min_width=20)
    table.add_column("Value", overflow="fold")
    table.add_row("Status", "[green]✓ Edit complete[/green]")
    table.add_row("Time", f"{elapsed:.1f}s")
    table.add_row("Original", str(original_path))
    table.add_row("Edited", str(edited_path))
    if comparison_path:
        table.add_row("Comparison", str(comparison_path))
    table.add_row("Edited size", f"{len(edited_bytes):,} bytes")
    table.add_row("Input tokens", str(input_tokens))
    table.add_row("Output tokens", str(output_tokens))
    table.add_row("Total tokens", str(total_tokens))
    table.add_row("Estimated cost", f"${cost_usd:.6f} USD" if cost_usd else "n/a (usage not returned)")

    console.print(Panel(
        table,
        title="[bold green]✓ images.edit() Validated[/bold green]",
        expand=False,
    ))

    console.print(
        "\n[dim]Open the comparison to evaluate identity preservation:[/dim]\n"
        f"  open {comparison_path or edited_path}"
    )


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
                pub_status = data.get("publishing", {}).get("status", "")
                if title and pub_status == "publish":
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

    user_message = "\n".join(context_lines)
    raw = claude.generate(
        system,
        [{"role": "user", "content": user_message}],
        thinking=False,
        model=settings.topic_model,
        label="topics:suggest",
    )
    topics = re.findall(r'^\d+\.\s+(.+)$', raw.strip(), flags=re.MULTILINE)
    return topics, raw, user_message


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
            topics, _, _ = _suggest_topics(
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
        img_parts.append(f"{state.img_from_drive} Drive original")
    if state.img_from_edited:
        img_parts.append(f"{state.img_from_edited} preservation edit")
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
    words: int = typer.Option(settings.default_word_count, "--words", "-w", min=300, max=950, help="Target word count (700–900; default 800, hard cap 950)."),
    language: ArticleLanguage = typer.Option(ArticleLanguage.EN, "--language", "-l", help="Article language (always English)."),
    status: str = typer.Option("draft", "--status", help="WordPress post status: draft or publish."),
    min_score: int = typer.Option(settings.seo_qa_min_score, "--min-score", min=0, max=100, help="Minimum SEO quality score."),
    no_image: bool = typer.Option(False, "--no-image", help="Skip image resolution and upload."),
    no_links: bool = typer.Option(False, "--no-links", help="Skip internal link enrichment."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory (overrides OUTPUT_DIR in .env)."),
    suggest_n: int = typer.Option(10, "--suggest-n", help="How many topic ideas to generate before picking the first."),
    dev: bool = typer.Option(False, "--dev", help="Dev mode: generate only. Skips QA, images, links, and WordPress."),
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

    # ── Pre-flight: credentials + city/state ──────────────────────
    website_url: str | None = None
    try:
        _creds = CredentialStore(settings.credentials_dir).load(resolved_client, resolved_website)
        website_url = _creds.url
    except Exception as _creds_exc:
        console.print(
            f"[red]Pre-flight failed:[/red] No credentials found for "
            f"{resolved_client}/{resolved_website}. "
            "Run [bold]seo-agent import-sites[/bold] first."
        )
        raise typer.Exit(code=1)

    # Warn (don't fail) if city/state cannot be resolved — resolver will try WP/heuristics
    _profile_city: str | None = None
    _profile_service: str | None = None
    try:
        from models.site_profile import SiteProfile as _SP
        _profile_path = settings.profiles_dir / resolved_client / resolved_website / "site.json"
        if _profile_path.exists():
            _sp = _SP.model_validate_json(_profile_path.read_text(encoding="utf-8"))
            _profile_city = _sp.city or None
            _profile_service = _sp.primary_service or _sp.niche or None
    except Exception:
        pass
    if not city and not _profile_city:
        console.print(
            "[yellow]Warning:[/yellow] No city/state in profile or --city flag. "
            "BusinessContextResolver will attempt to detect from WordPress."
        )

    console.print()
    console.print(Panel(
        f"[bold]SEO Agent — Auto Publish[/bold]\n"
        f"[dim]{resolved_client} / {resolved_website}[/dim]",
        expand=False,
    ))
    console.print()

    # ── Start call tracer for this run ────────────────────────────
    import services.call_tracer as _call_tracer
    _call_tracer.start()

    # ── Step 1: Suggest topics ────────────────────────────────────
    _TOPIC_PLACEHOLDER_RE = re.compile(
        r'\[(?:City|Service|Keyword|Topic|TOPIC|Location|Business|State|Country|Name|Date|Year|Niche)\]',
        re.IGNORECASE,
    )

    with console.status(f"[bold green]Generating {suggest_n} topic ideas...", spinner="dots"):
        try:
            topics, topic_raw, topic_prompt = _suggest_topics(
                resolved_client, resolved_website,
                service=service or _profile_service,
                city=city or _profile_city,
                language=language,
                n=suggest_n,
                base_dir=base_dir,
            )
        except ClaudeAPIError as exc:
            console.print(f"[red]Error:[/red] Topic suggestion failed: {exc}")
            raise typer.Exit(code=1)

    if not topics:
        console.print("[red]Error:[/red] No topics were generated. Try again.")
        console.print("\n[dim]── Prompt sent ──[/dim]")
        console.print(topic_prompt)
        console.print("\n[dim]── Raw response ──[/dim]")
        console.print(topic_raw)
        raise typer.Exit(code=1)

    console.print(f"[dim]Generated {len(topics)} topic ideas:[/dim]")
    selected_topic: str | None = None
    for i, t in enumerate(topics, 1):
        has_placeholder = bool(_TOPIC_PLACEHOLDER_RE.search(t))
        if has_placeholder:
            console.print(f"  [dim]{i}. [red]INVALID[/red] (placeholder) — {t}[/dim]")
        else:
            if selected_topic is None:
                console.print(f"  {i}. [green]VALID[/green] ← selected — {t}")
                selected_topic = t
            else:
                console.print(f"  {i}. [green]VALID[/green] — {t}")

    if selected_topic is None:
        console.print("\n[red]Error:[/red] All generated topics contain placeholders. Cannot auto-select.")
        console.print("\n[dim]── Prompt sent ──[/dim]")
        console.print(topic_prompt)
        console.print("\n[dim]── Raw response ──[/dim]")
        console.print(topic_raw)
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Selected:[/bold cyan] {selected_topic}")
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
        website_url=website_url,
    )
    request = BusinessContextResolver(settings.profiles_dir).resolve(
        resolved_client, resolved_website, request
    )

    article_path = _execute_generation(request, tenant, base_dir)
    article = Article.model_validate_json(article_path.read_text(encoding="utf-8"))
    _save_last_article(article)

    if dev:
        console.print(
            "[yellow]Dev mode:[/yellow] QA, images, links, and WordPress skipped. "
            "Article saved locally."
        )
        console.print(f"[dim]Saved:[/dim] {article_path}")
        _tracer = _call_tracer.get()
        if _tracer and _tracer.records:
            console.print(_tracer.summary())
        return

    # ── Steps 3–6: SEO QA + Image resolution + Publish ───────────
    updated, state = _run_publish_flow(
        article_path, article,
        status=status,
        min_score=min_score,
        no_image=no_image,
        no_links=no_links,
        show_pipeline_report=True,
        event_type="autopublish",
    )

    # ── Summary ───────────────────────────────────────────────────
    _display_autopublish_summary(selected_topic, article, updated, state)

    # ── LLM call cost profile ──────────────────────────────────────
    _tracer = _call_tracer.get()
    if _tracer and _tracer.records:
        console.print(_tracer.summary())


@app.command()
def republish(
    status: str = typer.Option("draft", "--status", help="WordPress post status: draft or publish."),
    no_links: bool = typer.Option(False, "--no-links", help="Skip internal link enrichment."),
    post_id: int | None = typer.Option(None, "--post-id", help="Update an existing WordPress post by ID instead of creating a new one."),
) -> None:
    """Republish the last generated article without re-running generation, QA, or image resolution."""

    if not _LAST_ARTICLE_PATH.exists():
        console.print(
            "[red]Error:[/red] No saved article found. "
            f"Run [bold]autopublish[/bold] first (expected at {_LAST_ARTICLE_PATH})."
        )
        raise typer.Exit(code=1)

    try:
        article = Article.model_validate_json(
            _LAST_ARTICLE_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] Could not parse {_LAST_ARTICLE_PATH}: {exc}")
        raise typer.Exit(code=1)

    # Strip image_plans so the mandatory-image gate in PublisherAgent does not
    # block republish — images were resolved (or skipped) in the original run.
    article = article.model_copy(update={"image_plans": []})

    console.print()
    console.print(Panel(
        f"[bold]SEO Agent — Republish[/bold]\n"
        f"[dim]{article.tenant.client_id} / {article.tenant.website_id}[/dim]\n"
        f"[dim]{article.title}[/dim]",
        expand=False,
    ))
    console.print()

    import services.call_tracer as _call_tracer
    _call_tracer.start()

    _run_publish_flow(
        _LAST_ARTICLE_PATH,
        article,
        status=status,
        no_image=True,
        no_links=no_links,
        no_qa=True,
        post_id=post_id,
        show_pipeline_report=True,
        event_type="republish",
    )

    _tracer = _call_tracer.get()
    if _tracer and _tracer.records:
        console.print(_tracer.summary())


# ── BigQuery health-check helpers ─────────────────────────────────────────────
# Used exclusively by test_bigquery. Defined at module level for testability.

# Columns Python writes per table. NULLABLE stubs in the DDL that Python never
# populates (e.g. canonical_client) are intentionally absent. Types use the
# canonical BigQuery names; _bqhc_norm_type handles alias comparison.
_BQ_EXPECTED_SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "articles_published": [
        ("article_id",         "STRING",    "REQUIRED"),
        ("client",             "STRING",    "REQUIRED"),
        ("website",            "STRING",    "REQUIRED"),
        ("canonical_client",   "STRING",    "NULLABLE"),
        ("topic",              "STRING",    "REQUIRED"),
        ("title",              "STRING",    "REQUIRED"),
        ("slug",               "STRING",    "REQUIRED"),
        ("url",                "STRING",    "NULLABLE"),
        ("publish_date",       "TIMESTAMP", "REQUIRED"),
        ("word_count",         "INTEGER",   "REQUIRED"),
        ("reading_time",       "INTEGER",   "REQUIRED"),
        ("focus_keyword",      "STRING",    "NULLABLE"),
        ("category",           "STRING",    "NULLABLE"),
        ("seo_score",          "INTEGER",   "REQUIRED"),
        ("editorial_score",    "INTEGER",   "REQUIRED"),
        ("writing_score",      "INTEGER",   "REQUIRED"),
        ("authenticity_score", "INTEGER",   "REQUIRED"),
        ("total_cost_usd",     "NUMERIC",   "REQUIRED"),
        ("claude_cost_usd",    "NUMERIC",   "REQUIRED"),
        ("openai_cost_usd",    "NUMERIC",   "REQUIRED"),
        ("reuse",              "BOOLEAN",   "REQUIRED"),
        ("reuse_similarity",   "FLOAT",     "REQUIRED"),
        ("generation_time",    "FLOAT",     "REQUIRED"),
        ("model_name",         "STRING",    "REQUIRED"),
        ("prompt_version",     "STRING",    "REQUIRED"),
        ("event_type",         "STRING",    "REQUIRED"),
        ("environment",        "STRING",    "REQUIRED"),
        ("git_commit",         "STRING",    "NULLABLE"),
        ("pipeline_version",   "STRING",    "NULLABLE"),
    ],
    "qa_results": [
        ("article_id",                "STRING",  "REQUIRED"),
        ("canonical_client",          "STRING",  "NULLABLE"),
        ("approved",                  "BOOLEAN", "REQUIRED"),
        ("revision_cycles",           "INTEGER", "REQUIRED"),
        ("claude_seo_score",          "INTEGER", "REQUIRED"),
        ("claude_editorial_score",    "INTEGER", "REQUIRED"),
        ("openai_writing_score",      "INTEGER", "REQUIRED"),
        ("openai_authenticity_score", "INTEGER", "REQUIRED"),
        ("overall_pass",              "BOOLEAN", "REQUIRED"),
        ("environment",               "STRING",  "REQUIRED"),
        ("git_commit",                "STRING",  "NULLABLE"),
    ],
    "llm_costs": [
        ("timestamp",        "TIMESTAMP", "REQUIRED"),
        ("article_id",       "STRING",    "NULLABLE"),
        ("canonical_client", "STRING",    "NULLABLE"),
        ("event_type",       "STRING",    "REQUIRED"),
        ("environment",  "STRING",    "REQUIRED"),
        ("git_commit",   "STRING",    "NULLABLE"),
        ("system",       "STRING",    "REQUIRED"),
        ("stage",        "STRING",    "REQUIRED"),
        ("provider",     "STRING",    "REQUIRED"),
        ("model",        "STRING",    "REQUIRED"),
        ("input_tokens", "INTEGER",   "REQUIRED"),
        ("output_tokens","INTEGER",   "REQUIRED"),
        ("cost_usd",     "NUMERIC",   "REQUIRED"),
        ("success",      "BOOLEAN",   "REQUIRED"),
    ],
}


def _bqhc_norm_type(field_type: str) -> str:
    """Normalize BigQuery field_type aliases for schema comparison.

    The BigQuery API accepts INT64/INTEGER, FLOAT64/FLOAT, BOOL/BOOLEAN as
    synonyms but always returns the canonical name (INTEGER, FLOAT, BOOLEAN)
    when reading back a table's schema.  This function normalises both sides so
    comparison is alias-safe regardless of how the table was originally created.
    """
    return {"INT64": "INTEGER", "FLOAT64": "FLOAT", "BOOL": "BOOLEAN"}.get(
        field_type.upper(), field_type.upper()
    )


def _bqhc_extract_error(exc: Exception) -> dict:
    """Return structured diagnostic fields from any exception.

    Extracts BigQuery-specific metadata (HTTP status, BQ error reason, field
    location) when the exception carries a Google API error payload.  Falls
    back gracefully for generic Python exceptions.
    """
    info: dict = {
        "type": type(exc).__name__,
        "message": str(exc),
        "http_status": None,
        "bq_reason": None,
        "location": None,
    }
    _resp = getattr(exc, "response", None)
    if _resp is not None:
        info["http_status"] = str(getattr(_resp, "status_code", "?"))
    _bq_errors = getattr(exc, "errors", None)
    if isinstance(_bq_errors, (list, tuple)) and _bq_errors:
        _e0 = _bq_errors[0] if isinstance(_bq_errors[0], dict) else {}
        info["bq_reason"] = _e0.get("reason")
        info["location"] = _e0.get("location")
    return info


@app.command("test-bigquery")
def test_bigquery(
    cleanup: bool = typer.Option(
        False, "--cleanup",
        help="Delete integration_test rows from all three tables after the health check.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose",
        help="Print SQL queries, row data, table schemas, and per-step detail.",
    ),
) -> None:
    """Full BigQuery health check: credentials, dataset, tables, schema, writes, reads, and optional cleanup."""
    import os
    import sys
    import traceback as _tb
    import json as _json
    from datetime import datetime, timezone
    from uuid import uuid4

    _ok   = "[bold green]✓[/bold green]"
    _fail = "[bold red]✗[/bold red]"
    _warn = "[bold yellow]⚠[/bold yellow]"
    _SEP  = "=" * 42
    _summary: list[tuple[str, bool, float]] = []  # (label, passed, elapsed_s)
    _exit_code = 0

    # ── Inner helpers (closures over locals) ──────────────────────────────────

    def _vprint(msg: str) -> None:
        if verbose:
            console.print(f"[dim]  {msg}[/dim]")

    def _section_ok(label: str, elapsed: float) -> None:
        _summary.append((label, True, elapsed))

    def _section_fail(label: str, elapsed: float) -> None:
        _summary.append((label, False, elapsed))

    def _print_step_error(step: str, exc: Exception) -> None:
        info = _bqhc_extract_error(exc)
        console.print(f"      [bold]Step:[/bold]             {step}")
        console.print(f"      [bold]Exception type:[/bold]   {info['type']}")
        _preview = info["message"][:400] + ("…" if len(info["message"]) > 400 else "")
        console.print(f"      [bold]Exception message:[/bold] [red]{_preview}[/red]")
        if info["http_status"]:
            console.print(f"      [bold]HTTP status:[/bold]      {info['http_status']}")
        if info["bq_reason"]:
            console.print(f"      [bold]BQ reason:[/bold]        {info['bq_reason']}")
        if info["location"]:
            console.print(f"      [bold]Location:[/bold]         {info['location']}")
        if verbose:
            _trace = _tb.format_exc()
            if _trace.strip() and _trace.strip() != "NoneType: None":
                console.print(f"[dim]{_trace}[/dim]")

    def _print_final_report(total_s: float) -> None:
        if not _summary:
            return
        _max_w  = max(len(lbl) for lbl, _, _ in _summary)
        _col_w  = _max_w + 6
        console.print()
        console.print(f"[bold]{_SEP}[/bold]")
        console.print("[bold]  BigQuery Health Check[/bold]")
        console.print(f"[bold]{_SEP}[/bold]")
        for _lbl, _passed, _ in _summary:
            console.print(f"  {_ok if _passed else _fail} {_lbl}")
        console.print(f"  {'─' * 38}")
        for _lbl, _, _t in _summary:
            _dots = "." * (_col_w - len(_lbl))
            console.print(f"  {_lbl} {_dots} {_t:.2f} s")
        _total_dots = "." * (_col_w - len("Total"))
        console.print(f"  {'─' * 38}")
        console.print(f"  Total {_total_dots} {total_s:.2f} s")
        console.print(f"  {'─' * 38}")
        console.print(f"  Environment:  {_env_name}")
        console.print(f"  Project:      {_env_project}")
        console.print(f"[bold]{_SEP}[/bold]")
        _all_ok = all(p for _, p, _ in _summary)
        if _all_ok:
            console.print("[bold green]  BigQuery integration PASSED[/bold green]")
        else:
            console.print("[bold red]  BigQuery integration FAILED[/bold red]")
        console.print(f"[bold]{_SEP}[/bold]")
        console.print()

    # ── Environment header ────────────────────────────────────────────────────
    _env_name    = os.environ.get("SEO_AGENT_ENV", "prod")
    _env_project = "rightidea-cortex"
    _env_dataset = "seo_content"
    _env_git     = "UNKNOWN"
    _env_pv      = "unknown"
    _py_ver      = sys.version.split()[0]

    try:
        from services.bq_sink_service import (
            _GCP_PROJECT as _hc_proj,
            _BQ_DATASET  as _hc_ds,
            _GIT_COMMIT  as _hc_git,
            _PIPELINE_VERSION as _hc_pv,
        )
        _env_project = _hc_proj
        _env_dataset = _hc_ds
        _env_git     = _hc_git[:8] if _hc_git else "UNKNOWN"
        _env_pv      = _hc_pv or "unknown"
    except Exception:
        pass  # fallback defaults already set

    console.print()
    console.print(f"[bold]{_SEP}[/bold]")
    console.print("[bold]  SEO-Agent BigQuery Health Check[/bold]")
    console.print(f"[bold]{_SEP}[/bold]")
    console.print()
    console.print(f"  Project:          [cyan]{_env_project}[/cyan]")
    console.print(f"  Dataset:          [cyan]{_env_dataset}[/cyan]")
    console.print(f"  Environment:      [cyan]{_env_name}[/cyan]")
    console.print(f"  Pipeline Version: [cyan]{_env_pv}[/cyan]")
    console.print(f"  Git Commit:       [cyan]{_env_git}[/cyan]")
    console.print(f"  Python:           [cyan]{_py_ver}[/cyan]")
    console.print()
    console.print(f"[bold]{_SEP}[/bold]")
    console.print()

    _t_total = time.perf_counter()

    try:
        # ── 1. Credentials ────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            console.print(f"  {_fail} Credentials: GOOGLE_APPLICATION_CREDENTIALS is not set")
            console.print("[dim]      Set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json[/dim]")
            _section_fail("Credentials", time.perf_counter() - _t0)
            raise typer.Exit(code=1)
        if not Path(creds_path).exists():
            console.print(f"  {_fail} Credentials: file not found: {creds_path}")
            _section_fail("Credentials", time.perf_counter() - _t0)
            raise typer.Exit(code=1)
        console.print(f"  {_ok} Credentials found")

        try:
            from services.bq_sink_service import (
                BqSinkService, _GCP_PROJECT, _BQ_DATASET,
                _TABLE_ARTICLES, _TABLE_QA, _TABLE_COSTS,
            )
        except ImportError as exc:
            console.print(f"  {_fail} Credentials (google-cloud-bigquery unavailable)")
            _print_step_error("Credentials import", exc)
            console.print("[dim]      Install: pip install 'google-cloud-bigquery>=3.0'[/dim]")
            _section_fail("Credentials", time.perf_counter() - _t0)
            raise typer.Exit(code=1)
        _section_ok("Credentials", time.perf_counter() - _t0)

        # ── 2. Authentication ─────────────────────────────────────────────────
        _t0 = time.perf_counter()
        try:
            _svc = BqSinkService()
            if _svc._client is None:
                console.print(f"  {_fail} Authentication")
                _print_step_error(
                    "Authentication",
                    RuntimeError(_svc._init_error or "BqSinkService client is None"),
                )
                _section_fail("Authentication", time.perf_counter() - _t0)
                raise typer.Exit(code=1)
            bq = _svc._client
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"  {_fail} Authentication")
            _print_step_error("Authentication", exc)
            _section_fail("Authentication", time.perf_counter() - _t0)
            raise typer.Exit(code=1)
        console.print(f"  {_ok} Authentication succeeded")
        console.print(f"  {_ok} Connected to project: {_GCP_PROJECT}")
        _section_ok("Authentication", time.perf_counter() - _t0)
        console.print()

        # ── 3. Dataset ────────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        try:
            from google.cloud import bigquery as _bq
            bq.get_dataset(_bq.DatasetReference(_GCP_PROJECT, _BQ_DATASET))
        except Exception as exc:
            console.print(f"  {_fail} Dataset: {_GCP_PROJECT}.{_BQ_DATASET}")
            _print_step_error("Dataset", exc)
            _section_fail("Dataset", time.perf_counter() - _t0)
            raise typer.Exit(code=1)
        console.print(f"  {_ok} Dataset found: {_GCP_PROJECT}.{_BQ_DATASET}")
        _section_ok("Dataset", time.perf_counter() - _t0)
        console.print()

        # ── 4. Tables ─────────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        _tables = [
            ("articles_published", _TABLE_ARTICLES),
            ("qa_results",         _TABLE_QA),
            ("llm_costs",          _TABLE_COSTS),
        ]
        _bq_tables: dict = {}
        for _tname, _tid in _tables:
            try:
                _bq_tables[_tname] = bq.get_table(_tid)
                console.print(f"  {_ok} Table found: {_tname}")
                if verbose:
                    for _f in _bq_tables[_tname].schema:
                        _vprint(
                            f"    {_f.name:<32} {_f.field_type:<10} {_f.mode}"
                        )
            except Exception as exc:
                console.print(f"  {_fail} Missing table: {_tname}")
                _print_step_error(f"Tables — {_tname}", exc)
                console.print("[dim]      Hint: Run docs/bq_schema.sql first.[/dim]")
                _section_fail("Tables", time.perf_counter() - _t0)
                raise typer.Exit(code=1)
        _section_ok("Tables", time.perf_counter() - _t0)
        console.print()

        # ── 5. Schema validation ──────────────────────────────────────────────
        # Compares against _BQ_EXPECTED_SCHEMA (module-level constant).
        # Validates only the columns Python writes; DDL-only stubs are skipped.
        _t0 = time.perf_counter()

        _schema_ok = True
        for _tname, _bq_table in _bq_tables.items():
            _actual = {f.name: (f.field_type, f.mode) for f in _bq_table.schema}
            _mismatches: list[str] = []
            for _col, _exp_type, _exp_mode in _BQ_EXPECTED_SCHEMA[_tname]:
                if _col not in _actual:
                    _mismatches.append(f"column '{_col}' missing from table")
                    continue
                _act_type, _act_mode = _actual[_col]
                if _bqhc_norm_type(_act_type) != _bqhc_norm_type(_exp_type):
                    _mismatches.append(
                        f"'{_col}': type expected {_exp_type}, found {_act_type}"
                    )
                if _act_mode != _exp_mode:
                    _mismatches.append(
                        f"'{_col}': mode expected {_exp_mode}, found {_act_mode}"
                    )
            if _mismatches:
                console.print(f"  {_fail} Schema mismatch: {_tname}")
                for _m in _mismatches:
                    console.print(f"      [red]{_m}[/red]")
                _schema_ok = False
            else:
                console.print(f"  {_ok} {_tname} schema matches")

        _elapsed = time.perf_counter() - _t0
        if not _schema_ok:
            _section_fail("Schema", _elapsed)
            raise typer.Exit(code=1)
        _section_ok("Schema", _elapsed)
        console.print()

        # ── 6. Write test ─────────────────────────────────────────────────────
        # Row dicts mirror exactly what BqSinkService writes.
        # insert_rows_json() is called directly (bypassing the fire-and-forget
        # wrapper) so BigQuery streaming errors surface as real exceptions.
        _t0      = time.perf_counter()
        _test_id = str(uuid4())
        _now     = datetime.now(tz=timezone.utc).isoformat()
        console.print(f"  [dim]test article_id: {_test_id}[/dim]")
        _vprint(f"Timestamp: {_now}")
        console.print()

        _art_row: dict = {
            "article_id":         _test_id,
            "client":             "__TEST__",
            "website":            "__TEST__",
            "canonical_client":   "__TEST__",
            "topic":              "BigQuery Health Check",
            "title":              "BigQuery Health Check",
            "slug":               "bigquery-health-check",
            "url":                None,
            "publish_date":       _now,
            "word_count":         0,
            "reading_time":       0,
            "focus_keyword":      None,
            "category":           None,
            "seo_score":          0,
            "editorial_score":    0,
            "writing_score":      0,
            "authenticity_score": 0,
            "total_cost_usd":     0.0,
            "claude_cost_usd":    0.0,
            "openai_cost_usd":    0.0,
            "reuse":              False,
            "reuse_similarity":   0.0,
            "generation_time":    0.0,
            "model_name":         "__TEST__",
            "prompt_version":     "__TEST__",
            "event_type":         "integration_test",
            "environment":        "__TEST__",
            "git_commit":         None,
            "pipeline_version":   None,
        }
        _qa_row: dict = {
            "article_id":                _test_id,
            "canonical_client":          "__TEST__",
            "approved":                  True,
            "revision_cycles":           1,
            "claude_seo_score":          0,
            "claude_editorial_score":    0,
            "openai_writing_score":      0,
            "openai_authenticity_score": 0,
            "overall_pass":              True,
            "environment":               "__TEST__",
            "git_commit":                None,
        }
        _cost_row: dict = {
            "timestamp":         _now,
            "article_id":        _test_id,
            "canonical_client":  "__TEST__",
            "event_type":        "integration_test",
            "environment":   "__TEST__",
            "git_commit":    None,
            "system":        "__TEST__",
            "stage":         "__TEST__",
            "provider":      "other",
            "model":         "__TEST__",
            "input_tokens":  0,
            "output_tokens": 0,
            "cost_usd":      0.0,
            "success":       True,
        }

        _write_ok = True
        for _label, _tid, _row in [
            ("insert_article",    _TABLE_ARTICLES, _art_row),
            ("insert_qa_results", _TABLE_QA,       _qa_row),
            ("insert_llm_costs",  _TABLE_COSTS,    _cost_row),
        ]:
            _vprint(f"INSERT {_tid}")
            _vprint("Row: " + _json.dumps(_row, default=str))
            try:
                _errs = bq.insert_rows_json(_tid, [_row])
                if _errs:
                    raise Exception(_errs)
                console.print(f"  {_ok} {_label} passed")
            except Exception as exc:
                console.print(f"  {_fail} {_label} failed")
                _print_step_error(f"Write — {_label}", exc)
                _write_ok = False

        _elapsed = time.perf_counter() - _t0
        if not _write_ok:
            _section_fail("Write", _elapsed)
            raise typer.Exit(code=1)
        _section_ok("Write", _elapsed)
        console.print()

        # ── 7. Read test ──────────────────────────────────────────────────────
        # BigQuery streaming inserts are immediately queryable via SELECT.
        # A 0-row result means the streaming buffer has not flushed yet — this
        # is a known BigQuery behaviour, NOT a write failure.  The health check
        # passes as long as the SELECT query itself executes without error.
        _t0          = time.perf_counter()
        _read_warns  = False
        _read_errors = False
        for _tname, _tid in _tables:
            _q = (
                f"SELECT COUNT(*) AS cnt FROM `{_tid}` "
                "WHERE article_id = @tid"
            )
            _cfg = _bq.QueryJobConfig(
                query_parameters=[_bq.ScalarQueryParameter("tid", "STRING", _test_id)]
            )
            _vprint(f"SQL: {_q}")
            _vprint(f"  @tid = {_test_id}")
            try:
                _cnt = next(iter(bq.query(_q, job_config=_cfg).result())).cnt
                if _cnt < 1:
                    console.print(f"  {_warn} Read: {_tname} (0 rows — streaming buffer)")
                    _read_warns = True
                else:
                    console.print(f"  {_ok} Read: {_tname} ({_cnt} row)")
            except Exception as exc:
                console.print(f"  {_fail} Read: {_tname}")
                _print_step_error(f"Read — {_tname}", exc)
                _read_errors = True

        _elapsed = time.perf_counter() - _t0
        if _read_errors:
            _section_fail("Read", _elapsed)
            raise typer.Exit(code=1)
        _section_ok("Read", _elapsed)
        if _read_warns:
            console.print(
                f"\n  {_warn} [dim]BigQuery streaming inserts may remain in the\n"
                "  streaming buffer for up to several minutes before becoming\n"
                "  queryable via SELECT.  The write test confirmed data was\n"
                "  accepted by BigQuery — 0-row reads are not a failure.[/dim]"
            )
        console.print()

        # ── 8. Cleanup (optional) ─────────────────────────────────────────────
        # DELETE on streaming-buffer rows may report 0 affected rows until the
        # buffer commits to table storage (~90 s).  Re-run --cleanup if needed.
        if cleanup:
            _t0        = time.perf_counter()
            _cleanup_ok = True
            for _tname, _tid in _tables:
                _del_q = f"DELETE FROM `{_tid}` WHERE article_id = @tid"
                _del_cfg = _bq.QueryJobConfig(
                    query_parameters=[
                        _bq.ScalarQueryParameter("tid", "STRING", _test_id)
                    ]
                )
                _vprint(f"SQL: {_del_q}")
                _vprint(f"  @tid = {_test_id}")
                try:
                    _n = (
                        bq.query(_del_q, job_config=_del_cfg)
                        .result()
                        .num_dml_affected_rows or 0
                    )
                    if _n == 0:
                        console.print(
                            f"  {_warn} Cleanup: {_tname} — 0 rows deleted\n"
                            "      (streaming buffer not yet committed; "
                            "re-run --cleanup in ~90 s)"
                        )
                    else:
                        console.print(f"  {_ok} Cleanup: {_tname} ({_n} row deleted)")
                except Exception as exc:
                    console.print(f"  {_fail} Cleanup: {_tname}")
                    _print_step_error(f"Cleanup — {_tname}", exc)
                    _cleanup_ok = False
            _elapsed = time.perf_counter() - _t0
            if _cleanup_ok:
                _section_ok("Cleanup", _elapsed)
            else:
                _section_fail("Cleanup", _elapsed)
                raise typer.Exit(code=1)
            console.print()
        else:
            console.print(
                f"  [dim]Test rows left for inspection (article_id={_test_id}).\n"
                f"  Re-run with --cleanup to delete, or execute manually:\n"
                f"    DELETE FROM `{_TABLE_ARTICLES}` WHERE article_id = '{_test_id}';\n"
                f"    DELETE FROM `{_TABLE_QA}` WHERE article_id = '{_test_id}';\n"
                f"    DELETE FROM `{_TABLE_COSTS}` WHERE article_id = '{_test_id}';[/dim]"
            )
            console.print()

    except typer.Exit as _e:
        _exit_code = _e.code

    # Final report — always shown whenever at least one section completed.
    _print_final_report(time.perf_counter() - _t_total)

    if _exit_code:
        raise typer.Exit(code=_exit_code)


_TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ── Bulk import helpers ───────────────────────────────────────────────────────

def _load_import_rows(path: Path) -> list[dict[str, str]]:
    """Read a .csv or .xlsx file and return rows as dicts with stripped string values."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read .xlsx files. "
                "Install it:  pip install openpyxl"
            ) from exc
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        raw_rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not raw_rows:
            return []
        headers = [str(h).strip() if h is not None else "" for h in raw_rows[0]]
        result = []
        for row in raw_rows[1:]:
            if all(v is None for v in row):
                continue
            result.append({
                headers[i]: str(row[i]).strip() if row[i] is not None else ""
                for i in range(len(headers))
            })
        return result
    raise ValueError(f"Unsupported file type '{path.suffix}'. Use .csv or .xlsx")


@dataclass
class _ImportResult:
    client_id: str
    website_id: str
    company_name: str
    url: str
    outcome: str  # READY | READY_WITH_WARNINGS | FAILED | SKIPPED | SKIPPED_EXISTS | INVALID
    errors: list[str] = _dc_field(default_factory=list)


def _display_import_report(results: list[_ImportResult]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1

    created = counts.get("READY", 0) + counts.get("READY_WITH_WARNINGS", 0) + counts.get("FAILED", 0)

    lines = [
        f"[bold]IMPORT SUMMARY[/bold]",
        "",
        f"{len(results)} rows processed",
        f"Credential files created: {created}",
        "",
        f"[green]READY:[/green]                  {counts.get('READY', 0)}",
        f"[yellow]READY_WITH_WARNINGS:[/yellow]    {counts.get('READY_WITH_WARNINGS', 0)}",
        f"[red]FAILED:[/red]                 {counts.get('FAILED', 0)}",
    ]
    if counts.get("SKIPPED"):
        lines.append(f"[dim]SKIPPED (not Pending):[/dim]   {counts['SKIPPED']}")
    if counts.get("SKIPPED_EXISTS"):
        lines.append(f"[dim]SKIPPED (already exists):[/dim] {counts['SKIPPED_EXISTS']}")
    if counts.get("INVALID"):
        lines.append(f"[red]INVALID (bad row):[/red]      {counts['INVALID']}")

    console.print()
    console.print(Panel("\n".join(lines), title="[bold]Bulk Import — Complete[/bold]", expand=False))

    attention = [r for r in results if r.outcome not in ("READY", "SKIPPED")]
    if attention:
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Site", style="bold", min_width=28)
        table.add_column("Status", min_width=22)
        table.add_column("Issues", overflow="fold")

        _outcome_color = {
            "READY_WITH_WARNINGS": "yellow",
            "FAILED": "red",
            "SKIPPED_EXISTS": "dim",
            "INVALID": "red",
        }
        for r in attention:
            color = _outcome_color.get(r.outcome, "dim")
            name = f"{r.client_id}/{r.website_id}"
            if r.company_name:
                name += f"\n[dim]{r.company_name}[/dim]"
            table.add_row(name, f"[{color}]{r.outcome}[/{color}]", "\n".join(r.errors))

        console.print(Panel(table, title="[bold]Sites Requiring Attention[/bold]", expand=False))


@app.command("import-sites")
def import_sites(
    file: Path = typer.Argument(..., help="Path to .xlsx or .csv spreadsheet."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing credential files."),
) -> None:
    """Bulk onboard multiple WordPress sites from a spreadsheet."""
    if not file.exists():
        console.print(f"[red]✗ File not found:[/red] {file}")
        raise typer.Exit(code=1)

    try:
        rows = _load_import_rows(file)
    except ImportError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1)

    if not rows:
        console.print("[yellow]Warning:[/yellow] No data rows found in file.")
        return

    console.print()
    console.print(Panel(
        f"[bold]SEO Agent — Bulk Site Import[/bold]\n"
        f"[dim]{file.name}  ·  {len(rows)} row{'s' if len(rows) != 1 else ''}[/dim]",
        expand=False,
    ))
    console.print()

    _REQUIRED = {"client_id", "website_id", "website_url", "wp_username", "app_password"}
    store = CredentialStore(settings.credentials_dir)
    profile_svc = SiteProfileService(settings.profiles_dir)
    results: list[_ImportResult] = []

    for i, row in enumerate(rows, 1):
        client_id = row.get("client_id", "").strip()
        website_id = row.get("website_id", "").strip()
        company_name = row.get("company_name", "").strip()
        url = row.get("website_url", "").strip().rstrip("/")
        wp_user = row.get("wp_username", "").strip()
        app_password_val = row.get("app_password", "").strip()
        default_cat_raw = row.get("default_category_id", "").strip()
        onboarding_status = row.get("onboarding_status", "Pending").strip()
        city = row.get("city", "").strip()
        state = row.get("state", "").strip()

        label = company_name or client_id or f"row {i}"

        # Validate required fields
        missing = sorted(f for f in _REQUIRED if not row.get(f, "").strip())
        if missing:
            results.append(_ImportResult(
                client_id=client_id or f"row_{i}",
                website_id=website_id or "?",
                company_name=company_name,
                url=url,
                outcome="INVALID",
                errors=[f"Missing required fields: {', '.join(missing)}"],
            ))
            console.print(f"[dim]Row {i:2d}[/dim]  [red]INVALID[/red]         {label}")
            continue

        # Skip non-Pending rows unless --overwrite (which re-syncs all existing sites)
        if not overwrite and onboarding_status.lower() != "pending":
            results.append(_ImportResult(
                client_id=client_id,
                website_id=website_id,
                company_name=company_name,
                url=url,
                outcome="SKIPPED",
                errors=[f"onboarding_status is '{onboarding_status}' (expected 'Pending')"],
            ))
            console.print(f"[dim]Row {i:2d}[/dim]  [dim]SKIPPED[/dim]         {label}")
            continue

        # Skip if credentials already exist and not overwriting
        if store.exists(client_id, website_id) and not overwrite:
            results.append(_ImportResult(
                client_id=client_id,
                website_id=website_id,
                company_name=company_name,
                url=url,
                outcome="SKIPPED_EXISTS",
                errors=["Credentials already exist. Use --overwrite to replace."],
            ))
            console.print(f"[dim]Row {i:2d}[/dim]  [yellow]SKIPPED_EXISTS[/yellow]  {label}")
            continue

        # Save credentials (WordPress auth only)
        default_category_id: int | None = None
        if default_cat_raw:
            try:
                default_category_id = int(default_cat_raw)
            except ValueError:
                pass

        creds = WordPressCredentials(
            url=url,
            user=wp_user,
            app_password=app_password_val,
            default_category_id=default_category_id,
        )
        store.save(client_id, website_id, creds)

        # Update or create site profile with city/state from CSV
        if city and state:
            existing_profile = profile_svc.load(client_id, website_id)
            if existing_profile:
                profile_svc.save(existing_profile.model_copy(update={"city": city, "state": state}))
            else:
                profile_svc.save(SiteProfile(
                    client_id=client_id,
                    website_id=website_id,
                    business_name=company_name or website_id,
                    niche="",
                    primary_service="",
                    city=city,
                    state=state,
                ))

        # Validate site using shared architecture
        with console.status(f"[bold blue]Validating {label}...", spinner="dots"):
            with WordPressService(creds) as wp:
                site_result = wp.validate_site()

        results.append(_ImportResult(
            client_id=client_id,
            website_id=website_id,
            company_name=company_name,
            url=url,
            outcome=site_result.status,
            errors=site_result.errors,
        ))
        _status_color = {"READY": "green", "READY_WITH_WARNINGS": "yellow", "FAILED": "red"}
        sc = _status_color[site_result.status]
        console.print(f"[dim]Row {i:2d}[/dim]  [{sc}]{site_result.status:22}[/{sc}]  {label}")

    _display_import_report(results)

    if any(r.outcome == "FAILED" for r in results):
        raise typer.Exit(code=1)


@app.command("onboard-site")
def onboard_site(
    force: bool = typer.Option(False, "--force", help="Overwrite existing credentials."),
) -> None:
    """Guided setup: collect credentials, validate the site, and save to the credential store."""
    console.print()
    console.print(Panel(
        "[bold]SEO Agent — Onboard New Site[/bold]\n"
        "[dim]Credentials are stored locally. No code changes required.[/dim]",
        expand=False,
    ))
    console.print()

    # ── Phase 1: Collect ──────────────────────────────────────────────────────

    while True:
        client_id = typer.prompt("Client ID (e.g. ACME)").strip()
        if _TENANT_ID_RE.match(client_id):
            break
        console.print("[red]Error:[/red] Client ID may only contain letters, digits, underscores, and hyphens.")

    while True:
        website_id = typer.prompt("Website ID (e.g. acme-chicago)").strip()
        if _TENANT_ID_RE.match(website_id):
            break
        console.print("[red]Error:[/red] Website ID may only contain letters, digits, underscores, and hyphens.")

    site_url = typer.prompt("WordPress site URL (e.g. https://example.com)").strip().rstrip("/")
    wp_user = typer.prompt("WordPress username").strip()
    app_password = typer.prompt("Application Password", hide_input=True).strip()

    # ── Phase 2: Validate (pre-save) ─────────────────────────────────────────

    store = CredentialStore(settings.credentials_dir)
    if store.exists(client_id, website_id) and not force:
        console.print(
            f"\n[yellow]Warning:[/yellow] Credentials already exist for "
            f"[bold]{client_id}/{website_id}[/bold].\n"
            "Use [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(code=1)

    # ── Phase 3: Save ─────────────────────────────────────────────────────────

    creds = WordPressCredentials(url=site_url, user=wp_user, app_password=app_password)
    saved_path = store.save(client_id, website_id, creds)
    console.print(f"\n[green]✓[/green] Credentials saved: [dim]{saved_path}[/dim]")

    # ── Phase 4: validate_site ────────────────────────────────────────────────

    console.print()
    with WordPressService(creds) as wp:
        with console.status("[bold blue]Validating site...", spinner="dots"):
            result = wp.validate_site()

    _display_site_validation(result, site_url)

    # ── Phase 5: Summary ──────────────────────────────────────────────────────

    console.print()
    if result.status == "READY":
        console.print(Panel(
            f"[green]Site is ready.[/green] Run your first article:\n\n"
            f"  [bold]seo-agent autopublish --client-id {client_id} --website-id {website_id}[/bold]",
            title="[bold green]✓ Onboarding Complete[/bold green]",
            expand=False,
        ))
    elif result.status == "READY_WITH_WARNINGS":
        warnings = "\n".join(f"  • {e}" for e in result.errors)
        console.print(Panel(
            f"[yellow]Site is functional but has warnings:[/yellow]\n\n{warnings}\n\n"
            f"Fix the above issues, then run:\n"
            f"  [bold]seo-agent validate wordpress --client-id {client_id} --website-id {website_id}[/bold]",
            title="[bold yellow]⚠ Onboarding Complete — Warnings[/bold yellow]",
            expand=False,
        ))
    else:
        issues = "\n".join(f"  • {e}" for e in result.errors)
        console.print(Panel(
            f"[red]Site validation failed:[/red]\n\n{issues}\n\n"
            f"Fix the above issues, then re-run onboard-site or validate:\n"
            f"  [bold]seo-agent validate wordpress --client-id {client_id} --website-id {website_id}[/bold]\n\n"
            f"[dim]Credentials are saved at {saved_path} — edit or delete to retry.[/dim]",
            title="[bold red]✗ Onboarding Incomplete — Action Required[/bold red]",
            expand=False,
        ))
        raise typer.Exit(code=1)


@app.command("benchmark-costs")
def benchmark_costs(
    no_qa: bool = typer.Option(False, "--no-qa", help="Skip QA stage (saves ~$0.05)."),
) -> None:
    """
    Measure the real cost of each pipeline stage using a tiny synthetic article.

    Exercises every LLM call in the autopublish pipeline (topics, plan, generate,
    SEO, Claude QA, OpenAI QA).  Does NOT write to WordPress, filesystem, or budget.
    Prints a per-stage cost table and cumulative percentages.

    Estimated cost per run: $0.15–$0.22 (same as a normal article — stages cannot
    be made cheaper without reducing max_tokens or switching models).
    """
    import services.call_tracer as _ct

    tracer = _ct.start()

    BENCH_TOPIC = "When to replace garage door springs: signs, costs, and safety"
    BENCH_KEYWORD = "garage door spring replacement"
    BENCH_SERVICE = "Garage door repair"

    request = ArticleRequest(
        topic=BENCH_TOPIC,
        service=BENCH_SERVICE,
        focus_keyword=BENCH_KEYWORD,
        word_count=800,
    )
    tenant = TenantContext(
        client_id=settings.default_client_id or "BENCHMARK",
        website_id=settings.default_website_id or "cost-test",
    )

    console.print(Panel(
        f"[bold]Pipeline Cost Benchmark[/bold]\n"
        f"Topic:   {BENCH_TOPIC}\n"
        f"Service: {BENCH_SERVICE}\n"
        f"Stages:  topics → plan → generate → SEO → QA (max 1 cycle)" +
        (" [dim](QA skipped)[/dim]" if no_qa else ""),
        expand=False,
    ))

    # ── Stage 0: topics:suggest ───────────────────────────────────────────────
    with console.status("[dim]topics:suggest...[/dim]"):
        _topic_system = (
            "You are an expert SEO content strategist for local service businesses. "
            "Generate fresh, unique blog topic ideas that have strong local search intent. "
            "All topics must be evergreen."
        )
        _topic_user = (
            f"Service: {BENCH_SERVICE}\n"
            "Return ONLY a numbered list of 5 topic ideas, one per line. "
            "No preamble, no markdown beyond the numbers."
        )
        claude.generate(
            _topic_system,
            [{"role": "user", "content": _topic_user}],
            thinking=False,
            model=settings.topic_model,
            label="topics:suggest",
        )
    console.print("[dim]  topics:suggest ✓[/dim]")

    # ── Stages 1–3: plan → generate → seo ────────────────────────────────────
    with console.status("[dim]plan → generate → seo...[/dim]"):
        try:
            article = article_agent.generate(request=request, tenant=tenant)
        except Exception as exc:
            console.print(f"[red]Generation failed:[/red] {exc}")
            raise typer.Exit(code=1)
    console.print(f"[dim]  plan + generate + seo ✓  ({article.word_count} words)[/dim]")

    # ── Stages 4–5: QA review (Claude + OpenAI) ──────────────────────────────
    if not no_qa:
        openai_reviewer = None
        if settings.openai_api_key:
            try:
                openai_reviewer = OpenAIReviewService(
                    api_key=settings.openai_api_key,
                    text_model=settings.openai_text_review_model,
                    vision_model=settings.openai_vision_review_model,
                )
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] OpenAI reviewer unavailable: {exc}")

        qa_agent = DualQAAgent(
            claude=claude,
            openai_reviewer=openai_reviewer,
            min_seo=settings.qa_min_seo,
            min_editorial=settings.qa_min_editorial,
            min_writing=settings.qa_min_writing,
            min_authenticity=settings.qa_min_authenticity,
            min_vision_claude=settings.qa_min_vision_claude,
            min_vision_openai=settings.qa_min_vision_openai,
            max_cycles=1,
            enable_rescue=False,
        )
        with console.status("[dim]QA review (1 cycle)...[/dim]"):
            try:
                qa_agent.run(article, [])
            except DualQAFailedError:
                pass  # QA failure is expected; we measured the cost
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] QA raised unexpected error: {exc}")
        console.print("[dim]  QA review ✓[/dim]")

    # ── Cost report ───────────────────────────────────────────────────────────
    records = tracer.records
    if not records:
        console.print("[yellow]No LLM calls were recorded.[/yellow]")
        raise typer.Exit(code=0)

    mandatory_cost = tracer.total_cost()
    sorted_records = sorted(records, key=lambda r: r.cost_usd, reverse=True)

    from rich import box as _box
    from rich.table import Table as _Table

    t = _Table(
        "Stage", "Model", "Input Tok", "Output Tok", "Cost (USD)", "% of Total",
        box=_box.SIMPLE,
        header_style="bold dim",
        title="[bold]Mandatory Stages (measured)[/bold]",
    )
    for r in sorted_records:
        pct = (r.cost_usd / mandatory_cost * 100) if mandatory_cost > 0 else 0.0
        bar_len = int(pct / 5)
        bar = "█" * bar_len
        t.add_row(
            r.stage,
            r.model,
            f"{r.input_tokens:,}",
            f"{r.output_tokens:,}",
            f"${r.cost_usd:.4f}",
            f"{pct:5.1f}%  {bar}",
        )

    total_in = sum(r.input_tokens for r in records)
    total_out = sum(r.output_tokens for r in records)
    t.add_row(
        f"[bold]MANDATORY TOTAL ({len(records)} calls)[/bold]", "",
        f"[bold]{total_in:,}[/bold]",
        f"[bold]{total_out:,}[/bold]",
        f"[bold cyan]${mandatory_cost:.4f}[/bold cyan]",
        "",
    )
    console.print(t)

    if sorted_records:
        console.print(
            f"[dim]Most expensive:[/dim] [bold]{sorted_records[0].stage}[/bold]  "
            f"${sorted_records[0].cost_usd:.4f}  "
            f"({sorted_records[0].cost_usd / mandatory_cost * 100:.1f}%)"
        )
        console.print(
            f"[dim]Cheapest:       [/dim] [bold]{sorted_records[-1].stage}[/bold]  "
            f"${sorted_records[-1].cost_usd:.4f}  "
            f"({sorted_records[-1].cost_usd / mandatory_cost * 100:.1f}%)"
        )

    # ── Conditional: image pipeline (estimated) ───────────────────────────────
    # These stages require Google Drive credentials and candidates to be present.
    # They cannot be directly measured in the benchmark environment, so costs are
    # derived from actual prompt sizes (read from source) and model pricing from
    # budget_service._MODEL_PRICING.  Estimates assume a typical 800-word article
    # with 2 image slots, 1 partial-match slot (score 40–74), and 1 edited image.
    _H_IN  = 1.00   # haiku-4 input  $/M tokens (budget_service._MODEL_PRICING)
    _H_OUT = 5.00   # haiku-4 output $/M tokens
    _M_IN  = 0.15   # gpt-4o-mini input  $/M tokens
    _M_OUT = 0.60   # gpt-4o-mini output $/M tokens
    _IMG_FLAT = 0.25  # gpt-image-1 edit, 1536×1024 high-quality, from budget_service

    def _est(in_tok: int, out_tok: int, in_price: float, out_price: float) -> float:
        return in_tok / 1_000_000 * in_price + out_tok / 1_000_000 * out_price

    # Each entry: (label, model_display, in_tok, out_tok, cost_usd, notes)
    # image:plan — system prompt ~350 tok + article markdown ~1650 tok; output JSON ~600 tok
    # image:vision-score — system ~200 + slot desc ~150 + 10 thumbnails ~1450 tok; JSON ~250; ×2 slots
    # image:edit-prompt — system + short user prompt ~500 tok; max_tokens=400 (actual ~120)
    # openai:image-edit — flat $0.25 per image; tracked by BudgetService, not call_tracer
    # qa:vision-review — system ~200 + full edited image ~3800 tok + context ~200; JSON ~400
    # openai:vision-review — image + text via gpt-4o-mini; ~1500 in, ~250 out
    _img_stages: list[tuple[str, str, int, int, float, str]] = [
        (
            "image:plan",
            settings.image_eval_model,
            2_000, 600,
            _est(2_000, 600, _H_IN, _H_OUT),
            "1× per article; Drive configured + candidates exist",
        ),
        (
            "image:vision-score",
            settings.image_eval_model,
            1_800, 250,
            _est(1_800, 250, _H_IN, _H_OUT) * 2,
            "2× (one per image slot); multimodal thumbnails",
        ),
        (
            "image:edit-prompt",
            settings.edit_prompt_model,
            500, 120,
            _est(500, 120, _H_IN, _H_OUT),
            "0–1×; only when vision score 40–74 (partial match)",
        ),
        (
            "openai:image-edit",
            "gpt-image-1",
            0, 0,
            _IMG_FLAT,
            "~$0.25 flat/image; tracked by BudgetService, not call_tracer",
        ),
        (
            "qa:vision-review",
            settings.image_eval_model,
            4_200, 400,
            _est(4_200, 400, _H_IN, _H_OUT),
            "1× per edited image; multimodal full-size image",
        ),
        (
            "openai:vision-review",
            settings.openai_vision_review_model,
            1_500, 250,
            _est(1_500, 250, _M_IN, _M_OUT),
            "1× per edited image",
        ),
    ]

    image_subtotal = sum(s[4] for s in _img_stages)
    full_pipeline_cost = mandatory_cost + image_subtotal

    img_t = _Table(
        "Stage", "Model", "Est. In Tok", "Est. Out Tok", "Est. Cost", "Notes",
        box=_box.SIMPLE,
        header_style="bold dim",
        title="\n[bold]Conditional: Image Pipeline (estimated — Drive not configured)[/bold]",
    )
    for label, model, in_tok, out_tok, cost, notes in _img_stages:
        in_str = f"{in_tok:,}" if in_tok else "—"
        out_str = f"{out_tok:,}" if out_tok else "—"
        img_t.add_row(label, model, in_str, out_str, f"${cost:.4f}", f"[dim]{notes}[/dim]")

    img_t.add_row(
        "[bold]IMAGE SUBTOTAL[/bold]", "[dim]1 edited image scenario[/dim]",
        "", "",
        f"[bold yellow]${image_subtotal:.4f}[/bold yellow]",
        "[dim]includes $0.25 OpenAI image edit[/dim]",
    )
    console.print(img_t)

    # ── Two-total summary ─────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Cost Summary[/bold]\n\n"
        f"  [dim]Until QA failure  (mandatory stages):[/dim]  "
        f"[bold cyan]${mandatory_cost:.4f}[/bold cyan]"
        f"  [dim]topics → plan → generate → SEO → QA[/dim]\n\n"
        f"  [dim]Full publish + image processing:[/dim]       "
        f"[bold green]${full_pipeline_cost:.4f}[/bold green]"
        f"  [dim]+ Drive image pipeline (1 edited image)[/dim]",
        expand=False,
    ))
    console.print(
        f"\n[dim]Note: cost is NOT recorded to the monthly budget (benchmark mode).[/dim]"
        f"\n[dim]Image pipeline costs are estimates derived from prompt sizes and model "
        f"pricing. Actual cost varies with article length, Drive candidate count, and "
        f"vision score distribution.[/dim]"
    )


if __name__ == "__main__":
    app()

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from config import settings
from models.article import Article
from models.enums import ArticleStatus, SEOPlugin
from models.media import ImageMetadata
from models.seo_report import SEOReport
from services.seo_qa_service import SEOQAService
from services.wordpress_service import WordPressService

if TYPE_CHECKING:
    from models.image_request import ImagePurpose, ImageRequest

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class SEOQualityError(Exception):
    """
    Raised when an article does not meet the SEO quality threshold.

    Carries the full SEOReport so the caller (main.py) can display
    a detailed breakdown without re-running the analysis.

    Two conditions trigger this error:
      - report.summary.critical > 0  → blocked unconditionally
      - report.score < min_score     → blocked by threshold
    """
    def __init__(self, report: SEOReport, min_score: int) -> None:
        self.report = report
        self.min_score = min_score
        if report.summary.critical > 0:
            reason = f"{report.summary.critical} critical issue(s)"
        else:
            reason = f"score {report.score}/100 (minimum: {min_score})"
        super().__init__(f"Article did not pass SEO QA: {reason}")


# ── Dry-run report ────────────────────────────────────────────────────────────

@dataclass
class DryRunReport:
    """
    Result of a dry-run validation pass.

    All checks run regardless of earlier failures so the user sees
    the full picture in one shot.
    """
    connection_ok: bool = False
    auth_ok: bool = False
    auth_user: str | None = None
    html_chars: int = 0
    post_action: str = ""
    qa_report: SEOReport | None = None
    category_name: str | None = None
    category_found: bool = False
    category_id: int | None = None
    tags_existing: list[str] = field(default_factory=list)
    tags_to_create: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (
            self.connection_ok
            and self.auth_ok
            and not self.validation_issues
            and not self.errors
        )


# ── Agent ─────────────────────────────────────────────────────────────────────

class PublisherAgent:
    """
    Orchestrates publishing an Article to WordPress.

    Responsibilities:
    - Pre-publish validation of the Article object.
    - Markdown → HTML conversion.
    - Taxonomy resolution (category lookup, tag lookup/create).
    - Post creation or explicit update via _publish_post().
    - SEO plugin meta injection.
    - Article state update after successful publish.

    Default behavior: every publish() call creates a NEW WordPress post.
    To update an existing post, pass update_post_id=<id> explicitly.
    """

    # ── SEO meta keys and labels (used externally for verification + display) ──

    YOAST_META_KEYS: list[str] = [
        "_yoast_wpseo_title",
        "_yoast_wpseo_metadesc",
        "_yoast_wpseo_focuskw",
        "_yoast_wpseo_opengraph-title",
        "_yoast_wpseo_opengraph-description",
        "_yoast_wpseo_twitter-title",
        "_yoast_wpseo_twitter-description",
    ]
    YOAST_META_KEYS_CONDITIONAL: list[str] = [
        "_yoast_wpseo_canonical",
        "_yoast_wpseo_opengraph-image",
        "_yoast_wpseo_twitter-image",
    ]
    RANKMATH_META_KEYS: list[str] = [
        "rank_math_title",
        "rank_math_description",
        "rank_math_focus_keyword",
        "rank_math_og_title",
        "rank_math_og_description",
        "rank_math_twitter_title",
        "rank_math_twitter_description",
    ]
    RANKMATH_META_KEYS_CONDITIONAL: list[str] = [
        "rank_math_canonical_url",
        "rank_math_og_image_url",
        "rank_math_twitter_image_url",
    ]
    YOAST_META_LABELS: dict[str, str] = {
        "_yoast_wpseo_title":                 "SEO Title",
        "_yoast_wpseo_metadesc":              "Meta Description",
        "_yoast_wpseo_focuskw":               "Focus Keyphrase",
        "_yoast_wpseo_opengraph-title":       "Open Graph Title",
        "_yoast_wpseo_opengraph-description": "Open Graph Description",
        "_yoast_wpseo_opengraph-image":       "Open Graph Image",
        "_yoast_wpseo_twitter-title":         "Twitter Title",
        "_yoast_wpseo_twitter-description":   "Twitter Description",
        "_yoast_wpseo_twitter-image":         "Twitter Image",
        "_yoast_wpseo_canonical":             "Canonical URL",
    }
    RANKMATH_META_LABELS: dict[str, str] = {
        "rank_math_title":               "SEO Title",
        "rank_math_description":         "Meta Description",
        "rank_math_focus_keyword":       "Focus Keyphrase",
        "rank_math_og_title":            "Open Graph Title",
        "rank_math_og_description":      "Open Graph Description",
        "rank_math_og_image_url":        "Open Graph Image",
        "rank_math_twitter_title":       "Twitter Title",
        "rank_math_twitter_description": "Twitter Description",
        "rank_math_twitter_image_url":   "Twitter Image",
        "rank_math_canonical_url":       "Canonical URL",
    }

    def __init__(self, service: WordPressService) -> None:
        self._service = service
        self.last_qa_report: SEOReport | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def validate(self, article: Article) -> list[str]:
        """
        Check whether an Article has all required fields for publishing.

        Returns a list of human-readable issues. An empty list means the
        article is ready.
        """
        issues: list[str] = []

        if not article.title.strip():
            issues.append("title is empty")
        if not article.content_markdown.strip():
            issues.append("content_markdown is empty")
        if not article.seo.seo_title.strip():
            issues.append("seo.seo_title is empty")
        if not article.seo.meta_description.strip():
            issues.append("seo.meta_description is empty")
        if not article.seo.slug.strip():
            issues.append("seo.slug is empty")
        if not article.seo.focus_keyword.strip():
            issues.append("seo.focus_keyword is empty")
        if not article.request.location or not article.request.location.city:
            issues.append(
                "location is missing — publishing a local SEO article without a city/state "
                "is blocked. Re-generate with --city/--state or create a site profile."
            )

        return issues

    def dry_run(self, article: Article, update_post_id: int | None = None) -> DryRunReport:
        """
        Validate everything without creating or modifying any post.

        Simulates the full publish flow — connection, auth, content conversion,
        taxonomy resolution, and post lookup — and reports what would happen.
        All checks run to completion so the report shows the full picture.
        """
        report = DryRunReport()

        # ── Validate Article fields ───────────────────────────
        report.validation_issues = self.validate(article)

        # ── WordPress connection ──────────────────────────────
        try:
            self._service.check_connection()
            report.connection_ok = True
        except Exception as exc:
            report.errors.append(f"Connection failed: {exc}")

        # ── Authentication ────────────────────────────────────
        if report.connection_ok:
            try:
                report.auth_user = self._service.check_auth()
                report.auth_ok = True
            except Exception as exc:
                report.errors.append(f"Authentication failed: {exc}")

        # ── SEO QA ────────────────────────────────────────────
        try:
            report.qa_report = SEOQAService().analyze(article)
        except Exception as exc:
            report.errors.append(f"SEO QA analysis failed: {exc}")

        # ── Markdown → HTML ───────────────────────────────────
        try:
            html = self._to_html(article.content_markdown, site_url=self._service.site_url)
            report.html_chars = len(html)
        except Exception as exc:
            report.errors.append(f"Markdown conversion failed: {exc}")

        # ── Post lookup (simulate upsert chain, read-only) ────
        if report.auth_ok:
            report.post_action = self._simulate_upsert(article, update_post_id)

        # ── Category ──────────────────────────────────────────
        if article.seo.suggested_category:
            report.category_name = article.seo.suggested_category
            try:
                cat_id = self._service.get_category_by_name(article.seo.suggested_category)
                if cat_id is not None:
                    report.category_found = True
                    report.category_id = cat_id
                elif self._service.default_category_id is not None:
                    report.category_found = False
                    report.category_id = self._service.default_category_id
            except Exception as exc:
                report.errors.append(f"Category lookup failed: {exc}")

        # ── Tags ──────────────────────────────────────────────
        for tag_name in article.seo.suggested_tags:
            try:
                existing_id = self._service.find_tag_by_name(tag_name)
                if existing_id is not None:
                    report.tags_existing.append(tag_name)
                else:
                    report.tags_to_create.append(tag_name)
            except Exception as exc:
                report.errors.append(f"Tag lookup failed for '{tag_name}': {exc}")

        return report

    def publish(
        self,
        article: Article,
        *,
        min_score: int | None = None,
        uploaded_images: "list[tuple[ImageRequest, ImageMetadata]] | None" = None,
        update_post_id: int | None = None,
        link_enricher: "Any | None" = None,
    ) -> Article:
        """
        Publish the Article to WordPress and return an updated Article.

        By default, always creates a NEW post. Pass update_post_id to update
        an existing WordPress post instead.

        article.content_markdown is the single source of truth for the body.
        Image placement markers (<!-- SEO_AGENT_IMAGE: id -->) must already be
        embedded in content_markdown before this method is called — the pipeline
        in main.py merges them in immediately after image planning.

        Args:
            article:         The article to publish. content_markdown must include
                             any inline image markers before this is called.
            min_score:       Minimum SEO QA score (overrides settings default).
            uploaded_images: (ImageRequest, ImageMetadata) pairs from MediaService.
                             Inline images replace their markers in the HTML;
                             the first FEATURED image sets featured_media on the post.
            update_post_id:  When set, updates that WordPress post instead of
                             creating a new one.

        Raises:
            ValueError:         Article failed pre-publish validation.
            SEOQualityError:    Article did not pass the SEO QA threshold.
            WordPressAuthError: Bad credentials.
            WordPressAPIError:  Any other WP REST API error.
        """
        issues = self.validate(article)
        if issues:
            raise ValueError(
                "Article is not ready to publish:\n"
                + "\n".join(f"  - {i}" for i in issues)
            )

        effective_min_score = min_score if min_score is not None else settings.seo_qa_min_score

        # ── SEO QA gate ───────────────────────────────────────
        qa_report = SEOQAService().analyze(article)
        self.last_qa_report = qa_report
        if qa_report.summary.critical > 0 or qa_report.score < effective_min_score:
            raise SEOQualityError(qa_report, effective_min_score)

        logger.info("Publishing article '%s' (QA score: %d/100)", article.title, qa_report.score)

        # ── Mandatory image gate ──────────────────────────────
        if article.image_plans:
            resolved_ids = {req.id for req, _ in (uploaded_images or [])}
            missing_mandatory = [
                p for p in article.image_plans
                if p.mandatory and p.image_id not in resolved_ids
            ]
            if missing_mandatory:
                ids = ", ".join(p.image_id for p in missing_mandatory)
                raise ValueError(
                    f"Publish blocked — {len(missing_mandatory)} mandatory image(s) not resolved: {ids}.\n"
                    "Every planned mandatory image must be resolved before publishing. "
                    "Check Drive for matching photos or verify the image resolution pipeline."
                )

        # Resolve effective SEO plugin (auto-detect if not configured)
        effective_plugin = self.resolve_seo_plugin(article.publishing.seo_plugin)
        if effective_plugin != SEOPlugin.NONE:
            logger.info("SEO plugin active: %s", effective_plugin.value)

        markdown_source = article.content_markdown

        # Enrich with internal links before HTML conversion
        if link_enricher is not None:
            posts = self._service.list_posts()
            markdown_source = link_enricher.enrich(article, posts, markdown_source)

        # Editorial placement: strip resolver marker positions and recompute
        # optimal positions for every inline image according to editorial rules.
        if uploaded_images:
            from agents.editorial_placement import (
                EditorialPlacementEngine,
                make_openai_embed_fn,
            )
            from models.image_request import ImagePurpose
            inline_images = [
                (req, meta)
                for req, meta in uploaded_images
                if req.purpose == ImagePurpose.INLINE
            ]
            if inline_images:
                embed_fn = make_openai_embed_fn()
                if embed_fn is None:
                    logger.info("Editorial placement: OpenAI embeddings unavailable; using lexical matching.")
                placement = EditorialPlacementEngine(embed_fn=embed_fn).place(
                    markdown_source, inline_images
                )
                markdown_source = placement.markdown
                logger.info(
                    "Editorial layout score: %d/100 "
                    "(distribution=%d  spacing=%d  matching=%d  rhythm=%d  cost=%.0f)",
                    placement.score.overall,
                    placement.score.distribution,
                    placement.score.spacing,
                    placement.score.section_matching,
                    placement.score.visual_rhythm,
                    placement.score.layout_cost,
                )
                for warning in placement.score.warnings:
                    logger.warning("Editorial: %s", warning)

        html_content = self._to_html(markdown_source, site_url=self._service.site_url)

        # Inject inline images; extract featured media ID and URL for SEO meta
        featured_media_id: int | None = None
        featured_image_url: str | None = None
        if uploaded_images:
            html_content = self._inject_images(html_content, uploaded_images)
            featured_media_id = self._extract_featured_media_id(uploaded_images)
            featured_image_url = self._extract_featured_image_url(uploaded_images)

        category_id = self._resolve_category(article)
        tag_ids = self._resolve_tags(article)
        payload = self._build_payload(
            article, html_content, category_id, tag_ids,
            featured_media_id, effective_plugin, featured_image_url,
        )

        post_id, post_url = self._publish_post(payload, update_post_id)

        return article.model_copy(update={
            "wp_post_id": post_id,
            "wp_post_url": post_url,
            "content_html": html_content,
            "status": ArticleStatus.PUBLISHED,
            "updated_at": datetime.now(timezone.utc),
            "publishing": article.publishing.model_copy(
                update={"seo_plugin": effective_plugin}
            ),
        })

    # ── Post creation / explicit update ──────────────────────────────────────

    def _publish_post(
        self,
        payload: dict[str, Any],
        update_post_id: int | None = None,
    ) -> tuple[int, str]:
        """
        Create a new post, or update a specific existing post when update_post_id
        is provided. Never searches for an existing post automatically.

        After every create or update the post is immediately re-fetched by ID to
        confirm it exists in WordPress before returning. This catches phantom IDs
        (caching proxies, rolled-back transactions, plugin hooks that delete the
        post after creation) so the pipeline never reports ✓ on a ghost post.
        """
        from services.wordpress_service import WordPressAPIError

        if update_post_id is not None:
            post = self._service.update_post(update_post_id, payload)
            post_id  = post.get("id")
            post_url = post.get("link", "")
            if not isinstance(post_id, int) or post_id <= 0:
                raise WordPressAPIError(
                    f"WordPress update returned an invalid post ID: {post_id!r}. "
                    f"Raw response keys: {list(post.keys())}"
                )
            verified = self._service.get_post(post_id)
            if verified is None:
                raise WordPressAPIError(
                    f"Post ID {post_id} was not retrievable after update — "
                    "the WordPress API accepted the request but the post does not exist."
                )
            logger.info("Post updated and verified (ID=%d): %s", post_id, post_url)
            return post_id, post_url

        post = self._service.create_post(payload)
        post_id  = post.get("id")
        post_url = post.get("link", "")

        if not isinstance(post_id, int) or post_id <= 0:
            raise WordPressAPIError(
                f"WordPress create returned an invalid post ID: {post_id!r}. "
                f"Raw response keys: {list(post.keys())}"
            )
        if not post_url:
            raise WordPressAPIError(
                f"WordPress create returned post ID {post_id} but no link/URL. "
                f"Raw response keys: {list(post.keys())}"
            )

        verified = self._service.get_post(post_id)
        if verified is None:
            raise WordPressAPIError(
                f"Post ID {post_id} was not retrievable after creation — "
                "the WordPress API accepted the request but the post does not exist. "
                f"Claimed URL was: {post_url}"
            )

        logger.info("Post created and verified (ID=%d): %s", post_id, post_url)
        return post_id, post_url

    def _simulate_upsert(self, article: Article, update_post_id: int | None = None) -> str:
        """
        Read-only simulation for dry-run reporting.
        """
        if update_post_id is not None:
            existing = self._service.get_post(update_post_id)
            if existing is not None:
                return f"UPDATE existing post (ID {update_post_id}, via --post-id)"
            return f"UPDATE requested (ID {update_post_id}) — post not found, will fail"

        return "CREATE new post"

    # ── Taxonomy resolution ───────────────────────────────────────────────────

    def _resolve_category(self, article: Article) -> int | None:
        default = self._service.default_category_id

        if not article.seo.suggested_category:
            return default

        cat_id = self._service.get_category_by_name(article.seo.suggested_category)

        if cat_id is not None:
            return cat_id

        if default is not None:
            logger.warning(
                "Category '%s' not found — using default category ID %d",
                article.seo.suggested_category,
                default,
            )
            return default

        logger.warning(
            "Category '%s' not found and no default configured — "
            "publishing without category.",
            article.seo.suggested_category,
        )
        return None

    def _resolve_tags(self, article: Article) -> list[int]:
        tag_ids: list[int] = []
        for tag_name in article.seo.suggested_tags:
            try:
                tag_ids.append(self._service.get_or_create_tag(tag_name))
            except Exception as exc:
                logger.warning("Could not resolve tag '%s': %s", tag_name, exc)
        return tag_ids

    # ── Payload assembly ──────────────────────────────────────────────────────

    # ── Image injection ───────────────────────────────────────────────────────

    @staticmethod
    def _inject_images(
        html: str,
        uploaded: "list[tuple[ImageRequest, ImageMetadata]]",
    ) -> str:
        """
        Replace <!-- SEO_AGENT_IMAGE: id --> markers with <figure> HTML tags.

        Only processes INLINE images. FEATURED images have no marker.
        Any unresolved markers (e.g. upload failed for one image) are removed
        cleanly rather than left as HTML comments in the published content.
        """
        from models.image_request import ImagePurpose

        for req, meta in uploaded:
            if req.purpose != ImagePurpose.INLINE:
                continue
            if not meta.url:
                logger.warning("Skipping image injection for %s — no URL available.", req.id)
                continue
            figure = PublisherAgent._build_figure(req, meta)
            html = html.replace(req.placement_marker, figure)

        # Remove any markers that were not replaced (upload failed or missing)
        html = re.sub(r'<!-- SEO_AGENT_IMAGE: [^>]+ -->', '', html)
        return html

    @staticmethod
    def _build_figure(req: "ImageRequest", meta: ImageMetadata) -> str:
        from services.editorial_html_renderer import EditorialHTMLRenderer
        return EditorialHTMLRenderer.render_figure(req, meta)

    @staticmethod
    def _extract_featured_media_id(
        uploaded: "list[tuple[ImageRequest, ImageMetadata]]",
    ) -> int | None:
        from models.image_request import ImagePurpose
        for req, meta in uploaded:
            if req.purpose == ImagePurpose.FEATURED and meta.wordpress_media_id:
                return meta.wordpress_media_id
        return None

    @staticmethod
    def _extract_featured_image_url(
        uploaded: "list[tuple[ImageRequest, ImageMetadata]]",
    ) -> str | None:
        from models.image_request import ImagePurpose
        for req, meta in uploaded:
            if req.purpose == ImagePurpose.FEATURED and meta.url:
                return meta.url
        return None

    def resolve_seo_plugin(self, configured: SEOPlugin) -> SEOPlugin:
        """
        Return the effective SEO plugin for this publish run.

        Priority: article.publishing.seo_plugin → settings.seo_plugin env var → auto-detect.
        Auto-detection queries the WP REST namespace registry (/wp-json/) once per publish.
        Called publicly from main.py for pipeline observability before agent.publish().
        """
        if configured != SEOPlugin.NONE:
            return configured
        if settings.seo_plugin == "none":
            return SEOPlugin.NONE
        if settings.seo_plugin in ("yoast", "rankmath"):
            return SEOPlugin(settings.seo_plugin)
        # "auto" (default) — query WP REST namespace registry
        return self._service.detect_seo_plugin()

    def _build_payload(
        self,
        article: Article,
        html_content: str,
        category_id: int | None,
        tag_ids: list[int],
        featured_media_id: int | None = None,
        effective_plugin: SEOPlugin = SEOPlugin.NONE,
        featured_image_url: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title":   article.title,
            "content": html_content,
            "slug":    article.seo.slug,
            "status":  article.publishing.status.value,
        }

        if article.publishing.scheduled_at is not None:
            payload["date"] = article.publishing.scheduled_at.isoformat()

        if article.publishing.wp_author_id is not None:
            payload["author"] = article.publishing.wp_author_id

        if category_id is not None:
            payload["categories"] = [category_id]

        if tag_ids:
            payload["tags"] = tag_ids

        if featured_media_id is not None:
            payload["featured_media"] = featured_media_id

        # Agent meta is always included: enables idempotent lookup and audit.
        # SEO plugin meta is merged on top (different key namespaces, no conflicts).
        payload["meta"] = {
            "_seo_agent_id":             str(article.id),
            "_seo_agent_schema_version": article.schema_version,
            "_seo_agent_prompt_version": article.prompt_version,
            "_seo_agent_model":          article.model_name,
            **self._build_seo_meta(article, effective_plugin, featured_image_url),
        }

        return payload

    def _build_seo_meta(
        self,
        article: Article,
        plugin: SEOPlugin,
        featured_image_url: str | None = None,
    ) -> dict[str, str]:
        """
        Build the meta dict for the active SEO plugin.

        Yoast and Rank Math receive an identical set of fields so there are
        no functional differences between the two integrations:
          - SEO title, meta description, focus keyphrase
          - Open Graph title + description + image (from featured image URL)
          - Twitter card title + description + image
          - Canonical URL (only when explicitly set on the article)
        """
        if plugin not in (SEOPlugin.YOAST, SEOPlugin.RANKMATH):
            return {}

        seo = article.seo
        og_title       = seo.og_title or seo.seo_title
        og_description = seo.og_description or seo.meta_description

        if plugin == SEOPlugin.YOAST:
            meta: dict[str, str] = {
                "_yoast_wpseo_title":                 seo.seo_title,
                "_yoast_wpseo_metadesc":              seo.meta_description,
                "_yoast_wpseo_focuskw":               seo.focus_keyword,
                "_yoast_wpseo_opengraph-title":       og_title,
                "_yoast_wpseo_opengraph-description": og_description,
                "_yoast_wpseo_twitter-title":         og_title,
                "_yoast_wpseo_twitter-description":   og_description,
            }
            if seo.canonical_url:
                meta["_yoast_wpseo_canonical"] = seo.canonical_url
            if featured_image_url:
                meta["_yoast_wpseo_opengraph-image"] = featured_image_url
                meta["_yoast_wpseo_twitter-image"]   = featured_image_url
            return meta

        # SEOPlugin.RANKMATH
        meta = {
            "rank_math_title":               seo.seo_title,
            "rank_math_description":         seo.meta_description,
            "rank_math_focus_keyword":       seo.focus_keyword,
            "rank_math_og_title":            og_title,
            "rank_math_og_description":      og_description,
            "rank_math_twitter_title":       og_title,
            "rank_math_twitter_description": og_description,
        }
        if seo.canonical_url:
            meta["rank_math_canonical_url"] = seo.canonical_url
        if featured_image_url:
            meta["rank_math_og_image_url"]      = featured_image_url
            meta["rank_math_twitter_image_url"] = featured_image_url
        return meta

    # ── Content conversion ────────────────────────────────────────────────────

    @staticmethod
    def _to_html(markdown_content: str, site_url: str = "") -> str:
        from services.editorial_html_renderer import EditorialHTMLRenderer
        return EditorialHTMLRenderer().render(markdown_content, site_url=site_url)

    @staticmethod
    def analyze_html(html: str, site_url: str = "") -> dict[str, int | bool]:
        """
        Inspect final HTML and return editorial/SEO structure counts.

        Used by main.py to populate the pipeline observability report.
        All counts reflect the actual published HTML, not the source markdown.
        """
        from urllib.parse import urlparse

        tables   = len(re.findall(r'<table\b', html))
        callouts = len(re.findall(r'class="wp-block-callout"', html))
        faq      = bool(re.search(
            r'<h[2-4][^>]*>\s*(?:FAQ|Frequently Asked Questions|Preguntas [Ff]recuentes)',
            html, re.IGNORECASE,
        ))

        hrefs = re.findall(r'<a\b[^>]*\bhref="([^"]*)"', html)
        internal = external = 0
        site_host = urlparse(site_url.rstrip("/")).netloc.lstrip("www.") if site_url else ""

        for href in hrefs:
            if not href or href.startswith(("#", "mailto:", "tel:")):
                continue
            parsed = urlparse(href)
            if not parsed.scheme:
                internal += 1
            elif site_host and site_host in parsed.netloc:
                internal += 1
            elif parsed.scheme in ("http", "https"):
                external += 1

        return {
            "tables":         tables,
            "callouts":       callouts,
            "faq":            faq,
            "internal_links": internal,
            "external_links": external,
        }

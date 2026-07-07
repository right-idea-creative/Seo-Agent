import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import markdown as md_parser

from config import settings
from models.article import Article
from models.enums import ArticleStatus, SEOPlugin
from models.media import ImageMetadata
from models.seo_report import SEOReport
from services.seo_qa_service import SEOQAService
from services.wordpress_service import WordPressService

if TYPE_CHECKING:
    from models.image_request import ImagePlacementPlan, ImagePurpose, ImageRequest

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
        image_plan: "ImagePlacementPlan | None" = None,
        uploaded_images: "list[tuple[ImageRequest, ImageMetadata]] | None" = None,
        update_post_id: int | None = None,
        link_enricher: "Any | None" = None,
    ) -> Article:
        """
        Publish the Article to WordPress and return an updated Article.

        By default, always creates a NEW post. Pass update_post_id to update
        an existing WordPress post instead.

        Args:
            article:         The article to publish.
            min_score:       Minimum SEO QA score (overrides settings default).
            image_plan:      Optional plan from ImageResolverAgent. When provided,
                             modified_markdown (which contains inline image markers)
                             is used instead of content_markdown.
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
        if qa_report.summary.critical > 0 or qa_report.score < effective_min_score:
            raise SEOQualityError(qa_report, effective_min_score)

        logger.info("Publishing article '%s' (QA score: %d/100)", article.title, qa_report.score)

        # Resolve effective SEO plugin (auto-detect if not configured)
        effective_plugin = self.resolve_seo_plugin(article.publishing.seo_plugin)
        if effective_plugin != SEOPlugin.NONE:
            logger.info("SEO plugin active: %s", effective_plugin.value)

        # Use modified_markdown (with markers) when a plan is provided
        markdown_source = image_plan.modified_markdown if image_plan else article.content_markdown

        # Enrich with internal links before HTML conversion
        if link_enricher is not None:
            posts = self._service.list_posts()
            markdown_source = link_enricher.enrich(article, posts, markdown_source)

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
        """
        if update_post_id is not None:
            post = self._service.update_post(update_post_id, payload)
            logger.info("Post updated (explicit --post-id=%d): %s", update_post_id, post["link"])
            return post["id"], post["link"]

        post = self._service.create_post(payload)
        logger.info("Post created: ID=%d URL=%s", post["id"], post["link"])
        return post["id"], post["link"]

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

            marker = req.placement_marker
            figure = PublisherAgent._build_figure(req, meta)
            html = html.replace(marker, figure)

        # Remove any markers that were not replaced (missing uploads)
        html = re.sub(r'<!-- SEO_AGENT_IMAGE: [^>]+ -->', '', html)
        return html

    @staticmethod
    def _build_figure(req: "ImageRequest", meta: ImageMetadata) -> str:
        img_style = PublisherAgent._S_IMG
        attrs = (
            f'src="{meta.url}" alt="{req.alt_text}" '
            f'loading="lazy" decoding="async" style="{img_style}"'
        )
        if meta.width:
            attrs += f' width="{meta.width}"'
        if meta.height:
            attrs += f' height="{meta.height}"'
        img = f'<img {attrs} />'
        figure_style = PublisherAgent._S_FIGURE
        if req.caption:
            caption_style = PublisherAgent._S_FIGCAPTION
            return (
                f'<figure class="wp-block-image size-full" style="{figure_style}">'
                f'{img}'
                f'<figcaption class="wp-element-caption" style="{caption_style}">'
                f'{req.caption}'
                f'</figcaption>'
                f'</figure>'
            )
        return f'<figure class="wp-block-image size-full" style="{figure_style}">{img}</figure>'

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
        html = md_parser.markdown(markdown_content, extensions=["extra"])
        return PublisherAgent._enrich_html(html, site_url=site_url)

    # ── Typography constants ──────────────────────────────────────────────────
    _S_H1 = (
        "font-size:44px;font-weight:700;line-height:1.2;"
        "margin-top:0;margin-bottom:32px;"
    )
    _S_H2 = (
        "font-size:34px;font-weight:700;line-height:1.3;"
        "margin-top:64px;margin-bottom:24px;"
    )
    _S_H3 = (
        "font-size:26px;font-weight:600;line-height:1.35;"
        "margin-top:40px;margin-bottom:16px;"
    )
    _S_P = (
        "font-size:19px;line-height:1.85;margin-bottom:24px;"
    )
    _S_LI = (
        "font-size:19px;line-height:1.85;margin-bottom:10px;"
    )
    _S_TABLE = (
        "width:100%;border-collapse:collapse;"
        "margin-top:32px;margin-bottom:40px;"
        "font-size:17px;line-height:1.6;"
    )
    _S_TH = (
        "padding:14px 18px;text-align:left;font-weight:700;"
        "background:#1e293b;color:#ffffff;"
        "border-bottom:3px solid #0f172a;"
    )
    _S_TD = (
        "padding:13px 18px;"
        "border-bottom:1px solid #e2e8f0;"
        "vertical-align:top;"
    )
    _S_CALLOUT = (
        "display:flex;align-items:flex-start;gap:14px;"
        "border-left:4px solid #f59e0b;"
        "padding:20px 24px;background:#fefce8;"
        "margin:40px 0 32px;border-radius:0 8px 8px 0;"
        "font-size:17px;line-height:1.7;"
    )
    _S_FIGURE = (
        "margin:40px 0 32px;display:block;"
    )
    _S_IMG = (
        "width:100%;height:auto;border-radius:8px;display:block;"
    )
    _S_FIGCAPTION = (
        "font-size:15px;line-height:1.6;color:#64748b;"
        "margin-top:10px;font-style:italic;text-align:center;"
    )
    _S_LINK_EXT = (
        "color:#2563eb;text-decoration:underline;text-underline-offset:3px;"
    )
    _S_LINK_INT = (
        "color:#0f172a;text-decoration:underline;text-underline-offset:3px;"
        "font-weight:500;"
    )

    @staticmethod
    def _enrich_html(html: str, site_url: str = "") -> str:
        """
        Post-process converted HTML for editorial quality and performance.

        Applies the full editorial style spec via inline CSS so the output is
        self-contained and theme-independent. WordPress (admin user, REST API)
        preserves inline style attributes via wp_kses_post().

        Transforms applied in order:
        1. Callouts   — <blockquote> → styled editorial callout div
        2. Tables     — add full inline table + th/td styles
        3. Typography — h1/h2/h3/p/li inline styles
        4. Images     — ensure loading="lazy" decoding="async" on all <img>
        5. Links      — external links get target="_blank" rel="noopener noreferrer"

        Note: figure/img/figcaption elements are built by _build_figure() and
        injected AFTER this method runs — their styles live in _build_figure().
        """
        from urllib.parse import urlparse

        site_host = urlparse(site_url.rstrip("/")).netloc.lstrip("www.") if site_url else ""

        def _add_style(tag: str, css: str) -> str:
            """Add css to an opening tag string; merge if style already present."""
            if 'style="' in tag:
                return tag.replace('style="', f'style="{css}', 1)
            close = tag.rindex('>')
            return tag[:close] + f' style="{css}"' + tag[close:]

        # ── 1. Blockquote callouts ────────────────────────────────────────────
        # Markdown renders  > ⚠️ **Important:** text  as:
        #   <blockquote><p>⚠️ <strong>Important:</strong> text</p></blockquote>
        # The emoji/icon comes from the article content; we don't add another.
        def _callout(m: re.Match) -> str:
            inner = m.group(1).strip()
            return (
                f'<div class="wp-block-callout" style="{PublisherAgent._S_CALLOUT}">'
                f'<div style="font-size:22px;flex-shrink:0;line-height:1;">💡</div>'
                f'<div style="flex:1;">{inner}</div>'
                f'</div>'
            )

        html = re.sub(
            r'<blockquote>\s*(<p>.*?</p>)\s*</blockquote>',
            _callout,
            html,
            flags=re.DOTALL,
        )

        # ── 2. Tables ─────────────────────────────────────────────────────────
        html = re.sub(
            r'<table\b[^>]*>',
            lambda m: _add_style(
                '<table class="wp-block-table">', PublisherAgent._S_TABLE
            ),
            html,
        )
        html = re.sub(
            r'<th\b[^>]*>',
            lambda m: _add_style('<th>', PublisherAgent._S_TH),
            html,
        )
        html = re.sub(
            r'<td\b[^>]*>',
            lambda m: _add_style('<td>', PublisherAgent._S_TD),
            html,
        )
        # Odd rows: alternate background via nth-child — apply to tbody rows
        html = re.sub(
            r'<tr\b[^>]*>',
            '<tr>',
            html,
        )

        # ── 3. Typography ─────────────────────────────────────────────────────
        # Only add style when not already present (figures/callouts already styled)
        def _style_tag(tag_name: str, css: str, m: re.Match) -> str:
            tag = m.group(0)
            if 'style=' in tag:
                return tag
            return _add_style(tag, css)

        html = re.sub(
            r'<h1\b[^>]*>',
            lambda m: _style_tag('h1', PublisherAgent._S_H1, m),
            html,
        )
        html = re.sub(
            r'<h2\b[^>]*>',
            lambda m: _style_tag('h2', PublisherAgent._S_H2, m),
            html,
        )
        html = re.sub(
            r'<h3\b[^>]*>',
            lambda m: _style_tag('h3', PublisherAgent._S_H3, m),
            html,
        )
        html = re.sub(
            r'<p\b(?![^>]*class="wp-element-caption")[^>]*>',
            lambda m: _style_tag('p', PublisherAgent._S_P, m),
            html,
        )
        html = re.sub(
            r'<li\b[^>]*>',
            lambda m: _style_tag('li', PublisherAgent._S_LI, m),
            html,
        )

        # ── 4. Inline images — lazy + decoding ───────────────────────────────
        def _add_lazy(m: re.Match) -> str:
            tag = m.group(0)
            if 'loading=' not in tag:
                tag = tag[:-2] + ' loading="lazy"' + tag[-2:]
            if 'decoding=' not in tag:
                tag = tag[:-2] + ' decoding="async"' + tag[-2:]
            return tag

        html = re.sub(r'<img\b[^>]*/>', _add_lazy, html)

        # Note: figure/img/figcaption blocks come from _build_figure(), which is
        # called by _inject_images() AFTER _enrich_html(). Those styles are applied
        # directly in _build_figure() via _S_FIGURE / _S_IMG / _S_FIGCAPTION.

        # ── 5. Links ─────────────────────────────────────────────────────────
        # External links: add target + rel + style
        # Internal links: add style only
        def _style_link(m: re.Match) -> str:
            full_tag = m.group(0)
            href_match = re.search(r'href="([^"]*)"', full_tag)
            if not href_match:
                return full_tag
            href = href_match.group(1)

            if not href or href.startswith(("#", "mailto:", "tel:")):
                return full_tag

            parsed = urlparse(href)
            is_external = (
                bool(parsed.scheme)
                and parsed.scheme in ("http", "https")
                and (not site_host or site_host not in parsed.netloc)
            )

            if is_external:
                tag = full_tag
                if 'target=' not in tag:
                    tag = tag.rstrip('>') + ' target="_blank" rel="noopener noreferrer">'
                return _add_style(tag, PublisherAgent._S_LINK_EXT) if 'style=' not in tag else tag
            else:
                return _add_style(full_tag, PublisherAgent._S_LINK_INT) if 'style=' not in full_tag else full_tag

        html = re.sub(r'<a\b[^>]*>', _style_link, html)

        return html

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

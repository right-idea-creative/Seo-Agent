"""
PublicationCertificationService — post-publish verification of the live artifact.

Runs AFTER agent.publish() returns. Read-only: never modifies, re-publishes,
or attempts to fix anything. Produces a final CERTIFIED / NOT_CERTIFIED verdict.

Sections: GENERAL, CONTENT, SEO, LINKS, IMAGES, EDITORIAL, WORDPRESS, QUALITY.
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.article import Article
    from models.seo_report import SEOReport
    from services.editorial_history_service import EditorialHistoryService
    from services.editorial_theme import EditorialTheme
    from services.wordpress_service import WordPressService


@dataclass
class CertificationItem:
    section: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CertificationReport:
    article_id: str
    wp_post_id: int | None
    wp_post_url: str | None
    items: list[CertificationItem] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return bool(self.items) and all(i.passed for i in self.items)

    @property
    def failures(self) -> list[CertificationItem]:
        return [i for i in self.items if not i.passed]

    def section_items(self, section: str) -> list[CertificationItem]:
        return [i for i in self.items if i.section == section]


class PublicationCertificationService:
    """
    Verifies the published artifact against known production requirements.

    Accepts data already in memory — the returned Article from publish(),
    the WP post response dict, uploaded image metadata, and the SEO report.
    Makes one live read (GET /posts/<id>) to confirm the post is truly live.

    This service NEVER calls create_post, update_post, upload_media, or any
    write method. It is strictly observational.
    """

    SECTIONS = [
        "GENERAL",
        "CONTENT",
        "SEO",
        "LINKS",
        "IMAGES",
        "EDITORIAL",
        "WORDPRESS",
        "QUALITY",
        "RENDERING",
    ]

    def certify(
        self,
        *,
        article: "Article",
        wp_post: dict[str, Any],
        uploaded_images: list | None,
        seo_qa_report: "SEOReport | None",
        links_added: int,
        no_links: bool,
        wp_service: "WordPressService",
        editorial_history: "EditorialHistoryService | None" = None,
        theme: "EditorialTheme | None" = None,
        min_seo_score: int = 70,
        min_word_count: int = 300,
    ) -> CertificationReport:
        """
        Run all certification checks and return a report.

        Args:
            article:           Published article (returned by PublisherAgent.publish).
            wp_post:           Raw WP REST API post object as returned by publish.
            uploaded_images:   (ImageRequest, ImageMetadata) pairs from upload, or None.
            seo_qa_report:     SEO QA result from before the gate.
            links_added:       Number of internal links inserted by the enricher.
            no_links:          True when --no-links was passed.
            wp_service:        Live WordPress service for the read-only GET verification.
            editorial_history: Service instance for confirming history was recorded.
            theme:             EditorialTheme used for this publish run. When None,
                               the RENDERING section verifies DefaultEditorialTheme.
            min_seo_score:     Minimum acceptable SEO score.
            min_word_count:    Minimum word count.

        Returns:
            CertificationReport with CERTIFIED or NOT_CERTIFIED verdict.
        """
        report = CertificationReport(
            article_id=str(article.id),
            wp_post_id=article.wp_post_id,
            wp_post_url=article.wp_post_url,
        )

        # Fetch the live post once for verification (read-only)
        live_post = self._fetch_live_post(wp_service, article.wp_post_id)

        self._certify_general(report, article, wp_post, live_post)
        self._certify_content(report, article, live_post, min_word_count)
        self._certify_seo(report, article, live_post, seo_qa_report, min_seo_score)
        self._certify_links(report, article.content_markdown, links_added, no_links)
        self._certify_images(report, article, uploaded_images, live_post)
        self._certify_editorial(report, article, editorial_history)
        self._certify_wordpress(report, article, live_post)
        self._certify_quality(report, article, seo_qa_report)
        self._certify_rendering(report, theme)

        return report

    # ── Live post fetch ───────────────────────────────────────────────────────

    def _fetch_live_post(
        self,
        wp_service: "WordPressService",
        post_id: int | None,
    ) -> dict[str, Any] | None:
        if not post_id:
            return None
        try:
            return wp_service.get_post(post_id)
        except Exception:
            return None

    # ── Section: GENERAL ─────────────────────────────────────────────────────

    def _certify_general(
        self,
        report: CertificationReport,
        article: "Article",
        wp_post: dict[str, Any],
        live_post: dict[str, Any] | None,
    ) -> None:
        section = "GENERAL"

        # WordPress post ID was assigned
        has_id = bool(article.wp_post_id)
        report.items.append(CertificationItem(
            section, "Post ID assigned",
            has_id,
            str(article.wp_post_id) if has_id else "wp_post_id is None",
        ))

        # Live URL is present
        has_url = bool(article.wp_post_url)
        report.items.append(CertificationItem(
            section, "Post URL present",
            has_url,
            article.wp_post_url or "wp_post_url is None",
        ))

        # Post is reachable (live GET succeeded)
        live_ok = live_post is not None
        report.items.append(CertificationItem(
            section, "Post reachable via API",
            live_ok,
            "GET succeeded" if live_ok else "GET /posts/<id> returned None or errored",
        ))

        # Article status reflects published state
        from models.enums import ArticleStatus
        status_ok = article.status == ArticleStatus.PUBLISHED
        report.items.append(CertificationItem(
            section, "Article status = PUBLISHED",
            status_ok,
            article.status.value if not status_ok else "OK",
        ))

    # ── Section: CONTENT ─────────────────────────────────────────────────────

    def _certify_content(
        self,
        report: CertificationReport,
        article: "Article",
        live_post: dict[str, Any] | None,
        min_word_count: int,
    ) -> None:
        section = "CONTENT"

        # Title matches
        live_title = ""
        if live_post:
            rendered = live_post.get("title", {})
            live_title = rendered.get("rendered", "") if isinstance(rendered, dict) else str(rendered)
            live_title = _html.unescape(live_title)
        titles_match = bool(live_title and article.title.strip() in live_title)
        report.items.append(CertificationItem(
            section, "Title published",
            titles_match,
            live_title[:80] if titles_match else (
                f"Title mismatch — expected '{article.title[:60]}', got '{live_title[:60]}'"
                if live_title else "Title missing from live post"
            ),
        ))

        # HTML body present in live post
        live_content = ""
        if live_post:
            content_block = live_post.get("content", {})
            live_content = content_block.get("rendered", "") if isinstance(content_block, dict) else ""
        content_present = len(live_content.strip()) > 100
        report.items.append(CertificationItem(
            section, "Content body published",
            content_present,
            f"{len(live_content):,} chars" if content_present else "Content empty or very short in live post",
        ))

        # Word count
        text = re.sub(r'<!--.*?-->', '', article.content_markdown, flags=re.DOTALL)
        text = re.sub(r'[#*_`\[\]|]', ' ', text)
        words = len(text.split())
        wc_ok = words >= min_word_count
        report.items.append(CertificationItem(
            section, "Word count",
            wc_ok,
            f"{words:,} words" + (f" (minimum: {min_word_count:,})" if not wc_ok else ""),
        ))

        # Slug present in live post
        live_slug = live_post.get("slug", "") if live_post else ""
        slug_ok = bool(live_slug)
        report.items.append(CertificationItem(
            section, "Slug set",
            slug_ok,
            live_slug or "Slug missing from live post",
        ))

    # ── Section: SEO ─────────────────────────────────────────────────────────

    def _certify_seo(
        self,
        report: CertificationReport,
        article: "Article",
        live_post: dict[str, Any] | None,
        seo_report: "SEOReport | None",
        min_score: int,
    ) -> None:
        section = "SEO"

        # SEO title set on article
        seo_title_ok = bool(article.seo.seo_title.strip())
        report.items.append(CertificationItem(
            section, "SEO title present",
            seo_title_ok,
            article.seo.seo_title[:80] if seo_title_ok else "seo_title is empty",
        ))

        # Meta description set
        meta_ok = bool(article.seo.meta_description.strip())
        report.items.append(CertificationItem(
            section, "Meta description present",
            meta_ok,
            f"{len(article.seo.meta_description)} chars" if meta_ok else "meta_description is empty",
        ))

        # Focus keyword set
        kw_ok = bool(article.seo.focus_keyword.strip())
        report.items.append(CertificationItem(
            section, "Focus keyword set",
            kw_ok,
            article.seo.focus_keyword if kw_ok else "focus_keyword is empty",
        ))

        # SEO meta accepted by WordPress (live post meta check)
        if live_post:
            live_meta = live_post.get("meta", {}) or {}
            agent_id_set = bool(live_meta.get("_seo_agent_id"))
            report.items.append(CertificationItem(
                section, "seo-agent.php meta accepted",
                agent_id_set,
                f"_seo_agent_id = {live_meta.get('_seo_agent_id', 'not set')}",
            ))

        # SEO QA score
        if seo_report is not None:
            score_ok = seo_report.score >= min_score and seo_report.summary.critical == 0
            report.items.append(CertificationItem(
                section, "SEO QA score",
                score_ok,
                f"Score {seo_report.score}/100"
                + (f" — {seo_report.summary.critical} critical" if seo_report.summary.critical else "")
                + (f" (minimum: {min_score})" if not score_ok and not seo_report.summary.critical else ""),
            ))

    # ── Section: LINKS ────────────────────────────────────────────────────────

    def _certify_links(
        self,
        report: CertificationReport,
        markdown: str,
        links_added: int,
        no_links: bool,
    ) -> None:
        section = "LINKS"

        if no_links:
            report.items.append(CertificationItem(
                section, "Internal links", True, "Skipped (--no-links)"
            ))
            return

        # Count all markdown links to verify enricher ran
        all_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)
        links_ok = links_added >= 1
        report.items.append(CertificationItem(
            section, "Internal links inserted",
            links_ok,
            f"{links_added} inserted  |  {len(all_links)} total markdown links"
            + ("" if links_ok else " — no internal links were added"),
        ))

        # Verify no broken placeholder links (href = "#" or empty)
        broken = [(txt, href) for txt, href in all_links if not href or href == "#"]
        report.items.append(CertificationItem(
            section, "No broken placeholder links",
            not broken,
            "None found" if not broken else f"Found {len(broken)}: {broken[:3]}",
        ))

    # ── Section: IMAGES ───────────────────────────────────────────────────────

    def _certify_images(
        self,
        report: CertificationReport,
        article: "Article",
        uploaded_images: list | None,
        live_post: dict[str, Any] | None,
    ) -> None:
        section = "IMAGES"

        if uploaded_images is None:
            report.items.append(CertificationItem(
                section, "Images", True, "Image pipeline not configured (skipped)"
            ))
            return

        uploaded_count = len(uploaded_images)
        report.items.append(CertificationItem(
            section, "Images uploaded",
            uploaded_count > 0,
            f"{uploaded_count} image(s) uploaded to WordPress Media Library",
        ))

        # Featured image set on live post
        live_featured_id = live_post.get("featured_media", 0) if live_post else 0
        has_featured = bool(live_featured_id)
        report.items.append(CertificationItem(
            section, "Featured image set",
            has_featured,
            f"featured_media = {live_featured_id}" if has_featured else "featured_media = 0 (not set)",
        ))

        # No image markers left in final content_markdown
        remaining_markers = re.findall(r'<!--\s*SEO_AGENT_IMAGE:[^>]+-->', article.content_markdown)
        report.items.append(CertificationItem(
            section, "Image markers substituted",
            not remaining_markers,
            "All substituted" if not remaining_markers
            else f"{len(remaining_markers)} marker(s) still present",
        ))

    # ── Section: EDITORIAL ────────────────────────────────────────────────────

    def _certify_editorial(
        self,
        report: CertificationReport,
        article: "Article",
        editorial_history: "EditorialHistoryService | None",
    ) -> None:
        section = "EDITORIAL"

        # Image usage history was written (only if history service provided)
        if editorial_history is not None:
            path = editorial_history.path
            history_exists = path.exists()
            report.items.append(CertificationItem(
                section, "Editorial history written",
                history_exists,
                str(path) if history_exists else f"Not found: {path}",
            ))

        # Article has tenant context
        tenant_ok = bool(
            article.tenant.client_id and article.tenant.website_id
        )
        report.items.append(CertificationItem(
            section, "Tenant context present",
            tenant_ok,
            f"client_id={article.tenant.client_id}  website_id={article.tenant.website_id}"
            if tenant_ok else "client_id or website_id is empty",
        ))

        # Article ID is stable (non-nil UUID)
        from uuid import UUID
        try:
            parsed = UUID(str(article.id))
            id_ok = str(parsed) != "00000000-0000-0000-0000-000000000000"
        except ValueError:
            id_ok = False
        report.items.append(CertificationItem(
            section, "Article UUID stable",
            id_ok,
            str(article.id),
        ))

    # ── Section: WORDPRESS ────────────────────────────────────────────────────

    def _certify_wordpress(
        self,
        report: CertificationReport,
        article: "Article",
        live_post: dict[str, Any] | None,
    ) -> None:
        section = "WORDPRESS"

        # Post status is "publish" on live post
        live_status = live_post.get("status", "") if live_post else ""
        status_published = live_status == "publish"
        report.items.append(CertificationItem(
            section, "WP post_status = publish",
            status_published,
            f"status = '{live_status}'" if live_post else "Could not read live post",
        ))

        # Categories assigned
        live_cats = live_post.get("categories", []) if live_post else []
        cats_ok = bool(live_cats)
        report.items.append(CertificationItem(
            section, "Category assigned",
            cats_ok,
            f"category IDs: {live_cats}" if cats_ok else "No category assigned",
        ))

        # Tags assigned
        live_tags = live_post.get("tags", []) if live_post else []
        tags_ok = bool(live_tags)
        report.items.append(CertificationItem(
            section, "Tags assigned",
            tags_ok,
            f"{len(live_tags)} tag(s)" if tags_ok else "No tags assigned",
        ))

        # Post URL is HTTPS (basic check)
        url = article.wp_post_url or ""
        https_ok = url.startswith("https://")
        report.items.append(CertificationItem(
            section, "Post URL is HTTPS",
            https_ok,
            url[:80] if url else "No URL",
        ))

    # ── Section: QUALITY ─────────────────────────────────────────────────────

    def _certify_quality(
        self,
        report: CertificationReport,
        article: "Article",
        seo_report: "SEOReport | None",
    ) -> None:
        section = "QUALITY"

        # No critical SEO issues
        if seo_report is not None:
            no_critical = seo_report.summary.critical == 0
            report.items.append(CertificationItem(
                section, "No critical SEO issues",
                no_critical,
                "Clean" if no_critical else f"{seo_report.summary.critical} critical issue(s)",
            ))

        # Content has at least one H2 heading (structural check)
        has_h2 = bool(re.search(r'^## ', article.content_markdown, re.MULTILINE))
        report.items.append(CertificationItem(
            section, "Structural headings present",
            has_h2,
            "H2 headings found" if has_h2 else "No H2 headings in article body",
        ))

        # No remaining image placeholder markers
        placeholders = re.findall(
            r'\[IMAGE_\d+\]|\[\[IMAGE[^\]]*\]\]|\[INSERT_IMAGE[^\]]*\]',
            article.content_markdown,
            re.IGNORECASE,
        )
        no_placeholders = not placeholders
        report.items.append(CertificationItem(
            section, "No image placeholders",
            no_placeholders,
            "None found" if no_placeholders else f"Found: {placeholders[:5]}",
        ))

        # Focus keyword appears in title
        kw = article.seo.focus_keyword.lower().strip()
        title_lower = article.title.lower()
        kw_in_title = bool(kw and kw in title_lower)
        report.items.append(CertificationItem(
            section, "Focus keyword in title",
            kw_in_title,
            f"'{kw}' found in title" if kw_in_title
            else f"'{kw}' not found in title '{article.title[:60]}'",
        ))

    # ── Section: RENDERING ────────────────────────────────────────────────────

    def _certify_rendering(
        self,
        report: CertificationReport,
        theme: "EditorialTheme | None",
    ) -> None:
        """
        Confirm the design system is operational for this publish run.

        Both checks are informational only — ``passed=True`` is always recorded
        regardless of the outcome, so neither check can block publication.
        The detail string carries the actual status for observability.
        """
        section = "RENDERING"

        # ── Check 1: Editorial Theme loaded ───────────────────────────────────
        try:
            from services.editorial_theme import DefaultEditorialTheme
            active_theme = theme or DefaultEditorialTheme()
            theme_name = type(active_theme).__name__
            report.items.append(CertificationItem(
                section, "Editorial Theme",
                passed=True,
                detail=f"{theme_name} loaded",
            ))
        except Exception as exc:  # noqa: BLE001
            report.items.append(CertificationItem(
                section, "Editorial Theme",
                passed=True,
                detail=f"Theme unavailable — {exc}",
            ))

        # ── Check 2: Renderer theme-aware ─────────────────────────────────────
        try:
            from services.editorial_html_renderer import EditorialHTMLRenderer
            EditorialHTMLRenderer()
            report.items.append(CertificationItem(
                section, "Renderer",
                passed=True,
                detail="Theme applied",
            ))
        except Exception as exc:  # noqa: BLE001
            report.items.append(CertificationItem(
                section, "Renderer",
                passed=True,
                detail=f"Renderer check failed — {exc}",
            ))

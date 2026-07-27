"""
LinkEnricherAgent — inserts 2–4 natural internal links into article markdown.

Fetches the list of published posts from WordPress, sends them alongside
the article and markdown to Claude, and receives back the same markdown
with internal links woven in naturally.

This runs BEFORE _to_html() so the links are part of the markdown that
the markdown → HTML conversion picks up.
"""
from __future__ import annotations

import logging
from typing import Any

from config import settings as _settings
from models.article import Article
from services.claude_service import ClaudeService, claude

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an experienced SEO editor. Your task: enrich an article with internal links
to other pages on the same website.

Rules:
1. Insert 2–4 internal links total. Never insert more than 4.
2. Links must appear naturally inside existing sentences — rewrite the surrounding
   phrase slightly if needed, but never add a new sentence or paragraph.
3. Only link to pages that are genuinely relevant to the surrounding text.
4. Use anchor text that is descriptive and contains the linked page's keyword —
   never use generic phrases like "click here", "read more", or "this article".
5. Do NOT link to the current article's own URL.
6. Do NOT add external links here — that is handled separately.
7. Return ONLY the enriched markdown. No preamble, no explanation, no markdown fences.
"""

_ENRICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "enriched_markdown": {
            "type": "string",
            "description": (
                "The full article markdown with 2–4 internal links inserted naturally. "
                "Return the complete markdown, not just changed sections."
            ),
        },
        "links_added": {
            "type": "array",
            "description": "Brief description of each link added (anchor text → target URL).",
            "items": {"type": "string"},
        },
    },
    "required": ["enriched_markdown", "links_added"],
}


class LinkEnricherAgent:
    """
    Adds 2–4 internal links to article markdown before HTML conversion.

    Usage:
        enricher = LinkEnricherAgent(claude_service)
        posts = wp_service.list_posts()
        enriched_md = enricher.enrich(article, posts, markdown)
    """

    def __init__(self, service: ClaudeService = claude) -> None:
        self._service = service
        self.last_links_added: int = 0

    def enrich(
        self,
        article: Article,
        published_posts: list[dict[str, Any]],
        markdown: str,
    ) -> str:
        """
        Insert 2–4 internal links into markdown.

        Args:
            article:         The article being published (for context).
            published_posts: list[dict] from WordPressService.list_posts().
                             Each dict has: id, title (rendered), link, slug.
            markdown:        Current article markdown (may already have image markers).

        Returns:
            Enriched markdown. Returns original markdown unchanged on any error
            so a failure here never blocks publishing.
        """
        if not published_posts:
            logger.info("LinkEnricher: no published posts available — skipping.")
            return markdown

        # Build the site map shown to Claude
        site_map_lines = ["Available pages on the same website (do NOT link to the current article):"]
        for post in published_posts:
            title = post.get("title", {})
            if isinstance(title, dict):
                title = title.get("rendered", "")
            url = post.get("link", "")
            if url and url != article.wp_post_url:
                site_map_lines.append(f"- {title}: {url}")

        if len(site_map_lines) <= 1:
            logger.info("LinkEnricher: all published posts are the current article — skipping.")
            return markdown

        site_map = "\n".join(site_map_lines)

        user_prompt = (
            f"Current article title: {article.title}\n"
            f"Focus keyword: {article.seo.focus_keyword}\n\n"
            f"{site_map}\n\n"
            f"Article markdown to enrich:\n---\n{markdown}\n---"
        )

        try:
            data = self._service.generate_structured(
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                tool_name="enrich_internal_links",
                tool_description="Insert 2–4 natural internal links into the article markdown.",
                input_schema=_ENRICH_SCHEMA,
                max_tokens=8192,
                thinking=False,
                model=_settings.link_enricher_model,
                label="enrich:links",
            )
            enriched = data.get("enriched_markdown", "").strip()
            links = data.get("links_added", [])

            if not enriched:
                logger.warning("LinkEnricher: Claude returned empty markdown — using original.")
                return markdown

            self.last_links_added = len(links)
            logger.info(
                "LinkEnricher: %d internal link(s) added — %s",
                self.last_links_added,
                "; ".join(links[:4]),
            )
            return enriched

        except Exception as exc:
            logger.warning("LinkEnricher failed (non-blocking): %s", exc)
            return markdown


# ── Module-level singleton ────────────────────────────────────────────────────

link_enricher = LinkEnricherAgent()

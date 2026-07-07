import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from models.article import Article, ArticleRequest, SEOMetadata
from models.enums import ArticleStatus
from models.tenant import TenantContext
from services.claude_service import ClaudeAPIError, ClaudeService, ClaudeRateLimitError, claude

logger = logging.getLogger(__name__)


class ArticleAgent:
    """
    Transforms an ArticleRequest into a complete Article using Claude.

    Two Claude calls:
      1. _generate_content() — streaming, full Markdown article
      2. _generate_seo()     — tool_use, structured SEOMetadata

    Knows nothing about WordPress, Drive, or publishing.
    """

    def __init__(self, service: ClaudeService = claude) -> None:
        self._service = service

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        request: ArticleRequest,
        tenant: TenantContext,
        on_content_ready: Callable[[str], None] | None = None,
    ) -> Article:
        """
        Generate a complete Article from an ArticleRequest.

        Steps:
          1. Generate article content via streaming
          2. Invoke on_content_ready callback (e.g. to save a checkpoint)
          3. Generate SEO metadata via tool_use (with retry on rate limit)
          4. Assemble and return Article with status=REVIEW
        """
        logger.info("Generating article: '%s'", request.topic)

        content = self._generate_content(request)
        logger.info("Content generated (%d words)", len(content.split()))

        if on_content_ready is not None:
            on_content_ready(content)

        seo = self._generate_seo(request, content)
        logger.info("SEO metadata generated: slug='%s'", seo.slug)

        return self._build_article(request, tenant, content, seo)

    # ── Content generation ────────────────────────────────────────────────────

    def _generate_content(self, request: ArticleRequest) -> str:
        system = self._build_system_prompt()
        messages = [{"role": "user", "content": self._build_content_prompt(request)}]
        return self._service.generate(system, messages)

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert SEO content writer specializing in local SEO "
            "for service businesses.\n\n"
            "Your writing must:\n"
            "- Be optimized for search engines while remaining natural and engaging "
            "for human readers\n"
            "- Use a clear H1 title followed by H2 and H3 headings to organize "
            "content logically\n"
            "- Include the focus keyword naturally in the first 100 words\n"
            "- Use short paragraphs (2–4 sentences) for web readability\n"
            "- Avoid keyword stuffing — prioritize quality prose over repetition\n"
            "- Always write in English — blog content is English-only, regardless of any other setting\n"
            "- Write evergreen content: avoid specific years (2024, 2025, 2026, etc.) "
            "in titles, headings, and body text. A title like 'Complete Guide to Garage Door "
            "Repair' outlasts 'Garage Door Repair: 2025 Guide' and never becomes dated. "
            "Exception: a year is appropriate only when the topic is explicitly tied to a "
            "calendar year — a new law, an annual report, a year-specific event.\n\n"
            "Editorial structure — use these elements where they add genuine value:\n"
            "- Bullet or numbered lists for steps, tips, or comparisons (3+ items)\n"
            "- A simple Markdown table when comparing options, costs, or features\n"
            "- A callout block (> ⚠️ **Important:** ...) for critical safety notes or warnings\n"
            "- A FAQ section (## Preguntas frecuentes / ## FAQ) near the end with 3–5 Q&A pairs "
            "targeting common reader questions and long-tail keywords\n"
            "- Bold key phrases on first mention so scanners can grasp the article quickly\n\n"
            "Process: before writing, mentally outline the article structure. "
            "Then write the complete article in a single, coherent pass."
        )

    def _build_content_prompt(self, request: ArticleRequest) -> str:
        lines: list[str] = [
            f"Write a {request.word_count}-word article about: {request.topic}",
        ]

        if request.service:
            lines.append(f"Service: {request.service}")

        if request.location:
            loc = request.location
            lines.append(f"Target location: {loc.city}, {loc.state}, {loc.country}")
            if loc.neighborhood:
                lines.append(f"Neighborhood: {loc.neighborhood}")

        if request.objective:
            lines.append(f"Objective: {request.objective}")

        if request.target_audience:
            lines.append(f"Target audience: {request.target_audience}")

        lines.append(f"Tone: {request.tone.value}")
        lines.append(f"Language: {request.language.value}")

        if request.focus_keyword:
            lines.append(f"Primary keyword to target: {request.focus_keyword}")
            lines.append(
                "Keyword placement — distribute naturally, never force or repeat mechanically:\n"
                "  - H1 title: include the keyword or a close natural variant\n"
                "  - Introduction: mention the keyword in the first paragraph\n"
                "  - Body: use it in at least one H2 heading when it fits the topic\n"
                "  - Conclusion: use it once in the closing paragraph\n"
                "  - FAQ (when included): weave it into at least one question or answer\n"
                "  - Target density: 1–2% of total words — never exceed 2% or repeat "
                "the exact phrase more than once in the same paragraph"
            )

        if request.internal_links_to_include:
            lines.append("Internal links to include naturally in the text:")
            for url in request.internal_links_to_include:
                lines.append(f"  - {url}")

        lines += [
            "",
            "Structure the article with a logical flow: a hook introduction, "
            "informative body sections with H2 headings, and a clear conclusion "
            "with a call to action.",
            "",
            "External links: add 1–2 external links to authoritative, non-competing sources "
            "(government agencies, manufacturer websites, industry associations, safety standards). "
            "Use natural anchor text — never 'click here'. Omit if no truly relevant source exists.",
            "",
            "Return ONLY the article in Markdown format. "
            "Start with the H1 title (# Title). No preamble, no meta-commentary.",
        ]

        return "\n".join(lines)

    # ── SEO generation ────────────────────────────────────────────────────────

    def _generate_seo(self, request: ArticleRequest, content: str) -> SEOMetadata:
        """Generate structured SEO metadata, retrying up to 3 times on rate limit errors."""
        schema = self._build_seo_schema()
        messages = self._build_seo_messages(request, content)

        for attempt in range(3):
            try:
                raw = self._service.generate_structured(
                    system=self._build_system_prompt(),
                    messages=messages,
                    tool_name="generate_seo_metadata",
                    tool_description=(
                        "Generate complete SEO metadata for the article. "
                        "Leave seo_plugin_score as null — it is set by WordPress after publish."
                    ),
                    input_schema=schema,
                )
                return SEOMetadata(**raw)
            except ValidationError as exc:
                raise ClaudeAPIError(
                    f"Claude returned malformed SEO metadata "
                    f"({exc.error_count()} validation error(s))."
                ) from exc
            except ClaudeRateLimitError:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Rate limit on SEO attempt %d/3 — retrying in %ds", attempt + 1, wait
                )
                time.sleep(wait)

        raise ClaudeRateLimitError("SEO generation failed after 3 attempts.")

    def _build_seo_messages(self, request: ArticleRequest, content: str) -> list[dict[str, Any]]:
        hints: list[str] = [f"Topic: {request.topic}"]
        if request.focus_keyword:
            hints.append(f"Keyword hint: {request.focus_keyword}")
        if request.location:
            loc = request.location
            hints.append(f"Location: {loc.city}, {loc.state}")
        if request.service:
            hints.append(f"Service: {request.service}")

        prompt = (
            "Based on the article below, generate complete and accurate SEO metadata "
            "optimized for local search.\n\n"
            "Evergreen rule: do NOT include specific years (2024, 2025, 2026, etc.) in the "
            "seo_title, slug, or focus_keyword unless the article topic is explicitly "
            "year-dependent. Prefer timeless titles: 'Complete Guide', 'What Homeowners "
            "Should Know', 'Pricing Factors Explained' — not 'Year Guide' or 'Year Prices'.\n\n"
            f"ORIGINAL REQUEST:\n{chr(10).join(hints)}\n\n"
            f"ARTICLE:\n---\n{content}\n---"
        )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _build_seo_schema() -> dict[str, Any]:
        schema = SEOMetadata.model_json_schema()
        schema.pop("title", None)
        return schema

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _build_article(
        self,
        request: ArticleRequest,
        tenant: TenantContext,
        content: str,
        seo: SEOMetadata,
    ) -> Article:
        title = self._extract_title(content) or request.topic
        body = self._strip_h1(content)

        return Article(
            tenant=tenant,
            request=request,
            status=ArticleStatus.REVIEW,
            title=title,
            content_markdown=body,
            seo=seo,
            model_name=self._service.model,
            prompt_version="1.0",
        )

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
        return None

    @staticmethod
    def _strip_h1(content: str) -> str:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("# "):
                remaining = lines[i + 1:]
                while remaining and not remaining[0].strip():
                    remaining.pop(0)
                return "\n".join(remaining)
        return content


# ── Module-level singleton ────────────────────────────────────────────────────

article_agent = ArticleAgent()

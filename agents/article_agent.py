import logging
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from models.article import Article, ArticleRequest, SEOMetadata
from models.article_plan import ArticlePlan
from models.enums import ArticleStatus
from models.errors import ArticleValidationError
from models.tenant import TenantContext
from config import settings
from services.article_planner_service import ArticlePlannerService
from services.claude_service import ClaudeAPIError, ClaudeService, ClaudeRateLimitError, claude

logger = logging.getLogger(__name__)

# Matches bracket placeholder tokens that must never reach generation.
_PLACEHOLDER_RE = re.compile(
    r'\[(?:City|Service|Keyword|Topic|TOPIC|Location|Business|State|Country|Name|Date|Year|Niche)\]',
    re.IGNORECASE,
)


# ── Generator system prompt ───────────────────────────────────────────────────
# Used for article content generation only.
# _build_system_prompt() (below) is kept separately for SEO metadata generation.

_GENERATOR_SYSTEM = """\
You are an expert SEO content writer specializing in local SEO for service businesses.

When you receive a TECHNICAL REASONING PLAN, your role is prose transformation:
take the structured expert reasoning and write natural, publication-quality prose from it.
Every technical claim, local detail, and authenticity signal in the article comes from
the plan — you transform that reasoning into prose; you do not invent it.

When no plan is provided, generate the article directly from topic and requirements.

══════════════════════════════════════════════════
HOW TO USE THE REASONING PLAN
══════════════════════════════════════════════════

Transform — never copy.
Every field in the plan is reasoning material, not prose. Rewrite it as natural sentences.

Section structure: the plan's section_plans appear in document order. Write them in order.
Use the heading exactly as given in the plan.

Counter-intuitive facts: reveal them naturally as part of the technical explanation.
Never announce them with "surprisingly," "interestingly," or "you might be shocked to learn."

Why-not examples: write them as — name the approach → state the prohibition → explain the
exact mechanical reason. This is the strongest authenticity signal in the plan.
Every why-not in the plan must appear in the article.

Specific numbers: every quantitative fact in the plan's specific_numbers lists MUST appear
in the article. These are the plan's primary authenticity anchors.

Local factors: embed them into technical arguments, never as standalone regional trivia.
"In a freeze-thaw climate, silicone resists grit accumulation better than WD-40" is correct
because it grounds the recommendation in a mechanical reason. "{resolved_city} has cold winters"
is not — that's a city mention, not local grounding. The city from TARGET LOCATION should
change the technical argument, not just appear as a name drop.

Realistic limitations: acknowledge them honestly. Expert credibility comes from acknowledging
what can go wrong, not from promising perfect outcomes.

Professional insight: write this as natural expertise — the kind of thing a technician
would say that makes a reader think "that person has actually done this work."

══════════════════════════════════════════════════
WRITING REQUIREMENTS
══════════════════════════════════════════════════

Structure:
• H1 title followed by H2 and H3 headings in logical order
• Focus keyword in first paragraph, at least one H2 heading, and conclusion
• Keyword density 1–2% — never exceed 2% or repeat in the same paragraph
• Short paragraphs (2–4 sentences) for web readability
• Bold key phrases on first mention so scanners can grasp the article quickly

Editorial elements (use when they add genuine value):
• Bullet or numbered lists for steps, tips, or comparisons (3+ items)
• A simple Markdown table when comparing options, costs, or features
• A callout block (> ⚠️ **Important:** ...) for critical safety notes or warnings
• A FAQ section (## FAQ or ## Frequently Asked Questions) near the end with 5 Q&A pairs

Prose quality:
• Vary sentence length deliberately: short punchy sentences (8–10 words) mixed with
  longer explanatory ones (25–35 words)
• At least one standalone single-sentence paragraph used for emphasis
• Start some sentences with "And" or "But" where it flows naturally
• Use contractions throughout: "you'll", "it's", "don't", "we've", "that's"
• Deliberately mix two-sentence paragraphs with longer ones

BANNED TRANSITIONS — must never appear anywhere in the output:
  Furthermore, Moreover, Additionally, In addition, In conclusion, It's worth noting,
  It's important to note, Needless to say, First and foremost, At the end of the day,
  When it comes to, Without a doubt, It goes without saying, To summarize

BANNED OPENERS — must not open any sentence:
  "In today's world...", "In today's digital landscape...", "In today's [anything]..."

Evergreen: avoid specific years in titles, headings, and body text unless the topic
is explicitly tied to a calendar year.

Always write in English — blog content is English-only.

IMAGE REFERENCE PROHIBITION — ABSOLUTE RULE:
Never write phrases that reference visual elements as body prose:
  BANNED: "as shown", "as pictured", "as illustrated", "in the photo", "the image shows",
  "see the comparison", "as you can see in the image", "shown above", "pictured below"
Never write a caption, label, or description of what an image would show as a standalone paragraph
or sentence adjacent to an image placeholder marker. The marker <!-- SEO_AGENT_IMAGE: id -->
stands alone on its own line — no surrounding caption text before or after it.
When an E-E-A-T opportunity involves a visual comparison (e.g., comparing two component types),
express the underlying technical fact in prose — explain WHY one is better, not what a photo would
depict. Example: instead of "corroded galvanized spring next to clean oil-tempered spring" as prose,
write about the metallurgical reason one corrodes faster in coastal humidity.

══════════════════════════════════════════════════
CONCLUSION GUIDELINES
══════════════════════════════════════════════════

The closing paragraph is an editorial conclusion — not a promotional CTA.

Write it as the last word from a subject-matter expert:
  • Circle back to the article's opening hook or thesis
  • Reinforce the single most important practical takeaway
  • Acknowledge a realistic limitation the reader should keep in mind
  • End with expert insight — the kind of final thought a 15-year technician would leave the reader with

One brief action step is acceptable if it flows naturally from the editorial close.

NEVER in the conclusion:
  • Open with "Contact our team...", "Call us...", or "Schedule a free..."
  • List contact methods, phone numbers, or estimate language
  • Introduce new geographic locations that weren't discussed in the article body
  • Add a city or neighborhood name just for local SEO ("from Chula Vista to La Jolla")
  • Repeat the focus keyword mechanically as a final drop
  • Write more than one CTA sentence

The conclusion must match the tone, expertise, and quality of the best paragraph in the article body.
A weak conclusion written in marketing language undermines an otherwise strong article.

══════════════════════════════════════════════════
LOCAL CONTEXT POLICY
══════════════════════════════════════════════════

USE CONFIDENTLY — no hedging required for broadly factual regional knowledge:
  • Climate and seasonal conditions (frost depth, humidity, precipitation, heat)
  • Typical housing styles, ages, and construction materials common to the area
  • Named neighborhoods, districts, or suburbs
  • Common infrastructure characteristics (age of housing stock, soil type, freeze-thaw)
  • Regional weather patterns, terrain, or environmental conditions
  • Common service problems associated with that climate or region
  • State or municipal regulations that are broadly established
  • Local building practices that are widely known

LOCATION CONSISTENCY RULE:
Never introduce a new city, suburb, neighborhood, or district in the conclusion or anywhere
else in the article that was not established earlier in the same article.
Adding a list of service-area cities at the end of an article ("homeowners from Chula Vista
to La Jolla...") is geographic padding, not local SEO — it sounds randomly generated because
it is. Reference a location in the conclusion ONLY if that location was introduced naturally
earlier in the article body.

NEVER INVENT:
  • Customers, homeowners, or clients — by name, description, or implication
  • Testimonials, reviews, or quotes — real or hypothetical
  • Case studies, completed jobs, or project outcomes
  • Personal experiences or conversations
  • Statistics without a cited authoritative source
  • Fictional events or scenarios presented as real

BANNED PHRASES — never write:
  "We recently helped..."
  "A local homeowner told us..."
  "One of our customers..."
  "In a recent project..."
  "Last [season/month/winter]..."
  "A homeowner in [neighborhood] called us..."
  "Recently we completed a job where..."

══════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════

Return ONLY the article in Markdown format.
Start with the H1 title: # Title Here
No preamble, no meta-commentary, no "Here is the article:" header.
Begin directly with #.

══════════════════════════════════════════════════
PLACEHOLDER PROHIBITION — ABSOLUTE RULE
══════════════════════════════════════════════════

NEVER write bracket placeholders in any output.
Always use the exact city, service, and values from the task brief and plan.
No city name appears in these instructions for you to copy.

  WRONG: "{resolved_service} repair [{resolved_city}]"
  RIGHT: use the actual service from BUSINESS NICHE and city from TARGET LOCATION

  WRONG: "homeowners in [{resolved_city}]"
  RIGHT: "homeowners in " + the city from TARGET LOCATION

Every city, state, and service name in the article must come from the task brief
or the technical reasoning plan — never from these instructions.

A title, heading, sentence, or keyword phrase containing [City], [Service],
[Location], [Keyword], or any other bracket placeholder is a hard error.\
"""


# ── SEO metadata validation helpers ──────────────────────────────────────────

# Fields that can be auto-repaired by truncation. Values are the max_length
# constraints declared on SEOMetadata. Any change here must stay in sync with
# models/article.py:SEOMetadata.
_SEO_LENGTH_LIMITS: dict[str, int] = {
    "seo_title": 70,
    "meta_description": 170,
}


def _truncate_to_words(text: str, max_length: int) -> str:
    """Truncate *text* to at most *max_length* chars, preserving whole words."""
    if len(text) <= max_length:
        return text
    cut = text[:max_length]
    last_space = cut.rfind(" ")
    return (cut[:last_space] if last_space > 0 else cut).rstrip()


def _seo_field_analysis(raw: dict[str, Any]) -> str:
    """Return a multi-line diagnostic string describing every field in *raw*."""
    lines: list[str] = []
    for field, value in raw.items():
        lines.append(f"  {field}")
        display = repr(value) if not isinstance(value, str) else repr(value[:80] + "…" if len(value) > 80 else value)
        lines.append(f"    value:  {display}")
        lines.append(f"    type:   {type(value).__name__}")
        if isinstance(value, str):
            length = len(value)
            lines.append(f"    length: {length}")
            if field in _SEO_LENGTH_LIMITS:
                limit = _SEO_LENGTH_LIMITS[field]
                over = length - limit
                tag = f"  ← EXCEEDS LIMIT by {over}" if over > 0 else "  ← OK"
                lines.append(f"    max_length: {limit}{tag}")
        elif field in _SEO_LENGTH_LIMITS:
            lines.append(f"    max_length: {_SEO_LENGTH_LIMITS[field]}  ← value is not a string")
    return "\n".join(lines)


class ArticleAgent:
    """
    Transforms an ArticleRequest into a complete Article using Claude.

    Three Claude calls:
      1. _plan_article()     — ArticlePlannerService: structured technical reasoning
      2. _generate_content() — streaming Markdown article (consumes plan)
      3. _generate_seo()     — tool_use structured SEOMetadata

    The planner reasons like a domain expert before the writer begins. The generator
    transforms that reasoning into prose — it never invents facts beyond the plan.
    If planning fails, generation falls back to the original unplanned path.

    Knows nothing about WordPress, Drive, or publishing.
    """

    def __init__(self, service: ClaudeService = claude) -> None:
        self._service = service
        self._planner = ArticlePlannerService(service)

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
          0. Validate request (raises ArticleValidationError on failure)
          1. Build a technical reasoning plan (ArticlePlannerService)
          2. Generate article content via streaming (consumes plan)
          3. Invoke on_content_ready callback (e.g. to save a checkpoint)
          4. Generate SEO metadata via tool_use (with retry on rate limit)
          5. Assemble and return Article with status=REVIEW
        """
        self._validate_request(request)
        logger.info("Generating article: '%s'", request.topic)

        plan = self._plan_article(request)

        content = self._generate_content(request, plan)
        logger.info("Content generated (%d words)", len(content.split()))

        if on_content_ready is not None:
            on_content_ready(content)

        # Validate and repair structure before any downstream processing.
        from services.structure_template_service import ArticleStructureService
        content = ArticleStructureService.validate_and_repair(content)

        seo = self._generate_seo(request, content)
        logger.info("SEO metadata generated: slug='%s'", seo.slug)

        return self._build_article(request, tenant, content, seo, plan=plan)

    # ── Planning stage ────────────────────────────────────────────────────────

    def _plan_article(self, request: ArticleRequest) -> ArticlePlan | None:
        """
        Build a technical reasoning plan before content generation.

        Returns None on failure or if the plan contains unresolved placeholders,
        so generation falls back to the unplanned path.
        """
        t0 = time.perf_counter()
        plan = self._planner.plan(request)
        elapsed = time.perf_counter() - t0

        if plan is None:
            logger.warning("Planning skipped — generating without plan (%.1fs)", elapsed)
            return None

        violations = self._find_plan_placeholders(plan)
        if violations:
            logger.warning(
                "Planner generated bracket placeholders — discarding plan and falling back "
                "to unplanned generation. Violations: %s",
                violations,
            )
            return None

        logger.info(
            "Plan built in %.1fs: %d sections, %d FAQs",
            elapsed,
            len(plan.section_plans),
            len(plan.faq_plans),
        )
        return plan

    # ── Pre-generation validation ─────────────────────────────────────────────

    @staticmethod
    def _validate_request(request: ArticleRequest) -> None:
        """
        Validate the request before any LLM call.

        Enforces two rules:
          1. Business context must be fully resolved — city, state, and service
             must be present. The LLM must never infer or invent these from examples.
          2. No bracket placeholders in topic or focus_keyword.

        BusinessContextResolver should be called before this to satisfy rule 1.
        """
        issues: list[str] = []

        # Location is intentionally NOT a hard requirement here.
        # BusinessContextResolver attempts resolution before this call; if it
        # could not resolve a city/state, generation continues as a non-local
        # article. The publish gate (PublisherAgent.validate()) enforces location.

        # Rule — no bracket placeholders.
        if _PLACEHOLDER_RE.search(request.topic):
            issues.append(
                f"Topic contains an unresolved placeholder: '{request.topic}' — "
                "replace [City], [Service], etc. with actual values"
            )
        if request.focus_keyword and _PLACEHOLDER_RE.search(request.focus_keyword):
            issues.append(
                f"Focus keyword contains an unresolved placeholder: '{request.focus_keyword}' — "
                "replace [City], [Service], etc. with actual values"
            )

        if issues:
            n = len(issues)
            raise ArticleValidationError(
                f"Article generation rejected — {n} validation error{'s' if n > 1 else ''}:\n"
                + "\n".join(f"  • {issue}" for issue in issues)
            )

    @staticmethod
    def _find_plan_placeholders(plan: ArticlePlan) -> list[str]:
        """
        Return a list of bracket placeholder violations in the plan.
        An empty list means the plan is clean.
        """
        violations: list[str] = []

        for field in ("article_thesis", "hook_angle", "focus_keyword"):
            val = getattr(plan, field, "")
            if _PLACEHOLDER_RE.search(val):
                violations.append(f"plan.{field}: '{val}'")

        for i, sec in enumerate(plan.section_plans):
            if _PLACEHOLDER_RE.search(sec.heading):
                violations.append(f"plan.section_plans[{i}].heading: '{sec.heading}'")

        for i, faq in enumerate(plan.faq_plans):
            if _PLACEHOLDER_RE.search(faq.question):
                violations.append(f"plan.faq_plans[{i}].question: '{faq.question}'")

        return violations

    # ── Content generation ────────────────────────────────────────────────────

    def _generate_content(
        self,
        request: ArticleRequest,
        plan: ArticlePlan | None = None,
    ) -> str:
        thinking_value = plan is None
        messages = [{"role": "user", "content": self._build_generator_prompt(request, plan)}]
        return self._service.generate(
            _GENERATOR_SYSTEM, messages,
            thinking=thinking_value,
            label="generate:article",
        )

    def _build_generator_prompt(
        self,
        request: ArticleRequest,
        plan: ArticlePlan | None,
    ) -> str:
        parts: list[str] = []

        if plan is not None:
            parts += [plan.to_generator_block(), ""]

        parts.append(f"Write a {request.word_count}-word article about: {request.topic}")

        if request.service:
            parts.append(f"Service: {request.service}")

        if request.location:
            loc = request.location
            parts.append(f"Target location: {loc.city}, {loc.state}, {loc.country}")
            if loc.neighborhood:
                parts.append(f"Neighborhood: {loc.neighborhood}")

        if request.objective:
            parts.append(f"Objective: {request.objective}")

        if request.target_audience:
            parts.append(f"Target audience: {request.target_audience}")

        parts.append(f"Tone: {request.tone.value}")
        parts.append(f"Language: {request.language.value}")

        if request.focus_keyword:
            parts += [
                f"Primary keyword to target: {request.focus_keyword}",
                (
                    "Keyword placement — distribute naturally, never force or repeat mechanically:\n"
                    "  - H1 title: include the keyword or a close natural variant\n"
                    "  - Introduction: mention the keyword in the first paragraph\n"
                    "  - Body: use it in at least one H2 heading when it fits the topic\n"
                    "  - Conclusion: use it once in the closing paragraph\n"
                    "  - FAQ (when included): weave it into at least one question or answer\n"
                    "  - Target density: 1–2% of total words — never exceed 2% or repeat "
                    "the exact phrase more than once in the same paragraph"
                ),
            ]

        if request.internal_links_to_include:
            parts.append("Internal links to include naturally in the text:")
            for url in request.internal_links_to_include:
                parts.append(f"  - {url}")

        if plan is None:
            parts += [
                "",
                "External links: add 1–2 external links to authoritative, non-competing sources "
                "(government agencies, manufacturer websites, industry associations, safety standards). "
                "Use natural anchor text — never 'click here'. Omit if no truly relevant source exists.",
            ]
        else:
            parts += [
                "",
                "External links: cite the authority sources listed in the plan where they fit. "
                "Add up to 2 additional authoritative links if relevant. "
                "Use natural anchor text — never 'click here'.",
            ]

        # ── Structural scaffold (from templates/article_structure.json) ────────
        from services.structure_template_service import ArticleStructureService
        structure_block = ArticleStructureService.build_structure_prompt()
        if structure_block:
            parts += ["", structure_block]

        parts += [
            "",
            "Return ONLY the article in Markdown format. "
            "Start with the H1 title (# Title). No preamble, no meta-commentary.",
        ]

        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """System prompt for SEO metadata generation — not article content."""
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
            "calendar year — a new law, an annual report, a year-specific event."
        )

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
                    model=settings.seo_model,
                    label="seo:metadata",
                )
                return SEOMetadata(**raw)
            except ValidationError as exc:
                # ── Diagnostic logging ─────────────────────────────────────
                logger.error(
                    "SEO validation failed\n"
                    "----------------------------------------\n"
                    "RAW CLAUDE RESPONSE\n"
                    "----------------------------------------\n"
                    "%s\n\n"
                    "----------------------------------------\n"
                    "PYDANTIC VALIDATION ERRORS\n"
                    "----------------------------------------\n"
                    "  error_count: %d\n"
                    "  errors:      %s\n"
                    "  json:        %s\n\n"
                    "----------------------------------------\n"
                    "FIELD ANALYSIS\n"
                    "----------------------------------------\n"
                    "%s",
                    raw,
                    exc.error_count(),
                    exc.errors(),
                    exc.json(),
                    _seo_field_analysis(raw),
                )

                # ── Auto-repair: length-only violations ────────────────────
                # Identifies whether every failing field is a string_too_long
                # error on seo_title or meta_description. If so, truncates to
                # the declared max_length (word-boundary-preserving) and
                # re-validates. Any other error type skips straight to raise.
                length_error_fields = {
                    e["loc"][0]
                    for e in exc.errors()
                    if e.get("type") == "string_too_long"
                    and len(e.get("loc", ())) == 1
                    and isinstance(e.get("loc", (None,))[0], str)
                }
                only_length_errors = bool(length_error_fields) and length_error_fields.issubset(
                    _SEO_LENGTH_LIMITS
                )

                if only_length_errors:
                    repaired = dict(raw)
                    repair_log: list[str] = []
                    for field in length_error_fields:
                        original = repaired.get(field)
                        if isinstance(original, str):
                            truncated = _truncate_to_words(original, _SEO_LENGTH_LIMITS[field])
                            repaired[field] = truncated
                            repair_log.append(
                                f"  {field}: {len(original)} chars → {len(truncated)} chars"
                                f' ("{truncated[:50]}{"…" if len(truncated) > 50 else ""}")'
                            )

                    logger.warning(
                        "SEO auto-repair: truncating length-violating fields\n%s",
                        "\n".join(repair_log),
                    )

                    try:
                        result = SEOMetadata(**repaired)
                        logger.warning(
                            "SEO auto-repair succeeded — continuing with repaired metadata."
                        )
                        return result
                    except ValidationError as repair_exc:
                        logger.error("SEO auto-repair did not resolve all validation errors.")
                        exc = repair_exc

                # ── Detailed exception ─────────────────────────────────────
                field_lines: list[str] = []
                for error in exc.errors():
                    loc = ".".join(str(part) for part in error.get("loc", []))
                    msg = error.get("msg", "unknown")
                    ctx = error.get("ctx", {})
                    entry = f"- {loc}\n  reason: {msg}"
                    if "max_length" in ctx:
                        raw_val = raw.get(loc)
                        actual = len(raw_val) if isinstance(raw_val, str) else "N/A"
                        entry += (
                            f"\n  expected <= {ctx['max_length']}"
                            f"\n  actual   =  {actual}"
                        )
                    field_lines.append(entry)

                raise ClaudeAPIError(
                    "Claude returned invalid SEO metadata.\n\n"
                    f"Failed fields:\n" + "\n".join(field_lines) + "\n\n"
                    f"Raw validation:\n{exc}"
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
            "Placeholder rule: NEVER use [City], [Service], [Location], [Keyword], or any "
            "bracket placeholder in seo_title, slug, or focus_keyword. Always use the actual "
            "city name and service name from the article.\n\n"
            "Tag quality rule: suggested_tags must be HIGHLY SPECIFIC to THIS article. "
            "Each tag should name a specific repair type, component, failure mode, cost factor, "
            "or technical concept discussed in the article. "
            "NEVER use generic tags like: the city name alone, the service name alone, "
            "'Home Maintenance', 'Home Improvement', 'Home Repair', 'DIY', or any tag that "
            "could apply to any article on the same broad topic. "
            "GOOD examples: 'Torsion Spring Replacement', 'Garage Door Spring Cost', "
            "'Broken Spring Diagnosis', 'Cable Drum Failure', 'Garage Door Safety'. "
            "BAD examples: 'San Diego', 'Garage Door', 'Home Maintenance', 'Repair'. "
            "Limit to 5–8 tags, each earning its place by naming something specific "
            "that appears in THIS article.\n\n"
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
        plan: ArticlePlan | None = None,
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
            image_plans=plan.image_plans if plan else [],
            topic_id=self._make_topic_id(request.topic, request),
        )

    @staticmethod
    def _make_topic_id(topic: str, request: ArticleRequest) -> str:
        """
        Generate a stable, location-agnostic topic identifier in kebab-case.

        Delegates to services.topic_normalization.normalize_topic_id for
        deterministic, semantically-normalized output.  See that module for
        the full algorithm and synonym dictionary.

        Examples:
          "Garage Door Spring Repair Denver CO"   → "door-garage-repair-spring"
          "Broken Garage Door Springs Denver"     → "door-garage-repair-spring"
          "Repair Garage Door Spring"             → "door-garage-repair-spring"
          "Garage Door Opener Repair"             → "door-garage-opener-repair"
          "Broken Garage Door Opener"             → "door-garage-opener-repair"
        """
        from services.topic_normalization import normalize_topic_id
        return normalize_topic_id(topic, request.location)

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

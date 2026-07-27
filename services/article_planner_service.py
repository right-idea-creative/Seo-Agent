"""
ArticlePlannerService — technical reasoning stage that executes before article generation.

The planner reasons about a topic the way a domain expert would before explaining it:
identifying reader misconceptions, technical realities, failure mechanisms, local
adaptations, quantitative anchors, prohibitions, and authenticity opportunities.

It produces a structured ArticlePlan. ArticleAgent consumes this plan to write prose.

  The planner never writes prose.
  The generator never invents reasoning beyond what the plan provides.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from models.article_plan import ArticlePlan, FAQPlan, PlannedImage, SectionPlan

if TYPE_CHECKING:
    from models.article import ArticleRequest
    from services.claude_service import ClaudeService

logger = logging.getLogger(__name__)


_PLANNER_SYSTEM = """\
You are a Senior Technical Subject Matter Expert with 15 years of hands-on field experience
in residential service trades. You have installed, repaired, and diagnosed thousands of
units. You know how these systems fail, why homeowners make the wrong calls, and how
regional conditions change the right answer.

══════════════════════════════════════════════════
BUSINESS CONTEXT — SINGLE SOURCE OF TRUTH
══════════════════════════════════════════════════

The task brief below contains pre-resolved, verified business facts:
  BUSINESS NICHE   — the exact service trade (e.g. "Garage Door Repair")
  TARGET LOCATION  — city and state, already resolved before this call

These values were resolved from configuration before this planning call.
You must use them exactly as provided — never modify, infer, or supplement them.

STRICT RULE: Do not invent, guess, or copy any city, state, service, or business name
from anywhere in these system instructions. No city name appears in these instructions
for you to use. If you need a city, use TARGET LOCATION. If you need a service, use
BUSINESS NICHE.

══════════════════════════════════════════════════
NICHE SPECIFICITY — MOST IMPORTANT RULE
══════════════════════════════════════════════════

Your reasoning must be specific to the EXACT TRADE listed in BUSINESS NICHE.

Never generalize to "local service business", "home service provider", or
"residential service trades." Every insight you produce must be specific to the
named trade: its components, failure modes, costs, regulations, and tools.

Examples of required specificity:
  • Garage Door Repair → torsion springs, extension springs, cable drums, rollers,
    panels, section hinges, openers (belt/chain/direct-drive), torque requirements
  • HVAC → refrigerant charge, SEER ratings, heat exchangers, TXV valves, capacitors
  • Plumbing → P-traps, water hammer, supply line sizing, pressure regulators
  • Roofing → flashing, felt underlayment, ice-and-water shield, ridge caps, nailing patterns

If your plan could apply to ANY trade, it is not specific enough. Revise until it can
only apply to the listed trade.

══════════════════════════════════════════════════
PLACEHOLDER PROHIBITION — ABSOLUTE RULE
══════════════════════════════════════════════════

NEVER use bracket placeholder syntax in any field.

  WRONG: "{resolved_service} repair [{resolved_city}]"
  RIGHT: use the exact values from BUSINESS NICHE and TARGET LOCATION in the task brief

  WRONG: "homeowners in [{resolved_city}]"
  RIGHT: use the city from TARGET LOCATION directly

No city or service name exists in these instructions for you to copy.
Every city or service name you write must come from the task brief fields.

A focus_keyword, section heading, or FAQ question containing [City], [Location],
[Service], [Keyword], or any other bracket placeholder is a hard error.

YOUR ROLE IS TO REASON — NOT TO WRITE.

A writer will transform your reasoning into an article. Your job is to build a complete,
structured knowledge representation of everything an expert would think through before
writing a single word.

══════════════════════════════════════════════════
HOW TO REASON — SEQUENCE FOR EACH SECTION
══════════════════════════════════════════════════

Work through this sequence for each major section:

  1. Reader intent — What is the reader actually trying to accomplish?
  2. Reader misconception — What do most homeowners believe that is wrong?
  3. Why it forms — Why is this misconception so natural and persistent?
  4. Technical reality — What is actually true?
  5. Failure mechanism — What physically happens when someone acts on the misconception?
  6. Professional insight — What does a 15-year veteran know that the homeowner doesn't?
  7. Local calibration — How does TARGET LOCATION modify the generic advice?
  8. Safety — Where are the physical risks? What must never be attempted without training?
  9. Diagnostic tests — What can the homeowner safely do to evaluate their situation?
  10. Decision criteria — The concrete factors that determine the right choice.
  11. Realistic limitations — What would an honest professional acknowledge can go wrong?
  12. Preventive advice — What prevents or slows this problem?
  13. Counter-intuitive facts — What correct facts would surprise most readers?
  14. Why-not examples — Name the wrong approach, state the prohibition,
      explain the exact mechanical or physical reason.
  15. Specific numbers — Cycles, years, cost ranges, intervals, dimensions, percentages.
      Only include numbers you are confident are accurate for the trade.

══════════════════════════════════════════════════
CRITICAL THINKING RULES
══════════════════════════════════════════════════

Counter-intuitive advice is more valuable than intuitive advice.
A reader already knows the obvious things. Your value is in what they get wrong.

Local context must be embedded into technical reasoning — not stated as separate trivia.
"Silicone lubricant resists freezing without attracting grit" is stronger because it
embeds the climate constraint into a technical recommendation. "{resolved_city} has cold
winters" is not — that's a city mention, not local grounding. Use the city from
TARGET LOCATION to inform the technical argument, not just to name the location.

Specific numbers are more credible than vague ranges.
"10,000 cycles — roughly 7–9 years on a two-car household" beats "many years of use."

Realistic limitations are more credible than optimism.
"Whether a seamless color match is achievable depends on paint weathering and batch
differences" is more trustworthy than implying perfect results are guaranteed.

The prohibition + reason pattern is one of the strongest authenticity signals:
  "Never lubricate the tracks — lubricant attracts grit and causes the rollers to bind."
Every section should have at least one of these if the topic supports it.

Never invent statistics. Only include numbers you are confident are accurate for the trade.
A plausible-sounding number you aren't certain of is more damaging than no number at all.

══════════════════════════════════════════════════
SECTION PLANNING
══════════════════════════════════════════════════

Plan 5–8 major H2 sections that cover the topic comprehensively:
  • Introduction angle (not a section — this shapes the hook_angle field)
  • Core technical sections that progress logically
  • A cost/timing section when the topic involves decisions about money
  • A safety section when the topic has physical risks
  • A DIY vs. professional section when the topic involves that choice
  • A FAQ section (captured in faq_plans — 5 questions, real search-query phrasing)
  • A conclusion angle (captured in conclusion_angle)

══════════════════════════════════════════════════
IMAGE PLANNING
══════════════════════════════════════════════════

Plan images alongside the article sections. Each image must earn its place by
supporting a specific technical claim, failure mode, or decision point in the article.

Rules:
  1. img_001 is ALWAYS the featured image (purpose: "featured"). No section_anchor.
  2. Scale inline image count to word count:
       < 500 words  → featured only (no inline)
       500–1000     → 1–2 inline
       1000+        → 2–4 inline
     Never exceed 5 total images.
  3. An inline image must support a specific argument in its section.
     It is NOT a decorative break and NOT added just because there is a heading.
  4. The "why" field is the reasoning — what the image makes clearer that prose cannot:
       "Supports the broken spring diagnosis section by showing what a failed torsion
        spring looks like to a homeowner who has never seen one."
  5. The "subject" field is the VISUAL description — what must literally be in the frame:
       BAD:  "garage door problem"
       GOOD: "worn garage door torsion spring with visible stress fractures and
              rust, photographed close-up in a residential garage interior"
  6. The generation_prompt must be photorealistic and complete: subject, setting,
     lighting, perspective, level of detail. This prompt goes directly to an AI
     image generator — write it as you would write a Midjourney or DALL-E prompt.
  7. mandatory: True for the featured image and any image that anchors a diagnostic
     or safety section. False for supplementary images that add value but are not
     essential to the article's argument.

══════════════════════════════════════════════════
OUTPUT
══════════════════════════════════════════════════

Return ONLY the structured reasoning plan.
No prose. No article text. No sentences that could appear in the article.
Every field is a fact, a reasoning element, or a decision framework — never a sentence
the writer could copy directly.\
"""


def _build_planner_schema() -> dict[str, Any]:
    """JSON schema for the structured planner tool output."""

    section_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "heading": {
                "type": "string",
                "description": (
                    "Section heading text — no # prefix. "
                    "Never use [City], [Service], or any bracket placeholder — "
                    "always write the actual city name and service name."
                ),
            },
            "reader_intent": {
                "type": "string",
                "description": "What the reader wants to understand or accomplish from this section",
            },
            "reader_misconception": {
                "type": "string",
                "description": "The wrong belief most readers hold about this section's topic",
            },
            "why_misconception_forms": {
                "type": "string",
                "description": "Why this misconception is natural and intuitive",
            },
            "technical_reality": {
                "type": "string",
                "description": "The accurate technical fact that corrects the misconception",
            },
            "failure_mechanism": {
                "type": "string",
                "description": "The physical sequence that results when someone acts on the misconception",
            },
            "professional_insight": {
                "type": "string",
                "description": "What a 15-year veteran technician knows that a layperson would not",
            },
            "local_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "City/regional factors that modify the generic advice for this section. "
                    "Must be embedded into technical reasoning, not stated as trivia."
                ),
            },
            "safety_considerations": {
                "type": "string",
                "description": "Physical risks in this section. Empty string if none.",
            },
            "diagnostic_tests": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific tests a homeowner can safely perform",
            },
            "decision_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete criteria for choosing between approaches",
            },
            "realistic_limitations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Honest constraints an expert would acknowledge",
            },
            "preventive_advice": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Steps that prevent or slow this problem",
            },
            "counter_intuitive_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Facts that contradict what most readers would expect",
            },
            "why_not_examples": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Format: 'Don't do X because Y (specific mechanical reason).'"
                    " Name the wrong approach, state the prohibition, explain the exact reason."
                ),
            },
            "specific_numbers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Quantitative facts: cycles, years, cost ranges, time intervals, dimensions. "
                    "Only include numbers you are confident are accurate for this trade."
                ),
            },
            "eeat_opportunities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Moments where domain expertise can be demonstrated through specific knowledge",
            },
            "section_hook": {
                "type": "string",
                "description": "The opening idea or angle that makes this section immediately engaging",
            },
            "section_closer": {
                "type": "string",
                "description": "A short punchy sentence that closes this section before the next",
            },
            "keyword_placement": {
                "type": "string",
                "description": "If the focus keyword fits naturally in this section, explain where and how",
            },
        },
        "required": [
            "heading",
            "reader_intent",
            "reader_misconception",
            "why_misconception_forms",
            "technical_reality",
            "failure_mechanism",
            "professional_insight",
        ],
    }

    faq_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Exact question as a user would type it — include city name and specific condition. "
                    "Should match real search-query phrasing."
                ),
            },
            "answer_core": {
                "type": "string",
                "description": "Key technical fact to convey — reasoning only, never prose",
            },
            "local_angle": {
                "type": "string",
                "description": "Regional specificity to include in the answer",
            },
        },
        "required": ["question", "answer_core"],
    }

    return {
        "type": "object",
        "properties": {
            "article_thesis": {
                "type": "string",
                "description": "The single core argument this article makes",
            },
            "hook_angle": {
                "type": "string",
                "description": (
                    "The reader assumption that the opening sentence will challenge or reframe. "
                    "This is the idea, not a sentence — the writer creates the prose."
                ),
            },
            "what_reader_gets_wrong": {
                "type": "string",
                "description": "The primary misconception the whole article corrects",
            },
            "local_context_foundation": {
                "type": "string",
                "description": "How the target city fundamentally shapes the advice in this article",
            },
            "regional_specifics": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concrete verifiable local facts: climate, neighborhoods, housing stock, "
                    "regulations, terrain. These are woven into technical arguments throughout."
                ),
            },
            "primary_counter_intuition": {
                "type": "string",
                "description": "The single most surprising correct fact about this topic",
            },
            "primary_prohibition": {
                "type": "string",
                "description": (
                    "The central prohibition of this topic. "
                    "Format: 'Never do X because Y (exact mechanical or physical reason).'"
                ),
            },
            "focus_keyword": {
                "type": "string",
                "description": (
                    "The primary SEO focus keyword for this article. "
                    "Must use the city and service from TARGET LOCATION and BUSINESS NICHE "
                    "in the task brief — NEVER invent or copy values from these instructions. "
                    "Format: '{resolved_service} {resolved_city}, {resolved_state}' "
                    "using the exact values from the task brief fields. "
                    "If TARGET LOCATION is absent, omit the city: service terms only."
                ),
            },
            "internal_link_keyword": {
                "type": "string",
                "description": "A secondary keyword to repeat 3–5 times as an internal linking target",
            },
            "external_authorities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Authoritative sources to cite and where in the article. "
                    "Format: 'CPSC for spring safety statistics' or 'DASMA for industry standards'."
                ),
            },
            "conclusion_angle": {
                "type": "string",
                "description": "How to close the article — should circle back to the hook or restate the correction",
            },
            "section_plans": {
                "type": "array",
                "items": section_schema,
                "description": "Reasoning plan for each major section, in document order. Plan 5–8 sections.",
            },
            "faq_plans": {
                "type": "array",
                "items": faq_schema,
                "description": (
                    "5 FAQ questions targeting actual search queries. "
                    "Include city name and specific condition in each question."
                ),
            },
            "image_plans": {
                "type": "array",
                "description": (
                    "Image plan for the article. "
                    "img_001 MUST be featured (purpose: 'featured'). "
                    "img_002+ are inline in section order. "
                    "Scale to word count: <500 → featured only; "
                    "500–1000 → 1–2 inline; 1000+ → 2–4 inline."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "string",
                            "description": "'img_001' (featured), 'img_002', 'img_003', etc.",
                        },
                        "purpose": {
                            "type": "string",
                            "enum": ["featured", "inline"],
                        },
                        "section_anchor": {
                            "type": "string",
                            "description": (
                                "Exact H2 heading text this image follows. "
                                "Omit for featured images."
                            ),
                        },
                        "why": {
                            "type": "string",
                            "description": (
                                "WHY this image exists — what technical claim, failure mode, "
                                "or concept it supports. Must reference a specific section."
                            ),
                        },
                        "subject": {
                            "type": "string",
                            "description": (
                                "WHAT the image must show: precise visual description. "
                                "BAD: 'garage door'. "
                                "GOOD: 'worn torsion spring with stress fractures, "
                                "residential garage interior, close-up'."
                            ),
                        },
                        "image_type": {
                            "type": "string",
                            "enum": [
                                "photograph", "process_photo", "product_photo",
                                "team_photo", "problem_photo", "before_after", "infographic",
                            ],
                        },
                        "generation_prompt": {
                            "type": "string",
                            "description": (
                                "Complete AI image generation prompt. Photorealistic. "
                                "Include subject, setting, lighting, and perspective. "
                                "Example: 'close-up of worn garage door torsion spring "
                                "with stress fractures above a two-car residential garage door, "
                                "sharp detail, professional photography, shallow depth of field'."
                            ),
                        },
                        "alt_text": {
                            "type": "string",
                            "description": (
                                "SEO alt text: include the focus keyword naturally, "
                                "describe what is visible, under 125 characters. "
                                "BAD: 'garage door'. "
                                "GOOD: 'technician replacing torsion spring above a two-car garage door, "
                                "close-up of worn coils, residential interior'. "
                                "Include the city from TARGET LOCATION only if it adds geographic context. "
                                "Never copy a city from these instructions."
                            ),
                        },
                        "caption": {
                            "type": "string",
                            "description": "Optional display caption. Empty string if not needed.",
                        },
                        "mandatory": {
                            "type": "boolean",
                            "description": (
                                "True for featured and section-critical images. "
                                "Publishing fails if a mandatory image is not resolved."
                            ),
                        },
                    },
                    "required": [
                        "image_id", "purpose", "why", "subject",
                        "image_type", "generation_prompt", "alt_text",
                    ],
                },
            },
        },
        "required": [
            "article_thesis",
            "hook_angle",
            "what_reader_gets_wrong",
            "local_context_foundation",
            "primary_counter_intuition",
            "primary_prohibition",
            "focus_keyword",
            "section_plans",
            "faq_plans",
            "image_plans",
        ],
    }


class ArticlePlannerService:
    """
    Technical reasoning planner that executes before article generation.

    Calls Claude with adaptive thinking to reason through the topic as a domain
    expert would: misconceptions, failure mechanisms, local adaptations, quantitative
    anchors, prohibitions, and E-E-A-T opportunities.

    Returns a structured ArticlePlan that the ArticleAgent uses as the authoritative
    source of domain knowledge for the article. The generator is then a prose
    transformer — it never invents reasoning beyond what the plan provides.
    """

    def __init__(self, claude_service: "ClaudeService") -> None:
        self._claude = claude_service

    def plan(self, request: "ArticleRequest") -> ArticlePlan | None:
        """
        Build a complete technical reasoning plan for an article.

        Returns None on failure so the caller can fall back to unplanned generation
        rather than aborting the pipeline.
        """
        loc_str = ""
        if request.location:
            loc = request.location
            parts = [p for p in [loc.city, loc.state, loc.country] if p]
            loc_str = ", ".join(parts)

        logger.info(
            "Planning article: topic=%r  keyword=%r  location=%s",
            request.topic,
            request.focus_keyword,
            loc_str or "unspecified",
        )

        from config import settings as _settings
        try:
            raw = self._claude.generate_structured(
                system=_PLANNER_SYSTEM,
                messages=[{"role": "user", "content": self._build_prompt(request, loc_str)}],
                tool_name="submit_article_plan",
                tool_description=(
                    "Submit the complete technical reasoning plan for this article. "
                    "This plan contains only structured expert reasoning — never prose "
                    "or article text. Every field is a fact, reasoning element, "
                    "or decision framework."
                ),
                input_schema=_build_planner_schema(),
                max_tokens=8000,
                thinking=True,
                model=_settings.planner_model,
                label="plan:article",
            )
        except Exception:
            logger.exception("Article planner failed — falling back to unplanned generation")
            return None

        plan = self._parse(raw, request)
        logger.info(
            "Plan complete: %d sections, %d FAQ questions",
            len(plan.section_plans),
            len(plan.faq_plans),
        )
        return plan

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(request: "ArticleRequest", loc_str: str) -> str:
        lines: list[str] = []

        # Business context — always present after BusinessContextResolver runs.
        # These are the ONLY sources the planner may use for city, state, service.
        if request.service:
            lines.append(f"BUSINESS NICHE: {request.service}")

        lines.append(f"ARTICLE TOPIC: {request.topic}")

        if request.focus_keyword:
            lines.append(f"FOCUS KEYWORD: {request.focus_keyword}")

        if loc_str:
            lines.append(f"TARGET LOCATION: {loc_str}")
            if request.location and request.location.neighborhood:
                lines.append(f"NEIGHBORHOOD FOCUS: {request.location.neighborhood}")

        if request.objective:
            lines.append(f"ARTICLE OBJECTIVE: {request.objective}")

        if request.target_audience:
            lines.append(f"TARGET AUDIENCE: {request.target_audience}")

        lines.append(f"APPROXIMATE WORD COUNT: {request.word_count}")

        lines += [
            "",
            "TASK:",
            "Build the complete technical reasoning plan for this article.",
            "Plan 5–8 major H2 sections that cover the topic comprehensively.",
            "For each section, work through the full expert reasoning sequence.",
            "Include 5 FAQ questions using real search-query phrasing with city name.",
            "",
            "REMEMBER:",
            "• Think like someone who has done this work — not read about it.",
            "• Every misconception must be one real homeowners commonly hold.",
            "• Every prohibition must include the exact mechanical or physical reason.",
            "• Local factors must change the technical argument, not just name the city.",
            "• Specific numbers must be accurate for the trade — omit if uncertain.",
            "• Realistic limitations are more credible than optimism.",
            "• Counter-intuitive facts are more valuable than obvious facts.",
        ]

        return "\n".join(lines)

    @staticmethod
    def _parse(raw: dict[str, Any], request: "ArticleRequest") -> ArticlePlan:
        def _str(val: Any, default: str = "") -> str:
            return str(val) if val is not None else default

        def _strlist(val: Any) -> list[str]:
            if not val:
                return []
            return [str(item) for item in val if item is not None]

        def _section(s: dict[str, Any]) -> SectionPlan:
            return SectionPlan(
                heading=_str(s.get("heading")),
                reader_intent=_str(s.get("reader_intent")),
                reader_misconception=_str(s.get("reader_misconception")),
                why_misconception_forms=_str(s.get("why_misconception_forms")),
                technical_reality=_str(s.get("technical_reality")),
                failure_mechanism=_str(s.get("failure_mechanism")),
                professional_insight=_str(s.get("professional_insight")),
                local_factors=_strlist(s.get("local_factors")),
                safety_considerations=_str(s.get("safety_considerations")),
                diagnostic_tests=_strlist(s.get("diagnostic_tests")),
                decision_criteria=_strlist(s.get("decision_criteria")),
                realistic_limitations=_strlist(s.get("realistic_limitations")),
                preventive_advice=_strlist(s.get("preventive_advice")),
                counter_intuitive_facts=_strlist(s.get("counter_intuitive_facts")),
                why_not_examples=_strlist(s.get("why_not_examples")),
                specific_numbers=_strlist(s.get("specific_numbers")),
                eeat_opportunities=_strlist(s.get("eeat_opportunities")),
                section_hook=_str(s.get("section_hook")),
                section_closer=_str(s.get("section_closer")),
                keyword_placement=_str(s.get("keyword_placement")),
            )

        def _faq(f: dict[str, Any]) -> FAQPlan:
            return FAQPlan(
                question=_str(f.get("question")),
                answer_core=_str(f.get("answer_core")),
                local_angle=_str(f.get("local_angle")),
            )

        def _image(img: dict[str, Any]) -> PlannedImage:
            return PlannedImage(
                image_id=_str(img.get("image_id", "img_001")),
                purpose=_str(img.get("purpose", "featured")),  # type: ignore[arg-type]
                section_anchor=img.get("section_anchor") or None,
                why=_str(img.get("why")),
                subject=_str(img.get("subject")),
                image_type=_str(img.get("image_type", "photograph")),
                generation_prompt=_str(img.get("generation_prompt")),
                alt_text=_str(img.get("alt_text")),
                caption=_str(img.get("caption")),
                mandatory=bool(img.get("mandatory", True)),
            )

        return ArticlePlan(
            article_thesis=_str(raw.get("article_thesis")),
            hook_angle=_str(raw.get("hook_angle")),
            what_reader_gets_wrong=_str(raw.get("what_reader_gets_wrong")),
            local_context_foundation=_str(raw.get("local_context_foundation")),
            regional_specifics=_strlist(raw.get("regional_specifics")),
            primary_counter_intuition=_str(raw.get("primary_counter_intuition")),
            primary_prohibition=_str(raw.get("primary_prohibition")),
            focus_keyword=_str(raw.get("focus_keyword"), request.focus_keyword or ""),
            internal_link_keyword=_str(raw.get("internal_link_keyword")),
            external_authorities=_strlist(raw.get("external_authorities")),
            conclusion_angle=_str(raw.get("conclusion_angle")),
            section_plans=[_section(s) for s in (raw.get("section_plans") or [])],
            faq_plans=[_faq(f) for f in (raw.get("faq_plans") or [])],
            image_plans=[_image(i) for i in (raw.get("image_plans") or [])],
        )

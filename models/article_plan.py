"""
ArticlePlan — structured technical reasoning produced before article generation.

The planner reasons like a 15-year trade technician: misconceptions, failure mechanisms,
local adaptations, quantitative anchors, prohibitions, counter-intuitions.
The generator transforms this structured reasoning into natural prose.

The planner never writes prose. The generator never invents reasoning.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannedImage(BaseModel):
    """
    A single image planned by the article planner.

    Specifies WHY the image exists (reasoning), WHAT to show (subject), and
    WHERE to place it (section_anchor). The ImageResolverAgent converts this
    into an ImageRequest and searches Drive before considering AI generation.
    """

    image_id: str = Field(
        description="'img_001' (featured), 'img_002', 'img_003' (inline), etc."
    )
    purpose: Literal["featured", "inline"] = Field(
        description=(
            "'featured' = WordPress featured_media (no markdown marker). "
            "'inline' = body placement with <!-- SEO_AGENT_IMAGE: id --> marker."
        )
    )
    section_anchor: str | None = Field(
        default=None,
        description=(
            "Exact H2 heading text this image follows. "
            "None for the featured image."
        ),
    )

    # ── Intent ────────────────────────────────────────────────────────────────
    why: str = Field(
        description=(
            "WHY this image exists: what technical claim, failure mode, or concept it "
            "supports. Must reference a specific section or argument in the article."
        )
    )
    subject: str = Field(
        description=(
            "WHAT the image must show: precise visual description in concrete terms. "
            "BAD: 'garage door'. "
            "GOOD: 'worn garage door torsion spring with visible stress fractures, "
            "residential garage interior'."
        )
    )

    # ── Type and generation ───────────────────────────────────────────────────
    image_type: str = Field(
        description=(
            "One of: photograph, process_photo, product_photo, team_photo, "
            "problem_photo, before_after, infographic."
        )
    )
    generation_prompt: str = Field(
        description=(
            "Complete AI image generation prompt. Photorealistic. "
            "Include subject, setting, lighting, and perspective."
        )
    )

    # ── SEO and display ───────────────────────────────────────────────────────
    alt_text: str = Field(
        description=(
            "SEO alt text: include the focus keyword naturally, "
            "describe what is visible, under 125 characters."
        )
    )
    caption: str = Field(
        default="",
        description="Optional display caption. Empty string if not needed.",
    )

    # ── Publication gate ──────────────────────────────────────────────────────
    mandatory: bool = Field(
        default=True,
        description=(
            "If True, publish() fails if this image is not resolved. "
            "Always True for the featured image. Set to False for supplementary inline images."
        ),
    )


class SectionPlan(BaseModel):
    """Structured expert reasoning for one article section."""

    # ── Structure ─────────────────────────────────────────────────────────────
    heading: str = Field(description="Section heading text — no # prefix")

    # ── Reader understanding ──────────────────────────────────────────────────
    reader_intent: str = Field(
        default="",
        description="What the reader wants to understand or accomplish from this section",
    )
    reader_misconception: str = Field(
        default="",
        description="The wrong belief most readers hold about this section's topic",
    )
    why_misconception_forms: str = Field(
        default="",
        description="Why this misconception is natural and intuitive",
    )

    # ── Technical core ────────────────────────────────────────────────────────
    technical_reality: str = Field(
        default="",
        description="The accurate technical fact that corrects the misconception",
    )
    failure_mechanism: str = Field(
        default="",
        description="The physical sequence that results when someone acts on the misconception",
    )
    professional_insight: str = Field(
        default="",
        description="What a 15-year veteran technician knows that a layperson would not",
    )

    # ── Local adaptation ──────────────────────────────────────────────────────
    local_factors: list[str] = Field(
        default_factory=list,
        description="City/regional factors that modify the generic advice for this section",
    )

    # ── Safety ───────────────────────────────────────────────────────────────
    safety_considerations: str = Field(
        default="",
        description="Physical risks in this section. Empty string if none.",
    )

    # ── Actionable knowledge ──────────────────────────────────────────────────
    diagnostic_tests: list[str] = Field(
        default_factory=list,
        description="Specific tests a homeowner can safely perform",
    )
    decision_criteria: list[str] = Field(
        default_factory=list,
        description="Concrete criteria for choosing between approaches",
    )
    realistic_limitations: list[str] = Field(
        default_factory=list,
        description="Honest constraints an expert would acknowledge",
    )
    preventive_advice: list[str] = Field(
        default_factory=list,
        description="Steps that prevent or slow this problem",
    )

    # ── Authenticity and E-E-A-T ──────────────────────────────────────────────
    counter_intuitive_facts: list[str] = Field(
        default_factory=list,
        description="Facts that contradict what most readers would expect",
    )
    why_not_examples: list[str] = Field(
        default_factory=list,
        description="Format: 'Don't do X because Y (specific mechanical reason)'",
    )
    specific_numbers: list[str] = Field(
        default_factory=list,
        description="Quantitative facts: cycles, years, cost ranges, time intervals",
    )
    eeat_opportunities: list[str] = Field(
        default_factory=list,
        description="Moments where domain expertise can be demonstrated through specific knowledge",
    )

    # ── Editorial ─────────────────────────────────────────────────────────────
    section_hook: str = Field(
        default="",
        description="The opening idea or angle that makes this section immediately engaging",
    )
    section_closer: str = Field(
        default="",
        description="A short punchy sentence that closes this section",
    )
    keyword_placement: str = Field(
        default="",
        description="If the focus keyword fits naturally here, explain where and how",
    )


class FAQPlan(BaseModel):
    """Planning for one FAQ question."""

    question: str = Field(
        description="Exact question as a user would type it — include city and specific condition",
    )
    answer_core: str = Field(
        description="Key technical fact to convey — reasoning only, not prose",
    )
    local_angle: str = Field(
        default="",
        description="Regional specificity to include in the answer",
    )


class ArticlePlan(BaseModel):
    """
    Complete technical reasoning plan produced by ArticlePlannerService.

    The ArticleAgent consumes this plan as the authoritative source of domain
    knowledge, local context, and authenticity signals for the article.

    Invariant: everything in the generated article that claims to be a fact,
    a local detail, or a technical insight must trace back to this plan.
    """

    # ── Article-level concept ─────────────────────────────────────────────────
    article_thesis: str = Field(
        description="The single core argument this article makes",
    )
    hook_angle: str = Field(
        description="The reader assumption that the opening sentence will challenge or reframe",
    )
    what_reader_gets_wrong: str = Field(
        description="The primary misconception the whole article corrects",
    )

    # ── Local context ─────────────────────────────────────────────────────────
    local_context_foundation: str = Field(
        description="How the target city fundamentally shapes the advice in this article",
    )
    regional_specifics: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete verifiable local facts: climate, neighborhoods, housing, "
            "regulations, terrain — to be woven into technical arguments"
        ),
    )

    # ── Primary E-E-A-T anchors ───────────────────────────────────────────────
    primary_counter_intuition: str = Field(
        description="The single most surprising correct fact about this topic",
    )
    primary_prohibition: str = Field(
        description="The central prohibition: 'Never do X because Y (exact mechanical reason)'",
    )

    # ── SEO ──────────────────────────────────────────────────────────────────
    focus_keyword: str = Field(
        default="",
        description="Primary SEO focus keyword",
    )
    internal_link_keyword: str = Field(
        default="",
        description="Secondary keyword to repeat 3–5 times as an internal linking target",
    )
    external_authorities: list[str] = Field(
        default_factory=list,
        description="Authoritative sources to cite and where in the article",
    )

    # ── Structure ─────────────────────────────────────────────────────────────
    section_plans: list[SectionPlan] = Field(
        default_factory=list,
        description="Reasoning plan for each major section in document order",
    )
    faq_plans: list[FAQPlan] = Field(
        default_factory=list,
        description="FAQ questions targeting actual search queries",
    )
    conclusion_angle: str = Field(
        default="",
        description="How to close — should circle back to the hook",
    )

    # ── Image plan ────────────────────────────────────────────────────────────
    image_plans: list[PlannedImage] = Field(
        default_factory=list,
        description=(
            "Image plan produced alongside the article plan. "
            "img_001 is always the featured image; img_002+ are inline in section order. "
            "Empty if planning was skipped or if word count is below the image threshold."
        ),
    )

    def to_generator_block(self) -> str:
        """
        Serialize the plan into a structured text block for the generator prompt.

        The generator reads this block and transforms it into natural prose.
        Every label makes the generator's job explicit: what to transform,
        what to embed, what must appear verbatim (specific numbers).
        """
        lines: list[str] = [
            "═" * 64,
            "TECHNICAL REASONING PLAN",
            "Transform every field into natural prose. Never copy this text directly.",
            "═" * 64,
            "",
            "ARTICLE CONCEPT",
            f"  Thesis:                  {self.article_thesis}",
            f"  Hook angle (sentence 1): {self.hook_angle}",
            "",
            "LOCAL CONTEXT",
            f"  Foundation: {self.local_context_foundation}",
        ]

        if self.regional_specifics:
            lines.append(
                "  Regional facts — embed into technical arguments, never as standalone trivia:"
            )
            for fact in self.regional_specifics:
                lines.append(f"    • {fact}")

        lines += [
            "",
            "PRIMARY E-E-A-T ANCHORS",
            f"  Counter-intuition: {self.primary_counter_intuition}",
            f"  Prohibition:       {self.primary_prohibition}",
        ]

        if self.internal_link_keyword:
            lines += [
                "",
                "INTERNAL LINKING",
                f"  Repeat 3–5× as internal link target: {self.internal_link_keyword}",
            ]

        if self.external_authorities:
            lines += ["", "AUTHORITY CITATIONS:"]
            for auth in self.external_authorities:
                lines.append(f"  • {auth}")

        if self.section_plans:
            lines += [
                "",
                "═" * 64,
                "SECTION REASONING (document order)",
                "═" * 64,
            ]

        for i, sec in enumerate(self.section_plans, 1):
            lines += [
                "",
                f"── SECTION {i}: ## {sec.heading} ──",
                "",
                f"  technical reality:     {sec.technical_reality}",
                f"  professional insight:  {sec.professional_insight}",
            ]

            if sec.local_factors:
                lines.append(
                    "  local factors (embed into technical reasoning, not as trivia):"
                )
                for f in sec.local_factors:
                    lines.append(f"    – {f}")

            if sec.safety_considerations:
                lines.append(f"  ⚠ safety: {sec.safety_considerations}")

            if sec.diagnostic_tests:
                lines.append("  diagnostic tests reader can perform:")
                for t in sec.diagnostic_tests:
                    lines.append(f"    – {t}")

            if sec.decision_criteria:
                lines.append("  decision criteria:")
                for c in sec.decision_criteria:
                    lines.append(f"    – {c}")

            if sec.realistic_limitations:
                lines.append("  realistic limitations (must be acknowledged honestly):")
                for lim in sec.realistic_limitations:
                    lines.append(f"    – {lim}")

            if sec.preventive_advice:
                lines.append("  preventive advice:")
                for p in sec.preventive_advice:
                    lines.append(f"    – {p}")

            if sec.counter_intuitive_facts:
                lines.append(
                    "  counter-intuitive facts (reveal naturally — never 'surprisingly...'):"
                )
                for f in sec.counter_intuitive_facts:
                    lines.append(f"    – {f}")

            if sec.why_not_examples:
                lines.append(
                    "  why-not examples (name → prohibit → exact mechanical reason):"
                )
                for w in sec.why_not_examples:
                    lines.append(f"    – {w}")

            if sec.specific_numbers:
                lines.append("  specific numbers (MUST appear in the prose):")
                for n in sec.specific_numbers:
                    lines.append(f"    – {n}")

            if sec.eeat_opportunities:
                lines.append(
                    "  E-E-A-T opportunities (express as prose expertise — NEVER as image"
                    " captions, photo descriptions, or 'as shown/pictured' references):"
                )
                for e in sec.eeat_opportunities:
                    lines.append(f"    – {e}")

            if sec.section_hook:
                lines.append(f"  section hook: {sec.section_hook}")

            if sec.section_closer:
                lines.append(f"  closer (1 punchy sentence): {sec.section_closer}")

            if sec.keyword_placement:
                lines.append(f"  keyword placement: {sec.keyword_placement}")

        if self.faq_plans:
            lines += [
                "",
                "═" * 64,
                "FAQ REASONING",
                "═" * 64,
            ]
            for i, faq in enumerate(self.faq_plans, 1):
                lines += [
                    "",
                    f"  FAQ {i}: {faq.question}",
                    f"    answer core: {faq.answer_core}",
                ]
                if faq.local_angle:
                    lines.append(f"    local angle: {faq.local_angle}")

        if self.conclusion_angle:
            lines += [
                "",
                "═" * 64,
                f"CONCLUSION ANGLE: {self.conclusion_angle}",
                "Write an editorial conclusion that circles back to this angle.",
                "Expert tone throughout — no generic CTAs, no 'contact us' language.",
                "One brief action step is acceptable only if it arises naturally from the editorial close.",
                "Never introduce a new city or neighborhood not mentioned earlier in the article.",
                "═" * 64,
            ]

        return "\n".join(lines)

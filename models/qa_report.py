from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RevisionAttempt:
    """
    Tracks whether one specific revision instruction was applied.

    Three outcomes:
      applied=True,  evaluable=True  — instruction was substantially implemented.
      applied=False, evaluable=True  — instruction was not implemented.
      applied=False, evaluable=False — insufficient information in the article text
                                       to assess this instruction (e.g. slug, meta
                                       description, keyword density). It is excluded
                                       from the compliance rate denominator.

    Populated after every revision cycle by the compliance checker.
    Stored on the iteration that *triggered* the revision (not the one that
    reviewed the result), so the report shows what was requested alongside
    whether it was actually done.
    """
    instruction: str
    priority: str = ""      # "High" | "Medium" | "Low"
    applied: bool = False
    evaluable: bool = True  # False → excluded from compliance rate
    location: str = ""      # "Introduction", "Section 2", "Throughout", etc.
    evidence: str = ""      # One sentence from the compliance checker

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "priority": self.priority,
            "applied": self.applied,
            "evaluable": self.evaluable,
            "location": self.location,
            "evidence": self.evidence,
        }


@dataclass
class DimensionDetail:
    """
    Structured explanation for one scored dimension.

    Populated by both reviewers. Empty fields mean the reviewer did not return
    that level of detail (e.g. the article passed with no notable weaknesses).
    """
    reasoning: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    priority: str = ""     # "High" | "Medium" | "Low" | ""

    def to_dict(self) -> dict:
        return {
            "reasoning": self.reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvements": self.improvements,
            "priority": self.priority,
        }


@dataclass
class ArticleReviewIteration:
    """Scores and feedback from one complete review cycle (both reviewers)."""

    iteration: int
    article_title: str = ""  # title at the point of this review (changes after revision)

    # Claude — SEO Editor
    seo_score: int = 0
    editorial_score: int = 0
    claude_approved: bool = False
    claude_feedback: str = ""
    claude_revision_instructions: str = ""

    # OpenAI — Human Authenticity Reviewer
    writing_score: int = 0
    authenticity_score: int = 0
    openai_approved: bool = False
    openai_feedback: str = ""
    openai_revision_instructions: str = ""

    # Per-dimension structured explanations
    seo_detail: DimensionDetail = field(default_factory=DimensionDetail)
    editorial_detail: DimensionDetail = field(default_factory=DimensionDetail)
    writing_detail: DimensionDetail = field(default_factory=DimensionDetail)
    authenticity_detail: DimensionDetail = field(default_factory=DimensionDetail)

    # Revision compliance — populated after this iteration triggers a revision cycle.
    # Empty when the iteration approved the article (no revision was run).
    revision_attempts: list[RevisionAttempt] = field(default_factory=list)

    # Timing
    elapsed_seconds: float = 0.0

    @property
    def approved(self) -> bool:
        return self.claude_approved and self.openai_approved

    @property
    def combined_score(self) -> float:
        return (
            self.seo_score + self.editorial_score
            + self.writing_score + self.authenticity_score
        ) / 4

    @property
    def rejection_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.claude_approved:
            reasons.append(
                f"Claude SEO {self.seo_score}/100 (min 90) | "
                f"Editorial {self.editorial_score}/100 (min 90)"
            )
        if not self.openai_approved:
            reasons.append(
                f"OpenAI Writing {self.writing_score}/100 (min 90) | "
                f"Authenticity {self.authenticity_score}/100 (min 90)"
            )
        return reasons

    def failed_dimensions(self) -> list[tuple[str, int, "DimensionDetail"]]:
        """
        Return (label, score, detail) for every dimension that did not pass 90.
        Used by terminal display and report saving to explain failures.
        """
        dims = []
        if self.seo_score < 90:
            dims.append(("SEO Quality", self.seo_score, self.seo_detail))
        if self.editorial_score < 90:
            dims.append(("Editorial Quality", self.editorial_score, self.editorial_detail))
        if self.writing_score < 90:
            dims.append(("Human Writing", self.writing_score, self.writing_detail))
        if self.authenticity_score < 90:
            dims.append(("Authenticity", self.authenticity_score, self.authenticity_detail))
        return dims

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "article_title": self.article_title,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "seo_score": self.seo_score,
            "editorial_score": self.editorial_score,
            "writing_score": self.writing_score,
            "authenticity_score": self.authenticity_score,
            "combined_score": round(self.combined_score, 1),
            "claude_approved": self.claude_approved,
            "openai_approved": self.openai_approved,
            "approved": self.approved,
            "claude_feedback": self.claude_feedback,
            "openai_feedback": self.openai_feedback,
            "claude_revision_instructions": self.claude_revision_instructions,
            "openai_revision_instructions": self.openai_revision_instructions,
            "seo_detail": self.seo_detail.to_dict(),
            "editorial_detail": self.editorial_detail.to_dict(),
            "writing_detail": self.writing_detail.to_dict(),
            "authenticity_detail": self.authenticity_detail.to_dict(),
            "revision_attempts": [a.to_dict() for a in self.revision_attempts],
            "revision_compliance_rate": self._compliance_rate(),
            "revision_not_evaluable_count": sum(
                1 for a in self.revision_attempts if not a.evaluable
            ),
        }

    def _compliance_rate(self) -> float | None:
        """
        Fraction of evaluable instructions that were applied.

        Not-evaluable instructions (slug, meta description, keyword density, etc.)
        are excluded from both numerator and denominator. Returns None when no
        revision was run, or when every instruction is not-evaluable.
        """
        if not self.revision_attempts:
            return None
        evaluable = [a for a in self.revision_attempts if a.evaluable]
        if not evaluable:
            return None
        applied = sum(1 for a in evaluable if a.applied)
        return round(applied / len(evaluable), 2)


@dataclass
class ImageQAResult:
    """Vision QA result for a single AI-generated image."""

    image_id: str
    source: str  # "variation" | "generated"

    claude_vision_score: int = 0
    claude_vision_approved: bool = False
    claude_vision_feedback: str = ""

    openai_vision_score: int = 0
    openai_vision_approved: bool = False
    openai_vision_feedback: str = ""

    @property
    def approved(self) -> bool:
        return self.claude_vision_approved and self.openai_vision_approved

    @property
    def combined_vision_score(self) -> float:
        return (self.claude_vision_score + self.openai_vision_score) / 2

    @property
    def rejection_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.claude_vision_approved:
            reasons.append(
                f"Claude Vision {self.claude_vision_score}/100 (min 90): "
                f"{self.claude_vision_feedback[:120]}"
            )
        if not self.openai_vision_approved:
            reasons.append(
                f"OpenAI Vision {self.openai_vision_score}/100 (min 90): "
                f"{self.openai_vision_feedback[:120]}"
            )
        return reasons

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "source": self.source,
            "claude_vision_score": self.claude_vision_score,
            "claude_vision_approved": self.claude_vision_approved,
            "claude_vision_feedback": self.claude_vision_feedback,
            "openai_vision_score": self.openai_vision_score,
            "openai_vision_approved": self.openai_vision_approved,
            "openai_vision_feedback": self.openai_vision_feedback,
            "combined_vision_score": round(self.combined_vision_score, 1),
            "approved": self.approved,
        }


@dataclass
class DualQAReport:
    """Complete dual-review QA report for a publish run."""

    # Article
    article_iterations: list[ArticleReviewIteration] = field(default_factory=list)
    article_passed: bool = False

    # Images — only AI images reviewed; Drive originals automatically approved
    image_results: list[ImageQAResult] = field(default_factory=list)
    images_passed: bool = True

    # Overall
    iterations_used: int = 0
    rejection_reasons: list[str] = field(default_factory=list)

    # ── Computed: Publication Readiness ──────────────────────────────────────
    publication_readiness_score: float = 0.0

    # ── Computed: Authenticity ────────────────────────────────────────────────
    article_authenticity: float = 0.0     # editorial + writing + authenticity (not SEO)
    image_authenticity: float | None = None  # avg vision score across AI images (None = no AI images)
    overall_authenticity: float = 0.0
    authenticity_label: str = ""
    authenticity_narrative: str = ""

    # ── Timing ────────────────────────────────────────────────────────────────
    qa_elapsed_seconds: float = 0.0

    # ── Costs (USD) ───────────────────────────────────────────────────────────
    claude_review_cost_usd: float = 0.0    # Claude article reviews
    openai_review_cost_usd: float = 0.0   # OpenAI article reviews
    revision_cost_usd: float = 0.0        # Claude article revisions
    vision_claude_cost_usd: float = 0.0   # Claude vision image reviews
    vision_openai_cost_usd: float = 0.0   # OpenAI vision image reviews

    # ── Authenticity rescue (post-cycle, one-shot) ────────────────────────────
    authenticity_revision_attempted: bool = False
    authenticity_revision_passed: bool = False
    authenticity_revision_cost_usd: float = 0.0   # Claude rewrite cost
    authenticity_revision_openai_cost_usd: float = 0.0  # final OpenAI re-review cost

    # ── Image counts (for reporting) ──────────────────────────────────────────
    drive_originals_count: int = 0
    preservation_edits_count: int = 0

    @property
    def total_qa_cost_usd(self) -> float:
        return round(
            self.claude_review_cost_usd
            + self.openai_review_cost_usd
            + self.revision_cost_usd
            + self.vision_claude_cost_usd
            + self.vision_openai_cost_usd
            + self.authenticity_revision_cost_usd
            + self.authenticity_revision_openai_cost_usd,
            6,
        )

    @property
    def avg_cycle_seconds(self) -> float:
        times = [it.elapsed_seconds for it in self.article_iterations if it.elapsed_seconds > 0]
        return sum(times) / len(times) if times else 0.0

    @property
    def passed(self) -> bool:
        return self.article_passed and self.images_passed

    @property
    def article_review_passed(self) -> bool:
        """Article text quality only — independent of image vision QA."""
        return self.article_passed

    @property
    def final_article_review(self) -> ArticleReviewIteration | None:
        return self.article_iterations[-1] if self.article_iterations else None

    # ── Score computation ─────────────────────────────────────────────────────

    def compute_final_scores(
        self,
        approved_images: list,
    ) -> None:
        """
        Compute all derived scores after the review loop completes.

        approved_images: list of (ImageRequest, ImageAsset) that passed QA.
        Call this once, after run() finishes, before saving the report.
        """
        from models.image_asset import ImageSource

        final = self.final_article_review
        if not final:
            return

        # ── Publication Readiness Score (25 / 25 / 25 / 25 weighted) ─────────
        base = (
            final.seo_score * 0.25
            + final.editorial_score * 0.25
            + final.writing_score * 0.25
            + final.authenticity_score * 0.25
        )

        # Image authenticity bonus — up to +5 points when edited photos passed vision review.
        edited_approved_count = sum(
            1 for _, a in approved_images
            if a.source == ImageSource.EDITED
        )
        if edited_approved_count > 0 and self.image_results:
            passing = [r for r in self.image_results if r.approved]
            if passing:
                avg_vision = sum(r.combined_vision_score for r in passing) / len(passing)
                bonus = max(0.0, (avg_vision - 90.0) / 2.0)  # 0–5 points
                base = min(100.0, base + bonus)

        self.publication_readiness_score = round(base, 1)

        # ── Authenticity ──────────────────────────────────────────────────────
        # Article authenticity = editorial quality + writing naturalness + AI resistance
        # SEO is excluded — it measures optimization, not authenticity.
        self.article_authenticity = round(
            (final.editorial_score + final.writing_score + final.authenticity_score) / 3, 1
        )

        # Image authenticity = average combined vision score across ALL reviewed images
        if self.image_results:
            avg = sum(r.combined_vision_score for r in self.image_results) / len(self.image_results)
            self.image_authenticity = round(avg, 1)

        # Overall = article (70%) + images (30%) if AI images were reviewed
        if self.image_authenticity is not None:
            self.overall_authenticity = round(
                self.article_authenticity * 0.70 + self.image_authenticity * 0.30, 1
            )
        else:
            self.overall_authenticity = self.article_authenticity

        # Label and narrative
        if self.overall_authenticity >= 95:
            self.authenticity_label = "Excellent"
            self.authenticity_narrative = (
                "This article appears to have been created by a real marketing team "
                "using authentic company photography."
            )
        elif self.overall_authenticity >= 90:
            self.authenticity_label = "Very Good"
            self.authenticity_narrative = (
                "The article reads as human-authored and images appear genuine. "
                "Only a trained AI-content reviewer would notice minor patterns."
            )
        elif self.overall_authenticity >= 85:
            self.authenticity_label = "Good"
            self.authenticity_narrative = (
                "The content is mostly convincing but retains some AI patterns "
                "that an observant reader might notice."
            )
        elif self.overall_authenticity >= 75:
            self.authenticity_label = "Fair"
            self.authenticity_narrative = (
                "AI patterns are detectable in places. Further revision is recommended "
                "before treating this as indistinguishable from human-authored content."
            )
        else:
            self.authenticity_label = "Needs Improvement"
            self.authenticity_narrative = (
                "The content shows clear signs of AI generation. "
                "Significant revision is required before publication."
            )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "article_passed": self.article_passed,
            "images_passed": self.images_passed,
            "iterations_used": self.iterations_used,
            "rejection_reasons": self.rejection_reasons,
            # Publication readiness
            "publication_readiness_score": self.publication_readiness_score,
            # Authenticity
            "article_authenticity": self.article_authenticity,
            "image_authenticity": self.image_authenticity,
            "overall_authenticity": self.overall_authenticity,
            "authenticity_label": self.authenticity_label,
            "authenticity_narrative": self.authenticity_narrative,
            # Images
            "drive_originals_count": self.drive_originals_count,
            "preservation_edits_count": self.preservation_edits_count,
            # Timing
            "qa_elapsed_seconds": round(self.qa_elapsed_seconds, 2),
            "avg_cycle_seconds": round(self.avg_cycle_seconds, 2),
            # Costs
            "claude_review_cost_usd": self.claude_review_cost_usd,
            "openai_review_cost_usd": self.openai_review_cost_usd,
            "revision_cost_usd": self.revision_cost_usd,
            "vision_claude_cost_usd": self.vision_claude_cost_usd,
            "vision_openai_cost_usd": self.vision_openai_cost_usd,
            "authenticity_revision_cost_usd": self.authenticity_revision_cost_usd,
            "authenticity_revision_openai_cost_usd": self.authenticity_revision_openai_cost_usd,
            "total_qa_cost_usd": self.total_qa_cost_usd,
            # Authenticity rescue
            "authenticity_revision_attempted": self.authenticity_revision_attempted,
            "authenticity_revision_passed": self.authenticity_revision_passed,
            # Per-cycle detail
            "article_reviews": [it.to_dict() for it in self.article_iterations],
            "image_reviews": [r.to_dict() for r in self.image_results],
        }

from __future__ import annotations

import base64
import io
import logging
import time as _time
from typing import TYPE_CHECKING, Any, Callable

from agents._marker_utils import insert_marker_at_section
from config import settings
from models.article import Article, ArticleRequest
from models.image_asset import ImageSource
from models.image_request import ImagePlacementPlan, ImagePurpose
from models.qa_report import ArticleReviewIteration, DimensionDetail, DualQAReport, ImageQAResult, RevisionAttempt

if TYPE_CHECKING:
    from services.claude_service import ClaudeService
    from services.openai_review_service import OpenAIReviewService

import re as _re

logger = logging.getLogger(__name__)

_VISION_MAX_BYTES = 8 * 1024 * 1024  # 8 MB hard ceiling for Claude Vision uploads


def _restore_displaced_markers(pre_revision: str, post_revision: str) -> str:
    """
    Re-insert image markers that Claude moved to the end of the document during revision.

    Despite _REVISION_SYSTEM "ABSOLUTE RULES", structural revision instructions
    (reorder sections, integrate keywords into headings) can displace markers to the
    last 20% of the document. This function detects that shift and restores each
    marker to its pre-revision section by matching H2 headings.
    """
    marker_pattern = _re.compile(r'<!-- SEO_AGENT_IMAGE: \S+ -->')
    pre_markers = marker_pattern.findall(pre_revision)
    if not pre_markers:
        return post_revision

    pre_lines = pre_revision.splitlines()
    result = post_revision

    for marker in pre_markers:
        if marker not in result:
            continue  # dropped marker; handled by defensive re-insertion in main.py

        post_lines = result.splitlines()
        pre_total = max(len(pre_lines), 1)
        post_total = max(len(post_lines), 1)

        pre_frac = next(
            (i / pre_total for i, ln in enumerate(pre_lines) if marker in ln), 1.0
        )
        post_frac = next(
            (i / post_total for i, ln in enumerate(post_lines) if marker in ln), 1.0
        )

        # Only act when marker moved significantly toward the end
        if post_frac < 0.80 or pre_frac >= 0.70:
            continue

        # Find the first H2 heading that follows the marker in the pre-revision document.
        # Markers sit immediately BEFORE the section they introduce, so the following
        # heading is the correct anchor for restoring the marker's position.
        marker_line = next(
            (i for i, ln in enumerate(pre_lines) if marker in ln), None
        )
        ref_heading: str | None = None
        if marker_line is not None:
            for i in range(marker_line + 1, len(pre_lines)):
                m = _re.match(r'^##\s+(.+)', pre_lines[i])
                if m:
                    ref_heading = m.group(1).strip()
                    break

        if not ref_heading:
            continue

        # Find best-matching H2 in the post-revision document
        ref_words = {w for w in _re.split(r'\W+', ref_heading.lower()) if len(w) > 3}
        best_heading: str | None = None
        best_overlap = 0
        for h in _re.findall(r'^##\s+.+', result, _re.MULTILINE):
            h_words = {w for w in _re.split(r'\W+', h.lower()) if len(w) > 3}
            overlap = len(ref_words & h_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_heading = h

        if best_heading and best_overlap > 0:
            # Remove marker from its displaced end position; insert before the matched heading
            result = _re.sub(r'\n*' + _re.escape(marker) + r'\n*', '\n\n', result).rstrip('\n') + '\n'
            result = result.replace(best_heading, f'{marker}\n{best_heading}', 1)
            logger.warning(
                "Marker %s was displaced to end during QA revision — "
                "restored before heading %r (matched %r, %d word(s) in common).",
                marker, best_heading, ref_heading, best_overlap,
            )

    return result


class VisionImageTooLargeError(Exception):
    """Raised when an image cannot be reduced below 8 MB for Claude Vision."""


def _bytes_to_mime(data: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


def prepare_image_for_claude(image_bytes: bytes, image_id: str) -> tuple[bytes, str]:
    """
    The single gateway for all image bytes sent to Claude Vision.

    Every image must pass through this function before generate_structured()
    is called. No image larger than 8 MB may reach the Anthropic API.

    Strategy (resize only as last resort):
      1. Already ≤ 8 MB → return as-is (no print).
      2. No transparency → JPEG quality 95, 92, 85 (no resize).
      3. Has alpha → keep PNG, scale 75% → 60% → 50% → 40% → 30%.
      4. No-alpha still over limit after JPEG → same scale steps.

    Prints [TRACE] only when optimization is actually performed.
    Raises VisionImageTooLargeError if Pillow is not installed and image > 8 MB.

    Returns (optimized_bytes, mime_type).
    """
    if len(image_bytes) <= _VISION_MAX_BYTES:
        return image_bytes, _bytes_to_mime(image_bytes)

    original_mb = len(image_bytes) / 1024 / 1024

    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        raise VisionImageTooLargeError(
            f"Image {image_id} is {original_mb:.1f} MB (limit 8 MB) and Pillow is not "
            "installed. Run: pip install Pillow"
        )

    img = Image.open(io.BytesIO(image_bytes))
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and img.info.get("transparency") is not None
    )

    # ── Try JPEG conversion at decreasing quality (no resize) ────────────────
    if not has_alpha:
        for quality in (95, 92, 85):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = buf.getvalue()
            if len(candidate) <= _VISION_MAX_BYTES:
                optimized_mb = len(candidate) / 1024 / 1024
                print(f"[TRACE]")
                print(f"  Image:     {image_id}")
                print(f"  Original:  {original_mb:.1f} MB")
                print(f"  Optimized: {optimized_mb:.1f} MB")
                print(f"  Upload:    {optimized_mb:.1f} MB")
                print(f"  Claude:    Accepted")
                return candidate, "image/jpeg"

    # ── Resize progressively ─────────────────────────────────────────────────
    for scale in (0.75, 0.60, 0.50, 0.40, 0.30):
        w = max(1, int(img.width * scale))
        h = max(1, int(img.height * scale))
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        if has_alpha:
            resized.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        else:
            resized.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
            mime = "image/jpeg"
        candidate = buf.getvalue()
        if len(candidate) <= _VISION_MAX_BYTES:
            optimized_mb = len(candidate) / 1024 / 1024
            print(f"[TRACE]")
            print(f"  Image:     {image_id}")
            print(f"  Original:  {original_mb:.1f} MB")
            print(f"  Optimized: {optimized_mb:.1f} MB")
            print(f"  Upload:    {optimized_mb:.1f} MB")
            print(f"  Claude:    Accepted")
            return candidate, mime

    raise VisionImageTooLargeError(
        f"Image {image_id} ({original_mb:.1f} MB) could not be reduced below 8 MB "
        "after JPEG conversion and progressive resizing to 30%."
    )


# ── Claude system prompts ─────────────────────────────────────────────────────

_CLAUDE_SEO_SYSTEM = """\
You are a Senior SEO Editor with 15 years of experience optimizing content for local service
businesses. You are one of two independent reviewers in a production quality gate.

YOUR RESPONSIBILITY: Review articles for SEO excellence and editorial readiness before publication.
Be honest and demanding — a score of 90+ must be genuinely earned.

SEO QUALITY (seo_score 0–100)
Evaluate critically:

Search Intent & Helpfulness
• Does the article fully satisfy what someone searching this keyword actually wants?
• Would Google's Helpful Content System classify this as written for people, not search engines?
• Is there genuine, useful information — or filler padded to reach word count?

E-E-A-T Signals
• Experience: does the article reflect practical domain expertise — accurate process knowledge,
  correct trade terminology, realistic cost ranges, material choices, and service timelines?
  Personal stories and customer anecdotes are NOT part of the editorial format and are NOT
  required to satisfy this criterion. Technical depth and local grounding demonstrate experience.
• Expertise: accurate technical detail, correct terminology, realistic specifics?
• Authoritativeness: cites reputable sources, demonstrates domain knowledge?
• Trustworthiness: specific verifiable claims, no vague promises, honest tone?

Technical SEO
• Focus keyword: present in H1, introduction (first 100 words), at least one H2, conclusion?
• Keyword density: 1–2% — not stuffed, not invisible?
• Heading hierarchy: logical H1 → H2 → H3, no skipped levels?
• Meta description: 120–160 chars, includes keyword, compelling?
• SEO title: 50–60 chars, includes keyword, enticing?
• Internal links: at least one to a relevant page?
  MANDATORY — check "INTERNAL LINKS CONFIGURED" in the article header before scoring:
  → If INTERNAL LINKS CONFIGURED: No — EXCLUDE this criterion entirely from your score.
    Award full credit. Do NOT deduct points. Do NOT list it as a weakness or improvement.
    Treat it as N/A. Scoring it would penalize a configuration choice, not an article flaw.
  → If INTERNAL LINKS CONFIGURED: Yes — evaluate normally; penalize if no internal links present.
• External links: 1–2 authoritative, non-competing sources?
• Slug: short, lowercase, keyword-focused?

Local SEO
• Geographic signals specific and convincing (neighborhoods, landmarks, local context)?
• Not just "[city] + [service]" dropped in generically?

Content Structure
• FAQ section with questions real users ask?
  SPECIFICATION: 3–4 FAQ questions is the CORRECT format for this content type.
  Do NOT deduct points for having 3–4 questions. Do NOT request more questions.
  Only penalize if: the FAQ section is entirely absent, OR the questions are
  irrelevant/generic and not tied to real user intent for this topic.
• CTA: clear, specific, relevant to the service?
• Tables or lists where they genuinely help (not filler)?

EDITORIAL QUALITY (editorial_score 0–100)
Evaluate critically:

Readability
• Short paragraphs (2–4 sentences) for web scanning?
• Varied sentence length and structure?
• No walls of text?

Structure
• Introduction hooks immediately — relevant, interesting first sentence?
• Body sections flow logically — each H2 earns its place?
• Conclusion earns its position — summarizes value, ends with CTA?

Completeness
• Covers the topic comprehensively without unnecessary padding?
• Answers the questions a real reader would have?
• Alt text in image placeholders is descriptive and accurate?

SCORING GUIDE:
95–100 = Publication-ready masterpiece
90–94  = Strong, ready to publish with minor tweaks at most
80–89  = Good but needs targeted improvements
70–79  = Significant issues affecting quality
< 70   = Requires substantial revision

DECISION RULE:
Both seo_score AND editorial_score must be >= 90 for approval.
Score each independently — a great SEO article with weak editing still fails.\
"""

_CLAUDE_VISION_SYSTEM = """\
You are a Quality Control Director reviewing edited company photographs for a local service business blog.

You will receive:
1. The ORIGINAL company photograph (labeled "Original photograph:").
2. The EDITED version with a minimal preservation edit applied (labeled "Edited photograph:").

PRIMARY QUESTION — answer this FIRST before scoring anything else:
"Does this still look like the SAME original company photograph?"

If the edited image no longer resembles the original — different technician, different house,
different truck, different garage door, different composition — REJECT IT immediately.
The edit FAILED. Score ≤ 40 and set approved = false.

IDENTITY PRESERVATION (highest priority)
• Are the same technician, uniform, face, hands, and equipment visible in both?
• Is the same house, garage door, driveway, and neighborhood visible in both?
• Is the same truck (if present) visible in both?
• Does the overall composition remain the same?
• Do approximately 90–100% of the original pixels appear unchanged?

EDIT QUALITY (secondary, only if identity is preserved)
• Was the edit minimal and surgical — not a full scene replacement?
• Is the edit cleanly applied without obvious artifacts, seams, or warping?
• Does the edit actually improve the photo for its intended use?

SCORING GUIDE:
95–100 = Same photograph — edit is invisible or nearly so; cleanly applied
90–94  = Same photograph with a clearly visible but clean and appropriate edit
80–89  = Same photograph but edit has minor artifacts or looks slightly unnatural
70–79  = Questionable — core elements mostly preserved but some deviations noticed
< 70   = Identity compromised — REJECT and use original Drive photo instead

DECISION RULE:
approved = true ONLY if vision_score >= 90 AND the image is still recognizably the same photograph.
If in doubt, reject — the original Drive photo is always the safe fallback.\
"""

_REVISION_SYSTEM = """\
You are a Senior Content Editor performing a full prose rewrite of a draft article.
Your output must read as though it was written by a different person from scratch —
not a polished version of the original.

STRATEGY: Preserve the skeleton. Replace the prose.

The skeleton (never change unless reviewers explicitly requested it):
  • Topic, search intent, and focus keyword
  • All H1, H2, H3 headings — their text, order, and hierarchy
  • All tables — structure and content
  • All Markdown links [anchor text](URL)
  • All image placeholders <!-- SEO_AGENT_IMAGE: ... -->
  • All factual claims, technical specifications, and verifiable statistics

The prose (rewrite from scratch in every section):
  • Introduction — completely new opening, different angle
  • Conclusion — completely new closing, no formulaic summary
  • Body paragraphs — fresh sentences expressing the same facts differently
  • Transitions — specific and contextual, not formulaic
  • Sentence openings — varied; no two consecutive sentences begin the same way

WORDING RULE:
Reuse as little phrasing from the previous draft as possible.
Technical terms (product names, trade terms, city names, service categories) are not
"phrasing" and may be used freely. Everything else — how sentences are constructed,
how paragraphs open, how ideas connect — should be freshly written.

WORD COUNT:
Target: 800 words — acceptable 700–900 — absolute maximum 950.
If the current article exceeds 900 words, trim the weakest sentences and padding
to bring it within range. Never cut factual claims, technical specifics, or keyword placements.

ABSOLUTE RULES — NEVER VIOLATE:
1. Preserve ALL image placeholder comments EXACTLY as written:
   <!-- SEO_AGENT_IMAGE: img_001 --> — do not change, move, or remove them
2. Preserve ALL Markdown links — both internal and external:
   [anchor text](URL) — keep every link, only improve anchor text if flagged
3. Maintain the focus keyword and overall article topic
4. The article language does not change on revision
5. If a conflict exists between SEO optimization and natural human writing, choose natural writing

REWRITE STANDARDS:
• A "complete rewrite" of a section means the output is unrecognizable from the original.
  Minor rewording does not count.
• Sentence variety means no two consecutive sentences in any paragraph start with the same word.
• Removing formulaic transitions means they are entirely absent — not reduced.
  BANNED: Furthermore, Moreover, Additionally, In conclusion, It's worth noting,
          First and foremost, At the end of the day, When it comes to, Needless to say

LOCAL CONTEXT POLICY:
The target city is the primary geographic context. Use its widely known characteristics
throughout — not just in passing mentions. Local expertise is shown through accurate
regional knowledge, not invented stories.

USE CONFIDENTLY (broadly factual regional knowledge — no hedging required):
  • Climate and seasonal conditions (frost depth, humidity, precipitation, heat)
  • Typical housing styles, ages, and construction materials common to the area
  • Named neighborhoods, districts, or suburbs
  • Common infrastructure characteristics (age of housing stock, soil type, freeze-thaw)
  • Regional weather patterns, terrain, or environmental conditions
  • Common service problems associated with that climate or region
  • State or municipal regulations that are broadly established
  • Local building practices that are widely known

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

OUTPUT FORMAT:
Return the COMPLETE rewritten article in Markdown format.
Start with the H1 title: # Title Here
Do not include any preamble, commentary, or explanation before or after the article.\
"""

_COMPLIANCE_SYSTEM = """\
You are a revision compliance auditor. You receive three inputs:
1. The ORIGINAL article (before revision).
2. The REVISED article (after revision).
3. A numbered list of REQUESTED IMPROVEMENTS.

For each numbered improvement, return one of three verdicts:

  "applied"        — the improvement was SUBSTANTIALLY implemented in the article text.
  "not_applied"    — the improvement was not implemented, or only superficially so.
  "not_evaluable"  — you cannot assess this instruction from the article text alone
                     because the required artifact is not present in the Markdown body.

WHEN TO USE "not_evaluable" — use it when checking the instruction would require:
  • The slug / URL (not present in Markdown body)
  • The meta description (not present in Markdown body)
  • Keyword density calculation (requires external word-count tooling)
  • SEO plugin settings, schema markup, or WordPress metadata
  • Any artifact that exists outside the article body text

"not_evaluable" is NOT an excuse for uncertainty about body-text changes. If you can
read both articles and compare them, use "applied" or "not_applied".

Strict standards for body-text verdicts:
• "Rewrite completely" = section must be unrecognizable from the original. Minor rewording = "not_applied".
• "Rewrite 30–40% of body paragraphs" = scan every body paragraph in the revised article and compare it
  to the corresponding original paragraph. A paragraph is "rewritten from scratch" when it opens
  differently AND uses visibly different sentence construction throughout — same facts, fresh wording.
  A paragraph is "lightly edited" when more than half its sentences are recognizably carried over
  from the original (same opening, same phrasing, minor word swaps). Count how many paragraphs fall
  into each category. "applied" if clearly 3 or more body paragraphs (or ≥30% of total, whichever
  is larger) are rewritten from scratch. "not_applied" only if the prose throughout reads
  as a light edit — most paragraphs structurally unchanged from the original.
• "Remove all formulaic transitions" = must be entirely absent. One instance = "not_applied".
• "Vary sentence openings" = no two consecutive sentences in any paragraph start the same way.
• Reviewer bullets = "applied" only if the specific issue named is visibly addressed.

For each result:
  - verdict: "applied" | "not_applied" | "not_evaluable"
  - location: section or area (e.g., "Introduction", "Throughout", "Not in article body").
  - evidence: one concrete sentence — what changed, what is still wrong, or why not evaluable.\
"""


# ── Main agent ────────────────────────────────────────────────────────────────

class DualQAFailedError(Exception):
    """Raised when an article fails dual QA after the maximum number of revision cycles."""

    def __init__(self, report: DualQAReport, message: str = "") -> None:
        self.report = report
        super().__init__(
            message or f"Article failed dual QA after {report.iterations_used} iteration(s)."
        )


class DualQAAgent:
    """
    Production quality gate: two independent reviewers must approve before publication.

    Reviewer #1 — Claude (SEO Editor):
      Evaluates SEO quality, E-E-A-T, editorial excellence, local credibility,
      structure, linking, meta fields, readability.

    Reviewer #2 — OpenAI (Human Authenticity Reviewer):
      Evaluates writing naturalness, sentence rhythm, authenticity, AI detection,
      storytelling, and whether the article reads like expert human writing.

    Article review runs in a loop (max max_cycles). On each rejection, Claude
    applies targeted revisions and the cycle repeats. If both reviewers approve,
    the loop exits. If max_cycles is exhausted, DualQAFailedError is raised.

    Image review is one-shot (no retry):
      — Drive originals: automatically approved, no vision review.
      — AI variations/generated: reviewed by Claude Vision + OpenAI Vision.
        Images that fail are excluded from the final publish; the article still publishes.

    Philosophy: human authenticity > SEO perfection.
                real company photography > any AI image.
    """

    def __init__(
        self,
        claude: ClaudeService,
        openai_reviewer: OpenAIReviewService | None = None,
        *,
        min_seo: int = 90,
        min_editorial: int = 90,
        min_writing: int = 90,
        min_authenticity: int = 90,
        min_vision_claude: int = 90,
        min_vision_openai: int = 90,
        max_cycles: int = 3,
        enable_rescue: bool = True,
    ) -> None:
        self._claude = claude
        self._openai = openai_reviewer
        self._min_seo = min_seo
        self._min_editorial = min_editorial
        self._min_writing = min_writing
        self._min_authenticity = min_authenticity
        self._min_vision_claude = min_vision_claude
        self._min_vision_openai = min_vision_openai
        self._max_cycles = max_cycles
        self._enable_rescue = enable_rescue

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        article: Article,
        resolved_images: list,
        *,
        image_plan: ImagePlacementPlan | None = None,
    ) -> tuple[Article, list, DualQAReport]:
        """
        Run the full dual QA pipeline.

        Phase 1 — Article review loop:
          Both reviewers evaluate the article. On failure, Claude applies targeted
          revisions and the cycle repeats (up to max_cycles). DualQAFailedError is
          raised if the article still fails after max_cycles.

          If image_plan is supplied, marker integrity is verified and any markers
          dropped during revision are re-inserted before the method returns.

        Phase 2 — Image vision review (one shot, no retry):
          Drive originals (ImageSource.DRIVE) are automatically approved.
          Edited photos (ImageSource.EDITED) are reviewed by both Claude Vision and
          OpenAI Vision. The original Drive photo (asset.original_data) is shown
          alongside the edited version so reviewers can evaluate identity preservation.
          Primary question: "Does this still look like the SAME original photograph?"
          Images that fail are excluded; the original Drive photo is used instead.

        Returns:
          (approved_article, approved_images, report)

        Raises:
          DualQAFailedError: article failed after max_cycles.
        """
        report = DualQAReport()
        current = article
        qa_start = _time.perf_counter()

        # Budget tracking helpers (reads Claude budget service snapshots)
        budget_svc = getattr(self._claude, 'budget', None)

        def _snap() -> dict | None:
            return budget_svc.status() if budget_svc else None

        def _claude_delta(before: dict | None, after: dict | None) -> float:
            if not (before and after):
                return 0.0
            return round(after['claude']['usd'] - before['claude']['usd'], 6)

        # ── Phase 1: Article review loop ──────────────────────────────────────
        for cycle in range(1, self._max_cycles + 1):
            cycle_start = _time.perf_counter()
            logger.info("Dual QA article review — cycle %d/%d", cycle, self._max_cycles)

            # Claude review (budget-tracked separately from revision)
            snap0 = _snap()
            claude_result = self._claude_review_article(current)
            snap1 = _snap()
            report.claude_review_cost_usd += _claude_delta(snap0, snap1)

            # OpenAI review
            if self._openai is not None:
                _oa_text_before = self._openai.text_cost_usd
                openai_result = self._openai_review_article(current)
                if budget_svc:
                    _oa_delta = round(self._openai.text_cost_usd - _oa_text_before, 6)
                    if _oa_delta > 0:
                        budget_svc.record_openai_text(_oa_delta)
                openai_approved = (
                    openai_result["writing_score"] >= self._min_writing
                    and openai_result["authenticity_score"] >= self._min_authenticity
                )
            else:
                logger.warning(
                    "OpenAI reviewer not configured — writing/authenticity scores unavailable. "
                    "OpenAI QA gate bypassed. Scores reported as 0 (not measured)."
                )
                openai_result = {
                    "writing_score": 0, "authenticity_score": 0, "approved": False,
                    "writing_feedback": "OpenAI reviewer not configured — check skipped.",
                    "authenticity_feedback": "OpenAI reviewer not configured — check skipped.",
                    "issues": [], "revision_instructions": "",
                }
                openai_approved = True  # explicit bypass, not score-derived

            # Assemble iteration
            claude_approved = (
                claude_result.get("seo_score", 0) >= self._min_seo
                and claude_result.get("editorial_score", 0) >= self._min_editorial
            )

            combined_openai_feedback = "\n".join(filter(None, [
                openai_result.get("writing_feedback", ""),
                openai_result.get("authenticity_feedback", ""),
            ]))

            iteration = ArticleReviewIteration(
                iteration=cycle,
                article_title=current.title,
                seo_score=claude_result.get("seo_score", 0),
                editorial_score=claude_result.get("editorial_score", 0),
                claude_approved=claude_approved,
                claude_feedback=claude_result.get("feedback", ""),
                claude_revision_instructions=claude_result.get("revision_instructions", ""),
                writing_score=openai_result["writing_score"],
                authenticity_score=openai_result["authenticity_score"],
                openai_approved=openai_approved,
                openai_feedback=combined_openai_feedback,
                openai_revision_instructions=openai_result.get("revision_instructions", ""),
                seo_detail=DimensionDetail(
                    reasoning=claude_result.get("seo_reasoning", ""),
                    strengths=claude_result.get("seo_strengths", []),
                    weaknesses=claude_result.get("seo_weaknesses", []),
                    improvements=claude_result.get("seo_improvements", []),
                    priority=claude_result.get("seo_priority", ""),
                ),
                editorial_detail=DimensionDetail(
                    reasoning=claude_result.get("editorial_reasoning", ""),
                    strengths=claude_result.get("editorial_strengths", []),
                    weaknesses=claude_result.get("editorial_weaknesses", []),
                    improvements=claude_result.get("editorial_improvements", []),
                    priority=claude_result.get("editorial_priority", ""),
                ),
                writing_detail=DimensionDetail(
                    reasoning=openai_result.get("writing_reasoning", ""),
                    strengths=openai_result.get("writing_strengths", []),
                    weaknesses=openai_result.get("writing_weaknesses", []),
                    improvements=openai_result.get("writing_improvements", []),
                    priority=openai_result.get("writing_priority", ""),
                ),
                authenticity_detail=DimensionDetail(
                    reasoning=openai_result.get("authenticity_reasoning", ""),
                    strengths=openai_result.get("authenticity_strengths", []),
                    weaknesses=openai_result.get("authenticity_weaknesses", []),
                    improvements=openai_result.get("authenticity_improvements", []),
                    priority=openai_result.get("authenticity_priority", ""),
                ),
            )

            if iteration.approved:
                iteration.elapsed_seconds = _time.perf_counter() - cycle_start
                report.article_iterations.append(iteration)
                report.iterations_used = cycle
                report.article_passed = True
                logger.info(
                    "Dual QA PASSED on cycle %d — SEO=%d, Editorial=%d, Writing=%d, Authenticity=%d",
                    cycle,
                    iteration.seo_score, iteration.editorial_score,
                    iteration.writing_score, iteration.authenticity_score,
                )
                break

            logger.info(
                "Dual QA cycle %d FAILED — Claude: SEO=%d Editorial=%d (%s) | "
                "OpenAI: Writing=%d Authenticity=%d (%s)",
                cycle,
                iteration.seo_score, iteration.editorial_score,
                "PASS" if claude_approved else "FAIL",
                iteration.writing_score, iteration.authenticity_score,
                "PASS" if openai_approved else "FAIL",
            )

            if cycle < self._max_cycles:
                logger.info("Applying targeted revisions before cycle %d...", cycle + 1)
                snap2 = _snap()
                current = self._revise(current, iteration)
                snap3 = _snap()
                report.revision_cost_usd += _claude_delta(snap2, snap3)

            iteration.elapsed_seconds = _time.perf_counter() - cycle_start
            report.article_iterations.append(iteration)
            report.iterations_used = cycle

            if cycle == self._max_cycles and not iteration.approved:
                for reason in iteration.rejection_reasons:
                    report.rejection_reasons.append(f"Final cycle: {reason}")

                # Authenticity rescue: if SEO + Editorial passed but only
                # Human Writing / Authenticity failed, try one dedicated rewrite.
                if self._enable_rescue and iteration.claude_approved and not iteration.openai_approved:
                    rescue_article = self._try_authenticity_rescue(
                        current, iteration, openai_result, report, _snap, _claude_delta
                    )
                    if rescue_article is not None:
                        current = rescue_article
                        break  # exit loop; article passed; proceed to Phase 2

                report.qa_elapsed_seconds = _time.perf_counter() - qa_start
                raise DualQAFailedError(
                    report,
                    f"Article failed dual QA after {self._max_cycles} revision cycle(s).",
                )

        # ── Marker integrity: restore any markers dropped during revision ────────
        if image_plan is not None:
            restored_md = self._restore_dropped_markers(current.content_markdown, image_plan)
            if restored_md != current.content_markdown:
                _rm_words = len(restored_md.split())
                current = current.model_copy(update={
                    "content_markdown": restored_md,
                    "word_count": _rm_words,
                    "reading_time_minutes": max(1, _rm_words // 200),
                })

        # ── SEO regen: one call on the final passing article ─────────────────
        # Only run when the article was actually revised (title could differ from original).
        if report.iterations_used > 1 or (
            report.article_iterations
            and not report.article_iterations[0].approved
        ):
            try:
                from agents.article_agent import ArticleAgent as _AA
                _seo = _AA(self._claude)._generate_seo(current.request, current.content_markdown)
                current = current.model_copy(update={"seo": _seo})
                logger.info("SEO metadata regenerated for final passing article.")
            except Exception as exc:
                logger.warning("SEO regen after QA failed (non-blocking): %s", exc)

        # ── Phase 2: Image vision review ──────────────────────────────────────
        snap4 = _snap()
        approved_images, image_results = self._review_images(resolved_images)
        snap5 = _snap()
        report.vision_claude_cost_usd = _claude_delta(snap4, snap5)

        if self._openai is not None:
            report.openai_review_cost_usd = round(self._openai.text_cost_usd, 6)
            report.vision_openai_cost_usd = round(self._openai.vision_cost_usd, 6)

        report.image_results = image_results
        failed = [r for r in image_results if not r.approved]
        report.images_passed = not failed
        for r in failed:
            for reason in r.rejection_reasons:
                report.rejection_reasons.append(f"Image {r.image_id}: {reason}")

        # Image counts for the authenticity report
        report.drive_originals_count = sum(
            1 for _, a in resolved_images if a.source == ImageSource.DRIVE
        )
        report.preservation_edits_count = sum(
            1 for _, a in resolved_images if a.source == ImageSource.EDITED
        )

        # Finalize
        report.qa_elapsed_seconds = _time.perf_counter() - qa_start
        report.compute_final_scores(approved_images=approved_images)

        return current, approved_images, report

    # ── Article review ────────────────────────────────────────────────────────

    def _claude_review_article(self, article: Article) -> dict:
        return self._claude.generate_structured(
            system=_CLAUDE_SEO_SYSTEM,
            messages=[{
                "role": "user",
                "content": self._format_for_review(article),
            }],
            tool_name="submit_seo_editorial_review",
            tool_description=(
                "Submit the SEO and editorial review scores and full structured explanation "
                "for this article."
            ),
            model=settings.qa_model,
            label="qa:claude-review",
            input_schema={
                "type": "object",
                "properties": {
                    # ── SEO dimension ─────────────────────────────────────────
                    "seo_score": {
                        "type": "integer", "minimum": 0, "maximum": 100,
                        "description": "SEO quality score 0–100.",
                    },
                    "seo_reasoning": {
                        "type": "string",
                        "description": "1–3 sentences explaining why this SEO score was given.",
                    },
                    "seo_strengths": {
                        "type": "array", "items": {"type": "string"},
                        "description": "What the article does well on SEO (2–4 bullets).",
                    },
                    "seo_weaknesses": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Specific SEO problems that lowered the score (2–5 bullets).",
                    },
                    "seo_improvements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Concrete, actionable SEO fixes (2–5 bullets).",
                    },
                    "seo_priority": {
                        "type": "string", "enum": ["High", "Medium", "Low"],
                        "description": "How urgently SEO issues must be resolved before publication.",
                    },
                    # ── Editorial dimension ───────────────────────────────────
                    "editorial_score": {
                        "type": "integer", "minimum": 0, "maximum": 100,
                        "description": "Editorial quality score 0–100.",
                    },
                    "editorial_reasoning": {
                        "type": "string",
                        "description": "1–3 sentences explaining why this editorial score was given.",
                    },
                    "editorial_strengths": {
                        "type": "array", "items": {"type": "string"},
                        "description": "What the article does well editorially (2–4 bullets).",
                    },
                    "editorial_weaknesses": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Specific editorial problems that lowered the score (2–5 bullets).",
                    },
                    "editorial_improvements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Concrete, actionable editorial fixes (2–5 bullets).",
                    },
                    "editorial_priority": {
                        "type": "string", "enum": ["High", "Medium", "Low"],
                        "description": "How urgently editorial issues must be resolved.",
                    },
                    # ── Combined fields ───────────────────────────────────────
                    "feedback": {
                        "type": "string",
                        "description": "Detailed combined feedback narrative (SEO + editorial).",
                    },
                    "revision_instructions": {
                        "type": "string",
                        "description": "Specific, actionable revision instructions for the writer.",
                    },
                },
                "required": [
                    "seo_score", "seo_reasoning", "seo_weaknesses", "seo_improvements",
                    "seo_priority",
                    "editorial_score", "editorial_reasoning", "editorial_weaknesses",
                    "editorial_improvements", "editorial_priority",
                    "feedback", "revision_instructions",
                ],
            },
            max_tokens=3000,
            thinking=False,
        )

    def _openai_review_article(self, article: Article) -> dict:
        seo_context = (
            f"Focus keyword: {article.seo.focus_keyword}\n"
            f"Topic: {article.title}\n"
            f"Meta description: {article.seo.meta_description}"
        )
        return self._openai.review_article(  # type: ignore[union-attr]
            self._format_for_review(article), seo_context
        )

    # ── Revision ──────────────────────────────────────────────────────────────

    def _revise(self, article: Article, iteration: ArticleReviewIteration) -> Article:
        """
        Apply priority-aware revisions and record compliance results in the iteration.

        High-priority dimensions trigger mandatory deep-rewrite directives in the prompt.
        After revision, a compliance checker verifies each requested instruction against
        the actual changes and stores the results in iteration.revision_attempts.
        """
        # Canonical instruction list — used for both the prompt and compliance check
        instructions = self._collect_all_instructions(iteration)

        prompt = self._build_revision_prompt(article, iteration, instructions)

        revised_content = self._claude.generate(
            system=_REVISION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8096,
            thinking=False,  # revision is targeted editing, not deep reasoning
            model=settings.qa_model,
            label="qa:revision",
        )

        from agents.article_agent import ArticleAgent

        # Rebuild article preserving identity and tenant context.
        title = ArticleAgent._extract_title(revised_content) or article.title
        body = ArticleAgent._strip_h1(revised_content)

        # Re-anchor any markers that Claude displaced to the end during structural revision
        body = _restore_displaced_markers(article.content_markdown, body)

        _rev_words = len(body.split())
        revised = article.model_copy(update={
            "title": title,
            "content_markdown": body,
            "word_count": _rev_words,
            "reading_time_minutes": max(1, _rev_words // 200),
        })

        # Regenerate SEO when the current cycle's SEO score was below threshold.
        # This ensures the next QA cycle evaluates a fresh seo_title rather than
        # re-scoring the same stale metadata that caused the failure.
        if iteration.seo_score < self._min_seo:
            try:
                _seo = ArticleAgent(service=self._claude)._generate_seo(
                    article.request, body
                )
                revised = revised.model_copy(update={"seo": _seo})
                logger.info(
                    "SEO metadata regenerated after revision (cycle SEO score was %d < %d).",
                    iteration.seo_score, self._min_seo,
                )
            except Exception as exc:
                logger.warning("SEO regen after revision failed (non-blocking): %s", exc)

        # Compliance check — verify each instruction was actually applied.
        # Disabled by default (12k-22k tokens per revision, reporting only).
        if settings.qa_compliance_check and instructions:
            try:
                attempts = self._check_revision_compliance(
                    original=article.content_markdown,
                    revised=body,
                    instructions=instructions,
                )
                iteration.revision_attempts = attempts
                evaluable = [a for a in attempts if a.evaluable]
                applied = sum(1 for a in evaluable if a.applied)
                not_evaluable = len(attempts) - len(evaluable)
                logger.info(
                    "Revision compliance: %d/%d evaluable instructions applied (%.0f%%)%s",
                    applied, len(evaluable),
                    100 * applied / len(evaluable) if evaluable else 0,
                    f"; {not_evaluable} not evaluable (excluded)" if not_evaluable else "",
                )
            except Exception as exc:
                logger.warning("Revision compliance check failed (non-blocking): %s", exc)

        return revised

    def _try_authenticity_rescue(
        self,
        current: Article,
        final_iteration: "ArticleReviewIteration",
        openai_result: dict,
        report: "DualQAReport",
        _snap: "Any",
        _claude_delta: "Any",
    ) -> "Article | None":
        """
        One-shot authenticity revision after all QA cycles are exhausted.

        Conditions for activation (checked by caller):
          - Claude approved (SEO + Editorial both passed)
          - OpenAI failed (Human Writing or Authenticity below threshold)

        Process:
          1. AuthenticityRevisionService rewrites the article prose
          2. One final OpenAI review on the rewritten article
          3. Return revised article on pass, None on fail (caller raises DualQAFailedError)

        This method runs exactly once. It never loops.
        """
        from services.authenticity_revision_service import AuthenticityRevisionService

        logger.info(
            "Authenticity rescue triggered — SEO=%d Editorial=%d passed; "
            "Writing=%d Authenticity=%d failed. Running one-shot rewrite.",
            final_iteration.seo_score, final_iteration.editorial_score,
            final_iteration.writing_score, final_iteration.authenticity_score,
        )

        report.authenticity_revision_attempted = True

        # ── Step 1: Rewrite for human authenticity ────────────────────────────
        snap_before = _snap()
        try:
            svc = AuthenticityRevisionService(self._claude)
            revised = svc.revise(
                current,
                writing_feedback=openai_result.get("writing_feedback", ""),
                authenticity_feedback=openai_result.get("authenticity_feedback", ""),
                revision_instructions=openai_result.get("revision_instructions", ""),
                issues=openai_result.get("issues", []),
            )
        except Exception as exc:
            logger.error("AuthenticityRevisionService failed: %s", exc)
            return None
        snap_after = _snap()
        report.authenticity_revision_cost_usd += _claude_delta(snap_before, snap_after)

        # ── Step 2: Final OpenAI re-review on rewritten article ───────────────
        openai_before_cost = getattr(self._openai, 'text_cost_usd', 0.0)
        if self._openai is not None:
            try:
                final_openai = self._openai_review_article(revised)
            except Exception as exc:
                logger.error("Final OpenAI re-review failed: %s", exc)
                return None
        else:
            final_openai = {
                "writing_score": 100, "authenticity_score": 100, "approved": True,
                "writing_feedback": "OpenAI reviewer not configured.",
                "authenticity_feedback": "OpenAI reviewer not configured.",
                "issues": [], "revision_instructions": "",
            }
        openai_after_cost = getattr(self._openai, 'text_cost_usd', 0.0)
        _rescue_oa_delta = round(openai_after_cost - openai_before_cost, 6)
        report.authenticity_revision_openai_cost_usd += _rescue_oa_delta
        if budget_svc and _rescue_oa_delta > 0:
            budget_svc.record_openai_text(_rescue_oa_delta)

        # ── Step 3: Verdict ───────────────────────────────────────────────────
        rescue_writing = final_openai["writing_score"]
        rescue_auth = final_openai["authenticity_score"]
        rescue_passed = (
            rescue_writing >= self._min_writing
            and rescue_auth >= self._min_authenticity
        )

        rescue_iter = ArticleReviewIteration(
            iteration=len(report.article_iterations) + 1,
            article_title=revised.title,
            seo_score=final_iteration.seo_score,
            editorial_score=final_iteration.editorial_score,
            claude_approved=True,
            claude_feedback="[Authenticity rescue — Claude review not re-run]",
            claude_revision_instructions="",
            writing_score=rescue_writing,
            authenticity_score=rescue_auth,
            openai_approved=rescue_passed,
            openai_feedback="\n".join(filter(None, [
                final_openai.get("writing_feedback", ""),
                final_openai.get("authenticity_feedback", ""),
            ])),
            openai_revision_instructions=final_openai.get("revision_instructions", ""),
        )
        report.article_iterations.append(rescue_iter)

        if rescue_passed:
            report.authenticity_revision_passed = True
            report.article_passed = True
            logger.info(
                "Authenticity rescue PASSED — Writing=%d Authenticity=%d",
                rescue_writing, rescue_auth,
            )
            return revised

        report.rejection_reasons.append(
            f"Authenticity rescue failed — "
            f"Writing={rescue_writing}/100 (min {self._min_writing}) | "
            f"Authenticity={rescue_auth}/100 (min {self._min_authenticity})"
        )
        logger.info(
            "Authenticity rescue FAILED — Writing=%d Authenticity=%d",
            rescue_writing, rescue_auth,
        )
        return None

    def _restore_dropped_markers(
        self,
        markdown: str,
        plan: ImagePlacementPlan,
    ) -> str:
        """
        Re-insert any inline markers that were completely omitted from the markdown
        during QA revision cycles.

        _restore_displaced_markers() handles the case where a marker moved toward the
        end of the document; this handles the rarer case where Claude omitted a marker
        entirely. Both recovery mechanisms are owned here so that DualQAAgent is the
        single owner of all marker integrity.
        """
        for req in plan.requests:
            if req.purpose == ImagePurpose.INLINE and req.placement_marker not in markdown:
                logger.warning(
                    "Marker %s missing after QA revision — re-inserting at section '%s'.",
                    req.id, req.section_title or "(unknown)",
                )
                markdown = insert_marker_at_section(
                    markdown, req.placement_marker, req.section_title
                )
        return markdown

    def _build_revision_prompt(
        self,
        article: Article,
        iteration: ArticleReviewIteration,
        instructions: list[tuple[str, str]],
    ) -> str:
        parts: list[str] = [
            "═" * 64,
            "DEEP REWRITE TASK",
            "═" * 64,
            "",
            "You are rewriting this article from scratch — not editing it.",
            "The output must read as though a different writer produced it.",
            "",
            "PRESERVE (do not change unless a reviewer instruction explicitly requires it):",
            "  • Topic, search intent, and focus keyword",
            "  • Every heading — text, order, and hierarchy (H1, H2, H3)",
            "  • All tables — structure and content",
            "  • All Markdown links [anchor text](URL) — every single one",
            "  • All image placeholders <!-- SEO_AGENT_IMAGE: ... --> — unchanged",
            "  • All factual claims and technical specifications",
            "  • Focus keyword placement (first paragraph, H2 headings, conclusion)",
            "",
            "REWRITE (from scratch in every section):",
            "  • Introduction — entirely new opening, different angle from the original",
            "  • Conclusion — entirely new closing, no formulaic summary",
            "  • All body paragraphs — express the same facts with fresh sentences",
            "  • All transitions — specific and contextual, none formulaic",
            "",
            "WORDING RULE:",
            "  Reuse as little phrasing from the previous draft as possible.",
            "  Technical terms (product names, trade terms, city names, service names)",
            "  are not 'phrasing' and may be used freely.",
            "  Everything else — how sentences are built, how paragraphs open,",
            "  how ideas connect — must be freshly written.",
            "",
            "SENTENCE RULE:",
            "  No two consecutive sentences in any paragraph may begin with the same word.",
            "  Vary sentence length throughout.",
            "",
            "BANNED TRANSITIONS (must not appear anywhere in the output):",
            "  Furthermore, Moreover, Additionally, In conclusion, It's worth noting,",
            "  It's important to note, Needless to say, First and foremost,",
            "  When it comes to, At the end of the day, In today's world.",
            "",
            "Apply every reviewer instruction listed below before returning the article.",
            "═" * 64,
            "",
            "CURRENT ARTICLE (rewrite — do not copy):",
            "---",
            f"# {article.title}",
            "",
            article.content_markdown,
            "---",
            "",
        ]

        # ── Helper ───────────────────────────────────────────────────────────────
        def _bullets(label: str, items: list[str]) -> list[str]:
            if not items:
                return []
            return [f"{label}:"] + [f"  - {item}" for item in items] + [""]

        def _location_label() -> str:
            loc = getattr(getattr(article, "request", None), "location", None)
            if loc:
                parts_loc = [loc.city]
                if loc.state:
                    parts_loc.append(loc.state)
                return ", ".join(parts_loc)
            return "the local area"

        location = _location_label()

        # ── Claude reviewer block ─────────────────────────────────────────────
        if not iteration.claude_approved:
            parts += [
                "═" * 64,
                "REVIEWER #1 — SEO EDITOR (Claude)",
                f"  SEO Score:       {iteration.seo_score}/100  (required ≥ {self._min_seo})",
                f"  Editorial Score: {iteration.editorial_score}/100  (required ≥ {self._min_editorial})",
                "",
            ]

            # SEO — mandatory rewrite block if High priority
            if iteration.seo_score < self._min_seo:
                if iteration.seo_detail.priority == "High":
                    parts += [
                        "⚠  SEO — HIGH PRIORITY — MANDATORY SECTION REWRITES",
                        "━" * 50,
                        "Do not apply light edits. Rewrite the affected sections completely.",
                        "Required actions:",
                        "  1. Rewrite every H2 section that is missing the focus keyword.",
                        "  2. Rewrite the introduction to include the keyword in the first 100 words.",
                        "  3. Replace any generic claims with specific, verifiable statements.",
                        "",
                    ]
                parts += _bullets("SEO Issues", iteration.seo_detail.weaknesses)
                parts += _bullets("SEO Required Fixes", iteration.seo_detail.improvements)

            # Editorial — mandatory rewrite block if High priority
            if iteration.editorial_score < self._min_editorial:
                if iteration.editorial_detail.priority == "High":
                    parts += [
                        "⚠  EDITORIAL — HIGH PRIORITY — MANDATORY SECTION REWRITES",
                        "━" * 50,
                        "Do not apply light edits. Identify the weakest sections and rewrite them from scratch.",
                        "",
                    ]
                parts += _bullets("Editorial Issues", iteration.editorial_detail.weaknesses)
                parts += _bullets("Editorial Required Fixes", iteration.editorial_detail.improvements)

            parts += [
                f"Combined Feedback: {iteration.claude_feedback}",
                f"Revision Instructions: {iteration.claude_revision_instructions}",
                "",
            ]

        # ── OpenAI reviewer block ─────────────────────────────────────────────
        if not iteration.openai_approved:
            parts += [
                "═" * 64,
                "REVIEWER #2 — HUMAN AUTHENTICITY (OpenAI)",
                f"  Writing Score:       {iteration.writing_score}/100  (required ≥ {self._min_writing})",
                f"  Authenticity Score:  {iteration.authenticity_score}/100  (required ≥ {self._min_authenticity})",
                "",
            ]

            # Authenticity — the most damaging plateau cause; gets the deepest mandatory block
            if (iteration.authenticity_score < self._min_authenticity
                    and iteration.authenticity_detail.priority == "High"):
                parts += [
                    "⚠  AUTHENTICITY — HIGH PRIORITY — MANDATORY DEEP REWRITE",
                    "━" * 50,
                    "This article reads as AI-generated. Light edits will not fix this.",
                    "You MUST apply all five of the following structural changes:",
                    "",
                    "1. REWRITE THE INTRODUCTION COMPLETELY",
                    "   Delete the current first 2–3 paragraphs and replace them from scratch.",
                    f"   Open with something concrete and specific to {location} — a real local detail,",
                    "   a common homeowner situation, or a climate/housing-specific observation.",
                    "   Do NOT start with 'In today's world', 'When it comes to', or any variant.",
                    "",
                    "2. REWRITE THE CONCLUSION COMPLETELY",
                    "   Delete the current closing paragraphs and replace them from scratch.",
                    "   Close with something locally relevant and action-oriented.",
                    "   Do NOT start with 'In conclusion' or 'To summarize'.",
                    "",
                    "3. REWRITE AT LEAST 30–40% OF BODY PARAGRAPHS",
                    "   Count the body paragraphs. Select the most formulaic ones.",
                    "   Delete and replace them — not minor edits, complete replacements.",
                    "   This is verified by comparing the before and after word-by-word.",
                    "",
                    "4. SENTENCE VARIETY — apply throughout the ENTIRE article:",
                    "   • No two consecutive sentences in any paragraph may begin with the same word.",
                    "   • Mix short sentences (5–10 words) with longer ones (20–30 words).",
                    "   • Include at least one single-sentence paragraph in the article for emphasis.",
                    "",
                    "5. REMOVE ALL FORMULAIC TRANSITIONS — zero exceptions:",
                    "   BANNED: Furthermore, Moreover, Additionally, In conclusion,",
                    "           It's worth noting, It's important to note, First and foremost,",
                    "           At the end of the day, When it comes to, Needless to say",
                    "   Replace each with a context-specific phrase or restructure the sentence.",
                    "",
                    "FACTUAL LOCAL CONTEXT FOR " + location.upper() + ":",
                    "When adding local detail, use only real and verifiable information:",
                    "  ALLOWED:  neighborhood names, local climate, seasonal weather, housing styles,",
                    "            building eras, typical homeowner situations, regional regulations",
                    "  BANNED:   fake customers, fabricated testimonials, invented experiences,",
                    "            made-up statistics, hypothetical events presented as real",
                    "",
                ]
                parts += _bullets("Authenticity Issues to Fix", iteration.authenticity_detail.weaknesses)

            elif iteration.authenticity_score < self._min_authenticity:
                # Medium / Low authenticity
                parts += _bullets("Authenticity Issues", iteration.authenticity_detail.weaknesses)
                parts += _bullets("Authenticity Required Fixes", iteration.authenticity_detail.improvements)

            # Writing — mandatory deep block if High priority
            if (iteration.writing_score < self._min_writing
                    and iteration.writing_detail.priority == "High"):
                parts += [
                    "⚠  HUMAN WRITING — HIGH PRIORITY — MANDATORY REWRITE",
                    "━" * 50,
                    "Writing quality failed. You must apply the following:",
                    "  • Rewrite every paragraph where every sentence follows the same pattern.",
                    "  • After writing each paragraph, verify sentence openings are varied.",
                    "  • Mix paragraph lengths: some should be 1–2 sentences, some 4–5.",
                    "",
                ]
                parts += _bullets("Writing Issues", iteration.writing_detail.weaknesses)
                parts += _bullets("Writing Required Fixes", iteration.writing_detail.improvements)

            elif iteration.writing_score < self._min_writing:
                parts += _bullets("Writing Issues", iteration.writing_detail.weaknesses)
                parts += _bullets("Writing Required Fixes", iteration.writing_detail.improvements)

            parts += [
                f"Combined Feedback: {iteration.openai_feedback}",
                f"Revision Instructions: {iteration.openai_revision_instructions}",
                "",
            ]

        # ── Compliance note ───────────────────────────────────────────────────
        parts += [
            "─" * 64,
            "COMPLIANCE NOTE:",
            "After you return the revised article, each instruction above will be verified",
            "against the original by comparing the two versions. Instructions marked as",
            "HIGH PRIORITY are checked with strict criteria — partial changes do not count.",
            "",
        ]

        return "\n".join(parts)

    def _collect_all_instructions(
        self, iteration: ArticleReviewIteration
    ) -> list[tuple[str, str]]:
        """
        Build the canonical list of all revision instructions with their priority.

        Returns list of (instruction_text, priority) where priority is "High" | "Medium" | "Low".
        The same list is fed into both the revision prompt (to guide Claude) and the
        compliance checker (to verify what was actually done).

        Mandatory structural instructions are prepended for High-priority dimensions so
        the compliance checker can verify them even if they are not in the reviewer's
        improvement bullets.
        """
        items: list[tuple[str, str]] = []

        # ── Mandatory structural changes for High authenticity ─────────────────
        if (iteration.authenticity_score < self._min_authenticity
                and iteration.authenticity_detail.priority == "High"):
            items += [
                ("Rewrite the introduction completely (first 2–3 paragraphs replaced from scratch)", "High"),
                ("Rewrite the conclusion completely (closing paragraphs replaced from scratch)", "High"),
                ("Rewrite at least 30–40% of body paragraphs from scratch", "High"),
                ("Ensure no two consecutive sentences in any paragraph begin with the same word", "High"),
                ("Remove all instances of formulaic transitions (Furthermore, Moreover, Additionally, etc.)", "High"),
            ]

        # ── Mandatory structural changes for High writing ─────────────────────
        if (iteration.writing_score < self._min_writing
                and iteration.writing_detail.priority == "High"):
            items += [
                ("Vary sentence rhythm: mix short (5–10 words) and long (20–30 words) sentences throughout", "High"),
                ("Rewrite every paragraph where all sentences follow the same structural pattern", "High"),
            ]

        # ── Mandatory section rewrites for High SEO ───────────────────────────
        if (iteration.seo_score < self._min_seo
                and iteration.seo_detail.priority == "High"):
            items += [
                ("Rewrite every H2 section missing the focus keyword", "High"),
                ("Rewrite the introduction to include the focus keyword in the first 100 words", "High"),
            ]

        # ── Reviewer-supplied improvement bullets ─────────────────────────────
        if iteration.seo_score < self._min_seo:
            p = iteration.seo_detail.priority or "Medium"
            for imp in iteration.seo_detail.improvements:
                items.append((imp, p))

        if iteration.editorial_score < self._min_editorial:
            p = iteration.editorial_detail.priority or "Medium"
            for imp in iteration.editorial_detail.improvements:
                items.append((imp, p))

        if iteration.writing_score < self._min_writing:
            p = iteration.writing_detail.priority or "Medium"
            for imp in iteration.writing_detail.improvements:
                items.append((imp, p))

        if iteration.authenticity_score < self._min_authenticity:
            p = iteration.authenticity_detail.priority or "Medium"
            for imp in iteration.authenticity_detail.improvements:
                items.append((imp, p))

        # Deduplicate while preserving order (highest priority wins on conflict)
        seen: dict[str, str] = {}
        for text, priority in items:
            if text not in seen:
                seen[text] = priority
        return list(seen.items())

    def _check_revision_compliance(
        self,
        original: str,
        revised: str,
        instructions: list[tuple[str, str]],
    ) -> list[RevisionAttempt]:
        """
        Compare original and revised article and verify each instruction was applied.

        Uses the LLM with thinking=False for a fast, cheap structured evaluation.
        Returns a RevisionAttempt for each instruction.
        """
        if not instructions:
            return []

        numbered = "\n".join(
            f"{i + 1}. [{priority}] {text}"
            for i, (text, priority) in enumerate(instructions)
        )

        user_content = (
            "ORIGINAL ARTICLE (before revision):\n"
            "```\n" + original[:15000] + "\n```\n\n"
            "REVISED ARTICLE (after revision):\n"
            "```\n" + revised[:15000] + "\n```\n\n"
            "REQUESTED IMPROVEMENTS:\n"
            + numbered
            + "\n\nFor each numbered improvement, return a compliance result."
        )

        try:
            result = self._claude.generate_structured(
                system=_COMPLIANCE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
                tool_name="submit_compliance_check",
                tool_description=(
                    "Submit compliance results for each requested revision improvement."
                ),
                model=settings.qa_model,
                label="qa:compliance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "compliance_results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "instruction_number": {
                                        "type": "integer",
                                        "description": "1-based index matching the numbered list.",
                                    },
                                    "verdict": {
                                        "type": "string",
                                        "enum": ["applied", "not_applied", "not_evaluable"],
                                        "description": (
                                            "'applied' = substantially done; "
                                            "'not_applied' = missing or superficial; "
                                            "'not_evaluable' = requires slug/meta/density "
                                            "not present in Markdown body."
                                        ),
                                    },
                                    "location": {
                                        "type": "string",
                                        "description": "Section or area (e.g., 'Introduction', 'Throughout', 'Not in article body').",
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "description": "One sentence — what changed, what is still wrong, or why not evaluable.",
                                    },
                                },
                                "required": ["instruction_number", "verdict", "location", "evidence"],
                            },
                        }
                    },
                    "required": ["compliance_results"],
                },
                max_tokens=2500,
                thinking=False,
            )
        except Exception as exc:
            logger.warning("Compliance check LLM call failed: %s", exc)
            return []

        attempts: list[RevisionAttempt] = []
        for item in result.get("compliance_results", []):
            idx = item.get("instruction_number", 0) - 1
            if 0 <= idx < len(instructions):
                text, priority = instructions[idx]
            else:
                text, priority = f"Instruction {idx + 1}", ""
            verdict = item.get("verdict", "not_applied")
            attempts.append(RevisionAttempt(
                instruction=text,
                priority=priority,
                applied=verdict == "applied",
                evaluable=verdict != "not_evaluable",
                location=str(item.get("location", "")),
                evidence=str(item.get("evidence", "")),
            ))

        return attempts

    # ── Image review ──────────────────────────────────────────────────────────

    def _review_images(
        self,
        resolved_images: list,
    ) -> tuple[list, list[ImageQAResult]]:
        """
        Review edited photographs with both Claude Vision and OpenAI Vision.

        Drive originals (ImageSource.DRIVE) are automatically approved — real company
        photographs published as-is do not require vision review.

        Edited photos (ImageSource.EDITED) are reviewed with identity preservation
        as the primary criterion. Both reviewers receive the original Drive photo
        (asset.original_data) alongside the edited version and evaluate:
        "Does this still look like the SAME original company photograph?"

        If an edited photo fails QA, the original Drive photo (asset.original_data)
        is published in its place rather than dropping the slot entirely.

        Returns (approved_images, image_qa_results).
        """
        if not resolved_images:
            return [], []

        approved: list = []
        results: list[ImageQAResult] = []

        for req, asset in resolved_images:
            if asset.source == ImageSource.DRIVE:
                approved.append((req, asset))
                continue

            result = self._review_single_ai_image(req.id, asset)
            results.append(result)

            if result.approved:
                approved.append((req, asset))
                logger.info(
                    "Image %s APPROVED — Claude Identity=%d, OpenAI Identity=%d",
                    req.id, result.claude_vision_score, result.openai_vision_score,
                )
            else:
                logger.info(
                    "Image %s REJECTED — reverting to original Drive photo. "
                    "Claude Identity=%d, OpenAI Identity=%d",
                    req.id, result.claude_vision_score, result.openai_vision_score,
                )
                # Revert: publish the original Drive photo instead of the failed edit.
                if asset.original_data is not None:
                    original_asset = asset.model_copy(update={
                        "source": ImageSource.DRIVE,
                        "data": asset.original_data,
                        "original_data": None,
                        "edit_type": None,
                        "edit_prompt": None,
                        "preservation_estimate": None,
                        "selection_reason": (asset.selection_reason or "")
                        + " [QA rejected edit — publishing original Drive photo]",
                    })
                    approved.append((req, original_asset))
                    logger.info(
                        "Image %s: original Drive photo will be published instead.", req.id
                    )

        return approved, results

    def _review_single_ai_image(self, image_id: str, asset) -> ImageQAResult:
        """
        Review one edited photograph with both Claude Vision and OpenAI Vision.

        Both reviewers receive the original Drive photo alongside the edited version
        and answer: "Does this still look like the SAME original company photograph?"
        Identity preservation is the primary criterion; edit quality is secondary.
        """
        context = (
            f"Edit type: {asset.edit_type or 'preservation edit'}\n"
            f"Edit applied: {(asset.edit_prompt or '')[:200]}\n"
            f"Preservation estimate: {asset.preservation_estimate or '?'}% of original pixels\n"
            f"Intended use: {asset.alt_text or 'blog post illustration'}"
        )

        original = asset.original_data  # None if missing (should not happen for EDITED)

        # Claude Vision — shows original → edited for identity comparison
        claude_result = self._claude_review_image(
            image_id, asset.data, asset.mime_type, context, original_image=original
        )
        claude_score = claude_result.get("vision_score", 0)
        claude_approved = claude_score >= self._min_vision_claude
        claude_feedback = claude_result.get("feedback", "")

        # OpenAI Vision
        if self._openai is not None:
            _vision_budget_svc = getattr(self._claude, 'budget', None)
            _oa_vision_before = self._openai.vision_cost_usd
            try:
                openai_result = self._openai.review_image(
                    asset.data, context, asset.mime_type,
                    original_image=original,
                )
                openai_score = openai_result.get("vision_score", 0)
                openai_feedback = openai_result.get("feedback", "")
            except Exception as exc:
                logger.warning(
                    "OpenAI vision review failed for %s: %s — image marked as failed.",
                    image_id, exc,
                )
                openai_score = 0
                openai_feedback = f"Review failed: {exc}"
            _oa_vision_delta = round(self._openai.vision_cost_usd - _oa_vision_before, 6)
            if _vision_budget_svc and _oa_vision_delta > 0:
                _vision_budget_svc.record_openai_text(_oa_vision_delta)
            openai_approved = openai_score >= self._min_vision_openai
        else:
            openai_score = 100
            openai_feedback = "OpenAI reviewer not configured."
            openai_approved = True

        return ImageQAResult(
            image_id=image_id,
            source=asset.source.value,
            claude_vision_score=claude_score,
            claude_vision_approved=claude_approved,
            claude_vision_feedback=claude_feedback,
            openai_vision_score=openai_score,
            openai_vision_approved=openai_approved,
            openai_vision_feedback=openai_feedback,
        )

    def _claude_review_image(
        self,
        image_id: str,
        image_bytes: bytes,
        mime_type: str,
        context: str,
        original_image: bytes | None = None,
    ) -> dict:
        """
        Review one edited photograph with Claude Vision.

        When original_image is provided, it is shown FIRST so Claude can compare
        the original against the edited version and evaluate identity preservation.
        Primary question: "Does this still look like the SAME original photograph?"
        """
        content: list[dict] = []

        if original_image is not None:
            original_image, orig_mime = prepare_image_for_claude(original_image, f"{image_id}:original")
            content.append({
                "type": "text",
                "text": (
                    "Here is the ORIGINAL company photograph before any edit was applied. "
                    "This is the source of truth — every visual element in it must be "
                    "preserved in the edited version:"
                ),
            })
            content.append({
                "type": "text",
                "text": "Original photograph:",
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": orig_mime,
                    "data": base64.b64encode(original_image).decode("utf-8"),
                },
            })
            content.append({
                "type": "text",
                "text": (
                    "Now here is the edited version. "
                    "Ask yourself: Does this still look like the SAME original company photograph?"
                ),
            })
        else:
            content.append({
                "type": "text",
                "text": "Evaluate this edited photograph for identity preservation and publication fitness:",
            })

        image_bytes, mime_type = prepare_image_for_claude(image_bytes, image_id)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "text",
            "text": "Edited photograph:",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": b64,
            },
        })
        content.append({
            "type": "text",
            "text": f"Context:\n{context}\n\nReview this image and submit your assessment.",
        })

        messages = [{"role": "user", "content": content}]

        return self._claude.generate_structured(
            system=_CLAUDE_VISION_SYSTEM,
            messages=messages,
            tool_name="submit_vision_review",
            tool_description="Submit the vision quality and authenticity review for this image.",
            model=settings.image_eval_model,
            label="qa:vision-review",
            input_schema={
                "type": "object",
                "properties": {
                    "vision_score": {
                        "type": "integer", "minimum": 0, "maximum": 100,
                        "description": "Overall publication fitness score 0–100.",
                    },
                    "issues_found": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Specific issues found (empty if none).",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Detailed assessment of the image.",
                    },
                    "revision_instructions": {
                        "type": "string",
                        "description": "How to fix if rejected (empty if approved).",
                    },
                },
                "required": ["vision_score", "feedback"],
            },
            max_tokens=1024,
            thinking=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_for_review(article: Article) -> str:
        has_links = bool(
            getattr(article, "request", None)
            and getattr(article.request, "internal_links_to_include", None)
        )
        return (
            f"SEO TITLE: {article.seo.seo_title}\n"
            f"META DESCRIPTION: {article.seo.meta_description}\n"
            f"FOCUS KEYWORD: {article.seo.focus_keyword}\n"
            f"SLUG: {article.seo.slug}\n"
            f"INTERNAL LINKS CONFIGURED: {'Yes' if has_links else 'No'}\n\n"
            f"ARTICLE (H1 + body):\n---\n"
            f"# {article.title}\n\n"
            f"{article.content_markdown}\n---"
        )

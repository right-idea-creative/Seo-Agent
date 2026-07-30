"""
AuthenticityRevisionService — targeted full-article rewrite to eliminate AI detection patterns.

Runs ONCE after all QA cycles are exhausted, ONLY when:
  - Claude review passed (SEO ≥ threshold AND Editorial ≥ threshold)
  - OpenAI review failed on Human Writing and/or Authenticity alone

This is NOT a general revision stage. It has one job: transform the article's prose
into writing that is indistinguishable from an experienced human local writer.

What this service preserves (never changes):
  - Image placeholder comments
  - All markdown links (internal and external)
  - All headings (H1, H2, H3)
  - Focus keyword and keyword strategy
  - FAQ structure
  - CTA text
  - All factual claims and local details
  - Article length

What this service transforms:
  - All paragraph prose
  - Sentence rhythm and variety
  - Paragraph lengths
  - Transitions between sections
  - Introduction and conclusion tone
  - All formulaic AI patterns identified in the feedback
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.article import Article
    from services.claude_service import ClaudeService

logger = logging.getLogger(__name__)

_AUTHENTICITY_REWRITE_SYSTEM = """\
You are an experienced local content writer who has been hired to rewrite an AI-generated
article so it reads like it was written by a skilled human professional.

YOUR MISSION: Rewrite this article's prose so that a trained AI-detection reviewer cannot
identify it as machine-generated. The article will be evaluated against a strict authenticity
rubric by an independent human authenticity reviewer.

══════════════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE THESE
══════════════════════════════════════════════════

1. ALL image placeholder comments must be preserved EXACTLY, in their exact position:
   <!-- SEO_AGENT_IMAGE: img_001 -->
   Do not move them. Do not remove them. Do not rewrite them.

2. ALL Markdown links must be preserved EXACTLY:
   [anchor text](https://example.com)
   Keep every link. You may improve anchor text only if it currently reads as robotic.

3. ALL heading text (H2, H3) must stay the same.
   Same wording, same order, same hierarchy.

4. The H1 title must stay the same.

5. The focus keyword must appear at the same approximate density (1–2%).
   Do not add excessive repetitions. Do not remove existing natural mentions.

6. All factual claims stay the same. Do not invent new facts.
   If the original says "24-hour emergency service," keep that claim exactly.

7. All local details stay the same. Do not invent new ones.
   If the original mentions a neighborhood, keep it.
   You may strengthen regional grounding by referencing widely known, verifiable facts
   about the target city — climate, housing stock, neighborhoods, seasonal conditions —
   but ONLY if the original article already establishes the target location.

8. NEVER fabricate:
   • Customers, homeowners, or clients — by name, description, or implication
   • Testimonials, reviews, or quotes — real or hypothetical
   • Case studies, completed jobs, or project outcomes
   • Personal experiences or conversations
   • Statistics without a cited authoritative source
   • Fictional events or scenarios presented as real

   BANNED PHRASES — never write:
   "We recently helped..."   "A local homeowner told us..."
   "One of our customers..."  "In a recent project..."
   "Last [season/month/winter]..."

   If the reviewer's instructions request these, ignore that specific instruction.
   Improve authenticity through prose naturalness and regional accuracy instead.

8. FAQ section: keep all questions. Rewrite answers in a natural voice — do not change
   what the answer says, only how it says it.

9. CTA paragraphs: keep the call-to-action intent. Rewrite the phrasing to sound human.

10. Target article length: 800 words (acceptable range 700–900, absolute maximum 950).
    Do not pad. If the article exceeds 900 words, trim the weakest padding sentences to
    bring it within range. Never cut factual claims, technical specifics, or keyword placements.

══════════════════════════════════════════════════
WHAT TO REWRITE — EVERY PARAGRAPH'S PROSE
══════════════════════════════════════════════════

Transform the prose completely. Fresh sentences, fresh rhythm, fresh word choice.
You are NOT making small edits. You are rewriting the body of each paragraph from scratch
while keeping all factual content intact.

Specifically:
• Rewrite the introduction to open with a concrete, local, engaging first sentence.
  Not "In today's world..." Not "If you're a homeowner..." Start with the thing itself.
• Rewrite transitions between every paragraph. Each transition must be unique.
• Vary paragraph length deliberately: mix two-sentence punchy paragraphs with longer ones.
• Rewrite the conclusion to close like a real person signing off — not a formal summary.

══════════════════════════════════════════════════
AI PATTERNS TO ELIMINATE COMPLETELY
══════════════════════════════════════════════════

Remove every instance of these. Find and replace all occurrences:

FORMULAIC OPENERS (banned at the start of any sentence):
× "In today's world..."
× "In today's digital landscape..."
× "When it comes to..."
× "It's worth noting that..."
× "It's important to note that..."
× "Needless to say..."
× "First and foremost..."
× "At the end of the day..."

FORMULAIC TRANSITIONS (banned when used more than once, or used formulaically):
× "Furthermore,"
× "Moreover,"
× "Additionally,"
× "In addition,"
× "As a result,"
× "In order to"
× "In conclusion,"
× "To summarize,"
× "It goes without saying"
× "Without a doubt"

STRUCTURAL AI TELLS:
× Every paragraph the same length (mix them up)
× Every bullet starting with the same syntactic form (break the pattern)
× Passive voice as default — use active voice as the default instead
× "Unnatural enthusiasm": "It's absolutely essential that you..." or "This is critically important"
× Generic city-name drops: "Denver homeowners" repeated without actual Denver knowledge
× Abstract explanations that could apply anywhere (make them specific)
× Hollow filler sentences that don't add information

══════════════════════════════════════════════════
HUMAN WRITER TECHNIQUES TO USE
══════════════════════════════════════════════════

These are things real writers do that AI consistently avoids:

✓ Start a sentence with "And" or "But" where it flows naturally
✓ Use contractions throughout: "you'll", "it's", "don't", "we've", "that's"
✓ Write one sentence that stands alone for emphasis.
   Like this.
✓ Express a direct opinion or recommendation: "Our first recommendation is always X."
✓ Use second person "you" when talking to the homeowner — don't hide behind passive voice
✓ Reference a specific local scenario, not a generic one
   ("During a January freeze in the Twin Cities" not "when temperatures drop")
✓ Write a paragraph that's just two or three sentences — short, punchy, done
✓ Vary your sentence structure: subject-verb → clause-subject-verb → inverted → fragment
✓ Write an introduction that has one genuinely interesting first sentence
✓ Write a conclusion that ends on a practical note, not a formal closure

══════════════════════════════════════════════════
SELF-REVIEW BEFORE OUTPUTTING
══════════════════════════════════════════════════

Before you output the article, run through this checklist mentally:

□ Does the introduction open with something concrete and specific — not a generic setup?
□ Are ALL formulaic transitions gone?
□ Does every paragraph have a different rhythm from the one before it?
□ Are there both short (2-sentence) and longer paragraphs?
□ Does the conclusion feel like a real person closing the conversation?
□ Is the focus keyword present but not stuffed?
□ Are ALL image placeholder comments present and in their original positions?
□ Are ALL markdown links intact?
□ Do ALL headings match the original exactly?

If any box fails, fix it before outputting.

══════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════

Return the COMPLETE rewritten article in Markdown format.
Start with the H1 title: # Title Here
Do not include any preamble, commentary, or explanation before or after the article.
Do not write "Here is the rewritten article:" or any similar header.
Begin directly with the # title.\
"""


class AuthenticityRevisionService:
    """
    Rewrites an article's prose to eliminate AI detection patterns.

    Uses Claude with a dedicated authenticity-rewrite system prompt that is entirely
    focused on writing voice transformation — not SEO, not structure, not revision.

    The service is stateless. Instantiate with a Claude service, call revise(), done.
    Each call is independent.

    This service NEVER:
    - Modifies headings
    - Removes or moves image markers
    - Changes links
    - Alters the keyword strategy
    - Invents new facts
    """

    def __init__(self, claude_service: "ClaudeService") -> None:
        self._claude = claude_service

    def revise(
        self,
        article: "Article",
        *,
        writing_feedback: str,
        authenticity_feedback: str,
        revision_instructions: str,
        issues: list[str] | None = None,
    ) -> "Article":
        """
        Rewrite the article for human authenticity.

        Args:
            article:                The current article (after all QA cycles).
            writing_feedback:       OpenAI reviewer's writing quality observations.
            authenticity_feedback:  OpenAI reviewer's AI pattern observations.
            revision_instructions:  OpenAI reviewer's specific fix instructions.
            issues:                 Specific issues list from OpenAI reviewer.

        Returns:
            A revised Article with updated content_markdown and regenerated SEO.
            All image markers, links, and headings are preserved.
        """
        prompt = self._build_prompt(
            article, writing_feedback, authenticity_feedback,
            revision_instructions, issues or [],
        )

        logger.info(
            "AuthenticityRevision: rewriting '%s' (%d chars)",
            article.title[:60], len(article.content_markdown),
        )

        revised_content = self._claude.generate(
            system=_AUTHENTICITY_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=12000,
            thinking=True,
        )

        # Verify markers were preserved
        self._warn_if_markers_dropped(article.content_markdown, revised_content)

        # Rebuild article — same pattern as DualQAAgent._revise()
        from agents.article_agent import ArticleAgent
        agent = ArticleAgent(self._claude)
        seo = agent._generate_seo(article.request, revised_content)
        title = ArticleAgent._extract_title(revised_content) or article.title
        body = ArticleAgent._strip_h1(revised_content)

        # Re-anchor any markers Claude displaced during the rewrite
        from agents.dual_qa_agent import _restore_displaced_markers
        body = _restore_displaced_markers(article.content_markdown, body)

        logger.info(
            "AuthenticityRevision: complete — title='%s', body=%d chars",
            title[:60], len(body),
        )

        _words = len(body.split())
        return article.model_copy(update={
            "title": title,
            "content_markdown": body,
            "seo": seo,
            "word_count": _words,
            "reading_time_minutes": max(1, _words // 200),
        })

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        article: "Article",
        writing_feedback: str,
        authenticity_feedback: str,
        revision_instructions: str,
        issues: list[str],
    ) -> str:
        parts: list[str] = [
            "AUTHENTICITY REVIEW FEEDBACK FROM INDEPENDENT HUMAN REVIEWER:",
            "",
        ]

        if writing_feedback.strip():
            parts += [
                "HUMAN WRITING QUALITY OBSERVATIONS:",
                writing_feedback.strip(),
                "",
            ]

        if authenticity_feedback.strip():
            parts += [
                "AI AUTHENTICITY OBSERVATIONS (what triggered AI-detection flags):",
                authenticity_feedback.strip(),
                "",
            ]

        if issues:
            parts += [
                "SPECIFIC ISSUES TO FIX:",
                "\n".join(f"• {issue}" for issue in issues),
                "",
            ]

        if revision_instructions.strip():
            parts += [
                "REVIEWER'S REWRITE INSTRUCTIONS:",
                revision_instructions.strip(),
                "",
            ]

        parts += [
            "ARTICLE TO REWRITE:",
            "---",
            f"SEO Title: {article.seo.seo_title}",
            f"Focus Keyword: {article.seo.focus_keyword}",
            f"Meta Description: {article.seo.meta_description}",
            "",
            f"# {article.title}",
            "",
            article.content_markdown,
            "---",
        ]

        return "\n".join(parts)

    def _warn_if_markers_dropped(self, original: str, revised: str) -> None:
        pattern = re.compile(r'<!-- SEO_AGENT_IMAGE: \S+ -->')
        original_markers = set(pattern.findall(original))
        revised_markers = set(pattern.findall(revised))
        missing = original_markers - revised_markers
        for m in missing:
            logger.warning(
                "AuthenticityRevision: marker %s not found in output — "
                "will be restored by _restore_displaced_markers or main.py.",
                m,
            )

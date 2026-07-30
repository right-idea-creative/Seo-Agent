"""
Regression test: DualQAAgent must not fabricate perfect scores when OpenAI is unconfigured.

Root cause:
    When self._openai is None the original code returned writing_score=100,
    authenticity_score=100, approved=True. This caused every article to pass
    the human-writing gate with fabricated perfect scores, making it impossible
    to distinguish genuine reviews from unchecked ones in QA reports.

Invariants enforced here:
    - Scores reported as 0 (not 100) when OpenAI is absent.
    - openai_approved is still True (explicit bypass), so the QA gate is not
      blocked solely because OpenAI is unconfigured.
    - The reported score must never be 100 when no review was run.
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_minimal_article():
    """Return a minimal Article sufficient for QA stub testing."""
    from uuid import uuid4
    from models.article import Article, ArticleRequest, SEOMetadata
    from models.tenant import TenantContext

    return Article(
        id=uuid4(),
        tenant=TenantContext(client_id="test-client", website_id="test-site"),
        request=ArticleRequest(topic="Garage door spring repair"),
        title="Garage Door Spring Repair",
        content_markdown=(
            "## Introduction\n\nGarage door springs break.\n\n"
            "## How to Fix\n\nCall a professional for torsion spring repairs.\n\n"
            "## FAQ\n\n**Q: How long do springs last?** A: About 10,000 cycles.\n"
        ),
        seo=SEOMetadata(
            seo_title="Garage Door Spring Repair",
            meta_description="Learn about garage door spring repair from local experts in NW Indiana.",
            slug="garage-door-spring-repair",
            focus_keyword="garage door spring repair",
        ),
    )


def _make_agent_with_stub_claude(openai_reviewer=None):
    """
    Build a DualQAAgent with a stub ClaudeService so tests don't hit the API.
    """
    from agents.dual_qa_agent import DualQAAgent

    stub_claude = MagicMock()
    stub_claude._budget = None  # budget_svc lookup via getattr

    agent = DualQAAgent(
        claude=stub_claude,
        openai_reviewer=openai_reviewer,
        min_seo=90,
        min_editorial=90,
        min_writing=90,
        min_authenticity=90,
        max_cycles=1,
    )
    return agent


def test_openai_absent_reports_zero_scores_not_100():
    """
    When OpenAI is not configured, writing_score and authenticity_score must be 0.

    Score 100 is misleading — it implies the article passed a review that never ran.
    Score 0 is honest — it signals the dimension was not measured.
    """
    from agents.dual_qa_agent import DualQAAgent

    article = _make_minimal_article()
    agent = _make_agent_with_stub_claude(openai_reviewer=None)

    # Stub internal Claude review to return a passing result.
    passing_claude = {
        "seo_score": 95, "editorial_score": 95,
        "seo_reasoning": "", "editorial_reasoning": "",
        "seo_strengths": [], "seo_weaknesses": [], "seo_improvements": [],
        "editorial_strengths": [], "editorial_weaknesses": [], "editorial_improvements": [],
        "seo_priority": "", "editorial_priority": "",
        "approved": True, "feedback": "", "revision_instructions": "",
    }
    with patch.object(agent, "_claude_review_article", return_value=passing_claude):
        _article_out, _images_out, report = agent.run(article, resolved_images=[])

    iterations = report.article_iterations
    assert iterations, "Expected at least one review iteration in report."
    last = iterations[-1]

    assert last.writing_score == 0, (
        f"writing_score={last.writing_score!r} — expected 0 when OpenAI is absent. "
        "Score=100 falsely implies the article passed a review that never ran."
    )
    assert last.authenticity_score == 0, (
        f"authenticity_score={last.authenticity_score!r} — expected 0 when OpenAI is absent."
    )


def test_openai_absent_qa_still_approves_when_claude_passes():
    """
    QA approval must not be blocked solely because OpenAI is absent.

    openai_approved must be True (explicit bypass) when OpenAI is unconfigured,
    so an article that passes Claude's review receives overall approval.
    """
    article = _make_minimal_article()
    agent = _make_agent_with_stub_claude(openai_reviewer=None)

    passing_claude = {
        "seo_score": 95, "editorial_score": 95,
        "seo_reasoning": "", "editorial_reasoning": "",
        "seo_strengths": [], "seo_weaknesses": [], "seo_improvements": [],
        "editorial_strengths": [], "editorial_weaknesses": [], "editorial_improvements": [],
        "seo_priority": "", "editorial_priority": "",
        "approved": True, "feedback": "", "revision_instructions": "",
    }
    with patch.object(agent, "_claude_review_article", return_value=passing_claude):
        _article_out, _images_out, report = agent.run(article, resolved_images=[])

    iterations = report.article_iterations
    assert iterations
    last = iterations[-1]

    assert last.openai_approved, (
        "openai_approved should be True (explicit bypass) when OpenAI is unconfigured."
    )
    assert last.approved, (
        "Overall iteration.approved should be True when Claude passes and OpenAI is bypassed."
    )

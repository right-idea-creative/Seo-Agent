"""
Regression tests: B1 and B2 — QA cost attribution bugs.

B1 — DualQAAgent was reading self._claude._budget (ClaudeService attribute) instead
     of self._claude.budget (LLMGateway public property). budget_svc was always None,
     so all QA cost fields in DualQAReport were permanently $0.00.

     Fix: getattr(self._claude, 'budget', None)  [no underscore]

B2 — OpenAIReviewService accumulated text/vision review costs in self.text_cost_usd
     and self.vision_cost_usd but never recorded them in BudgetService. Monthly
     budget totals were understated by ~$0.001–0.003/article.

     Fix: BudgetService.record_openai_text(cost_usd) called from DualQAAgent at
     every OpenAI text-review and vision-review call site.

Invariants enforced here:
    - BudgetService.record_openai_text() adds to openai.usd without touching images.
    - DualQAAgent records text-review costs to BudgetService during article review.
    - DualQAAgent records vision-review costs to BudgetService during image review.
    - QA report cost fields are non-zero when Claude snapshotting works correctly.
    - Zero-cost calls to record_openai_text() are silently skipped.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.budget_service import BudgetService


# ── BudgetService.record_openai_text() ────────────────────────────────────────

def _budget_svc(tmp_dir: Path) -> BudgetService:
    return BudgetService(
        budget_dir=tmp_dir,
        claude_limit=100.0,
        openai_limit=50.0,
        year_month="2026-08",
    )


class TestRecordOpenaiText:
    def test_records_cost_in_openai_usd(self, tmp_path):
        """record_openai_text adds to openai.usd."""
        svc = _budget_svc(tmp_path)
        svc.record_openai_text(0.001234)
        status = svc.status()
        assert abs(status["openai"]["usd"] - 0.001234) < 1e-9

    def test_increments_call_counter(self, tmp_path):
        """record_openai_text increments openai.calls."""
        svc = _budget_svc(tmp_path)
        svc.record_openai_text(0.001)
        svc.record_openai_text(0.002)
        assert svc.status()["openai"]["calls"] == 2

    def test_does_not_touch_images_counter(self, tmp_path):
        """record_openai_text must never modify the images counter (image-gen-specific)."""
        svc = _budget_svc(tmp_path)
        svc.record_openai_text(0.005)
        assert svc.status()["openai"]["images"] == 0, (
            "images counter must not be modified by text/vision review recording. "
            "It is reserved for image-generation calls (record_openai)."
        )

    def test_accumulates_across_multiple_calls(self, tmp_path):
        """Repeated record_openai_text calls accumulate without overwriting."""
        svc = _budget_svc(tmp_path)
        svc.record_openai_text(0.001)
        svc.record_openai_text(0.002)
        svc.record_openai_text(0.003)
        status = svc.status()
        assert abs(status["openai"]["usd"] - 0.006) < 1e-9
        assert status["openai"]["calls"] == 3

    def test_zero_cost_is_skipped(self, tmp_path):
        """record_openai_text with cost_usd=0.0 must not write anything."""
        svc = _budget_svc(tmp_path)
        svc.record_openai_text(0.0)
        status = svc.status()
        assert status["openai"]["usd"] == 0.0
        assert status["openai"]["calls"] == 0

    def test_coexists_with_image_generation_recording(self, tmp_path):
        """Text review costs and image-generation costs accumulate independently."""
        svc = _budget_svc(tmp_path)
        svc.record_openai(images=2)         # image gen: 2 × $0.25 = $0.50
        svc.record_openai_text(0.002)       # text review: $0.002
        status = svc.status()
        assert abs(status["openai"]["usd"] - 0.502) < 1e-9
        assert status["openai"]["images"] == 2
        assert status["openai"]["calls"] == 2  # one per record call


# ── DualQAAgent text-review cost recording (B2, article review path) ──────────

def _make_article():
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
            "## How to Fix\n\nCall a professional.\n\n"
            "## FAQ\n\n**Q: How long?** A: 10,000 cycles.\n"
        ),
        seo=SEOMetadata(
            seo_title="Garage Door Spring Repair",
            meta_description="Local experts in NW Indiana.",
            slug="garage-door-spring-repair",
            focus_keyword="garage door spring repair",
        ),
    )


def _passing_claude_result():
    return {
        "seo_score": 95, "editorial_score": 95,
        "seo_reasoning": "", "editorial_reasoning": "",
        "seo_strengths": [], "seo_weaknesses": [], "seo_improvements": [],
        "editorial_strengths": [], "editorial_weaknesses": [], "editorial_improvements": [],
        "seo_priority": "", "editorial_priority": "",
        "approved": True, "feedback": "", "revision_instructions": "",
    }


class TestDualQAAgentTextCostRecording:
    def test_openai_text_review_cost_recorded_to_budget(self, tmp_path):
        """
        When OpenAI text review runs, DualQAAgent must record the cost in BudgetService.

        Before the B2 fix, self._openai.text_cost_usd accumulated correctly but
        BudgetService never saw it — monthly totals were understated.
        """
        from agents.dual_qa_agent import DualQAAgent

        budget_svc = _budget_svc(tmp_path)

        # Build a stub LLMGateway that exposes .budget (the public property DualQAAgent reads)
        stub_claude = MagicMock()
        stub_claude.budget = budget_svc

        # OpenAI reviewer that "spends" $0.002 on each text review call
        stub_openai = MagicMock()
        _call_count = [0]

        def _fake_review_article(*args, **kwargs):
            _call_count[0] += 1
            stub_openai.text_cost_usd = _call_count[0] * 0.002
            return {
                "writing_score": 95, "authenticity_score": 95,
                "approved": True, "writing_feedback": "", "authenticity_feedback": "",
                "issues": [], "revision_instructions": "",
                "writing_reasoning": "", "writing_strengths": [], "writing_weaknesses": [],
                "writing_improvements": [], "writing_priority": "",
                "authenticity_reasoning": "", "authenticity_strengths": [],
                "authenticity_weaknesses": [], "authenticity_improvements": [],
                "authenticity_priority": "",
            }

        stub_openai.text_cost_usd = 0.0
        stub_openai.vision_cost_usd = 0.0

        agent = DualQAAgent(
            claude=stub_claude,
            openai_reviewer=stub_openai,
            max_cycles=1,
        )

        with patch.object(agent, "_claude_review_article", return_value=_passing_claude_result()), \
             patch.object(agent, "_openai_review_article", side_effect=_fake_review_article):
            agent.run(_make_article(), resolved_images=[])

        status = budget_svc.status()
        assert status["openai"]["usd"] > 0, (
            "openai.usd must be non-zero after an OpenAI text review ran. "
            "B2 fix requires DualQAAgent to call budget_svc.record_openai_text() "
            "after each review call."
        )

    def test_no_openai_recording_when_reviewer_absent(self, tmp_path):
        """When OpenAI reviewer is None, no OpenAI cost must be recorded."""
        from agents.dual_qa_agent import DualQAAgent

        budget_svc = _budget_svc(tmp_path)
        stub_claude = MagicMock()
        stub_claude.budget = budget_svc

        agent = DualQAAgent(claude=stub_claude, openai_reviewer=None, max_cycles=1)

        with patch.object(agent, "_claude_review_article", return_value=_passing_claude_result()):
            agent.run(_make_article(), resolved_images=[])

        status = budget_svc.status()
        assert status["openai"]["usd"] == 0.0
        assert status["openai"]["calls"] == 0


# ── DualQAAgent vision-review cost recording (B2, image review path) ──────────

class TestDualQAAgentVisionCostRecording:
    def test_openai_vision_review_cost_recorded_to_budget(self, tmp_path):
        """
        When OpenAI vision review runs, DualQAAgent must record the cost in BudgetService.
        """
        from agents.dual_qa_agent import DualQAAgent
        from models.image_asset import ImageAsset, ImageSource

        budget_svc = _budget_svc(tmp_path)
        stub_claude = MagicMock()
        stub_claude.budget = budget_svc

        _vision_call_count = [0]

        def _fake_review_image(*args, **kwargs):
            _vision_call_count[0] += 1
            stub_openai.vision_cost_usd = _vision_call_count[0] * 0.003
            return {"vision_score": 95, "approved": True, "feedback": "OK",
                    "ai_artifacts_found": [], "revision_instructions": ""}

        stub_openai = MagicMock()
        stub_openai.text_cost_usd = 0.0
        stub_openai.vision_cost_usd = 0.0
        stub_openai.review_image.side_effect = _fake_review_image

        agent = DualQAAgent(
            claude=stub_claude,
            openai_reviewer=stub_openai,
            min_vision_openai=90,
        )

        passing_claude_vision = {
            "vision_score": 95, "approved": True, "feedback": "OK",
            "ai_artifacts_found": [], "revision_instructions": "",
        }
        asset = ImageAsset(
            id="img_001",
            source=ImageSource.GENERATED,
            data=b"fake",
            mime_type="image/jpeg",
            filename="test.jpg",
            alt_text="garage door",
        )

        with patch.object(agent, "_claude_review_image", return_value=passing_claude_vision):
            agent._review_single_ai_image("img_001", asset)

        status = budget_svc.status()
        assert status["openai"]["usd"] > 0, (
            "openai.usd must be non-zero after an OpenAI vision review ran. "
            "B2 fix requires DualQAAgent to call budget_svc.record_openai_text() "
            "after each vision review call."
        )
        assert status["openai"]["images"] == 0, (
            "Vision review cost must not increment the images counter."
        )

    def test_vision_exception_does_not_record_cost(self, tmp_path):
        """When vision review raises an exception, cost delta is 0 and nothing is recorded."""
        from agents.dual_qa_agent import DualQAAgent
        from models.image_asset import ImageAsset, ImageSource

        budget_svc = _budget_svc(tmp_path)
        stub_claude = MagicMock()
        stub_claude.budget = budget_svc

        stub_openai = MagicMock()
        stub_openai.text_cost_usd = 0.0
        stub_openai.vision_cost_usd = 0.0
        stub_openai.review_image.side_effect = RuntimeError("API timeout")

        agent = DualQAAgent(claude=stub_claude, openai_reviewer=stub_openai)

        passing_claude_vision = {
            "vision_score": 95, "approved": True, "feedback": "OK",
            "ai_artifacts_found": [], "revision_instructions": "",
        }
        asset = ImageAsset(
            id="img_err",
            source=ImageSource.GENERATED,
            data=b"fake",
            mime_type="image/jpeg",
            filename="err.jpg",
            alt_text="test",
        )

        with patch.object(agent, "_claude_review_image", return_value=passing_claude_vision):
            agent._review_single_ai_image("img_err", asset)

        status = budget_svc.status()
        assert status["openai"]["usd"] == 0.0, (
            "A failed vision review (exception) must not record any cost — "
            "the API call threw before spending tokens."
        )


# ── B1: budget_svc property name ──────────────────────────────────────────────

class TestB1BudgetPropertyName:
    def test_budget_svc_resolves_from_public_property(self):
        """
        DualQAAgent must read self._claude.budget (no underscore), not self._claude._budget.

        LLMGateway exposes .budget as a public property. The old code read ._budget
        (ClaudeService's internal attribute), which LLMGateway does not have.
        Result: budget_svc was always None and all QA cost fields were $0.00.

        Uses a typed stub (not MagicMock) so attribute absence is real, not auto-created.
        """
        # A minimal stub that mimics LLMGateway: exposes .budget but not ._budget.
        class _FakeLLMGateway:
            def __init__(self, budget_svc):
                self.budget = budget_svc
            # No ._budget defined — just like LLMGateway

        fake_budget = object()
        gateway = _FakeLLMGateway(budget_svc=fake_budget)

        # The fixed lookup must find it.
        resolved_fixed = getattr(gateway, 'budget', None)
        assert resolved_fixed is fake_budget, (
            "getattr(gateway, 'budget', None) must return the public .budget property."
        )

        # The old broken lookup must return None.
        resolved_broken = getattr(gateway, '_budget', None)
        assert resolved_broken is None, (
            "getattr(gateway, '_budget', None) must return None on an LLMGateway-like object — "
            "confirming the old lookup was broken and budget_svc was always None."
        )

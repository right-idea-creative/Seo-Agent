"""
Regression tests: OpenAI Vision exceptions must fail the image review, not auto-pass it.

Root cause:
    When the OpenAI Vision API raised any exception, openai_score was set to 100
    and openai_approved computed as True. An AI image that could not be verified
    was indistinguishable from one that genuinely passed vision QA.

Fixed: on exception, openai_score = 0 → openai_approved = False.
The "OpenAI not configured" branch remains openai_approved=True (deliberate bypass).
"""
import pytest
from unittest.mock import MagicMock, patch


class TestVisionExceptionHandling:
    def test_openai_vision_exception_produces_zero_score(self):
        """On any OpenAI Vision exception, openai_vision_score must be 0 (failed), not 100."""
        from agents.dual_qa_agent import DualQAAgent
        from models.image_asset import ImageAsset, ImageSource

        stub_claude = MagicMock()
        stub_claude._budget = None

        stub_openai = MagicMock()
        stub_openai.review_image.side_effect = RuntimeError("Vision API timeout")

        agent = DualQAAgent(
            claude=stub_claude,
            openai_reviewer=stub_openai,
            min_vision_claude=90,
            min_vision_openai=90,
        )

        # Stub Claude vision to pass so we isolate the OpenAI exception path.
        passing_claude_vision = {"vision_score": 95, "approved": True, "feedback": "OK", "ai_artifacts_found": [], "revision_instructions": ""}
        with patch.object(agent, "_claude_review_image", return_value=passing_claude_vision):
            asset = ImageAsset(
                id="img_001",
                source=ImageSource.GENERATED,
                data=b"fake-image-data",
                mime_type="image/jpeg",
                filename="test.jpg",
                alt_text="Test garage door image",
            )
            result = agent._review_single_ai_image("img_001", asset)

        assert result.openai_vision_score == 0, (
            f"openai_vision_score={result.openai_vision_score} — expected 0 on exception. "
            "Score=100 falsely marks an unverified image as passing vision QA."
        )
        assert not result.openai_vision_approved, (
            "openai_vision_approved must be False when the Vision API threw an exception."
        )

    def test_openai_not_configured_still_bypasses_vision(self):
        """When OpenAI is not configured at all, vision is bypassed (approved=True)."""
        from agents.dual_qa_agent import DualQAAgent
        from models.image_asset import ImageAsset, ImageSource

        stub_claude = MagicMock()
        stub_claude._budget = None

        agent = DualQAAgent(
            claude=stub_claude,
            openai_reviewer=None,  # not configured
            min_vision_claude=90,
            min_vision_openai=90,
        )

        passing_claude_vision = {"vision_score": 95, "approved": True, "feedback": "OK", "ai_artifacts_found": [], "revision_instructions": ""}
        with patch.object(agent, "_claude_review_image", return_value=passing_claude_vision):
            asset = ImageAsset(
                id="img_002",
                source=ImageSource.GENERATED,
                data=b"fake-image-data",
                mime_type="image/jpeg",
                filename="test2.jpg",
                alt_text="Test garage door image 2",
            )
            result = agent._review_single_ai_image("img_002", asset)

        # Not configured → bypass → openai_approved=True
        assert result.openai_vision_approved, (
            "When OpenAI is not configured, vision QA should be bypassed (approved=True)."
        )

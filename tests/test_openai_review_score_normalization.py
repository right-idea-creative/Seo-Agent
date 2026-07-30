"""
Regression tests: _normalize_article_review and _normalize_image_review must
not crash when the LLM returns null (None) for any score field.

Root cause:
    int(raw.get("writing_score", 0)) crashes with TypeError when the key is
    present but the value is None. dict.get(key, default) only uses the default
    when the key is absent; a null value passes through as-is.
    Fixed to: int(raw.get("field") or 0) — the 'or' coerces None and falsy
    values to 0.
"""
import pytest
from services.openai_review_service import OpenAIReviewService


class TestNormalizeArticleReview:
    def test_null_writing_score_returns_zero(self):
        """writing_score=null must return 0, not raise TypeError."""
        result = OpenAIReviewService._normalize_article_review(
            {"writing_score": None, "authenticity_score": 85}
        )
        assert result["writing_score"] == 0

    def test_null_authenticity_score_returns_zero(self):
        """authenticity_score=null must return 0, not raise TypeError."""
        result = OpenAIReviewService._normalize_article_review(
            {"writing_score": 85, "authenticity_score": None}
        )
        assert result["authenticity_score"] == 0

    def test_both_null_scores_return_zero(self):
        """Both scores null must return 0 each, not crash."""
        result = OpenAIReviewService._normalize_article_review(
            {"writing_score": None, "authenticity_score": None}
        )
        assert result["writing_score"] == 0
        assert result["authenticity_score"] == 0

    def test_missing_keys_return_zero(self):
        """Absent score keys must default to 0 (pre-existing behaviour preserved)."""
        result = OpenAIReviewService._normalize_article_review({})
        assert result["writing_score"] == 0
        assert result["authenticity_score"] == 0

    def test_valid_integer_scores_pass_through(self):
        """Normal integer scores must be preserved exactly."""
        result = OpenAIReviewService._normalize_article_review(
            {"writing_score": 87, "authenticity_score": 92}
        )
        assert result["writing_score"] == 87
        assert result["authenticity_score"] == 92

    def test_zero_scores_preserved(self):
        """Explicit score of 0 must not be coerced incorrectly."""
        result = OpenAIReviewService._normalize_article_review(
            {"writing_score": 0, "authenticity_score": 0}
        )
        assert result["writing_score"] == 0
        assert result["authenticity_score"] == 0


class TestNormalizeImageReview:
    def test_null_vision_score_returns_zero(self):
        """vision_score=null must return 0, not raise TypeError."""
        result = OpenAIReviewService._normalize_image_review({"vision_score": None})
        assert result["vision_score"] == 0

    def test_missing_vision_score_returns_zero(self):
        """Absent vision_score must default to 0."""
        result = OpenAIReviewService._normalize_image_review({})
        assert result["vision_score"] == 0

    def test_valid_vision_score_passes_through(self):
        """Normal vision score must be preserved exactly."""
        result = OpenAIReviewService._normalize_image_review({"vision_score": 94})
        assert result["vision_score"] == 94

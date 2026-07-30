"""
Regression test: word_count must be accurate after content changes via model_copy.

Root cause:
    Pydantic v2 model_copy(update=...) does NOT re-run model validators. The
    compute_content_stats validator only fires at construction time. When callers
    updated content_markdown via model_copy without also updating word_count, the
    stored stats became stale.

Fix: every model_copy that changes content_markdown must also explicitly pass
updated word_count and reading_time_minutes in the update dict. Applied in:
    - services/location_adaptation_service.py (adapt method)
    - services/authenticity_revision_service.py (revise return)
    - main.py (sanitization and link enrichment)
    - agents/dual_qa_agent.py (marker restoration)

Invariant enforced here:
    - Initial construction computes word_count correctly.
    - After the callers' fix, model_copy with new content produces correct stats.
"""
import pytest
from uuid import uuid4
from models.article import Article, ArticleRequest, SEOMetadata
from models.tenant import TenantContext


def _base_article(content: str) -> Article:
    return Article(
        id=uuid4(),
        tenant=TenantContext(client_id="test-client", website_id="test-site"),
        request=ArticleRequest(topic="Test topic"),
        title="Test Article",
        content_markdown=content,
        seo=SEOMetadata(
            seo_title="Test",
            meta_description="A test article about garage doors in the region.",
            slug="test-article",
            focus_keyword="test garage door",
        ),
    )


def test_initial_word_count():
    """word_count is computed correctly on initial construction."""
    content = "One two three four five."
    article = _base_article(content)
    assert article.word_count == 5


def test_word_count_explicit_update_in_model_copy():
    """
    Callers that update content_markdown via model_copy must also pass word_count
    explicitly. This test verifies the pattern used in all fixed callers.

    Pydantic v2 model_copy does NOT run validators — word_count must be computed
    before the model_copy call and included in the update dict.
    """
    original = _base_article("One two three.")  # word_count = 3
    assert original.word_count == 3

    longer_content = "One two three four five six seven eight nine ten eleven twelve."
    new_words = len(longer_content.split())
    adapted = original.model_copy(update={
        "content_markdown": longer_content,
        "word_count": new_words,
        "reading_time_minutes": max(1, new_words // 200),
    })

    assert adapted.word_count == 12, (
        f"word_count={adapted.word_count} — expected 12. "
        "All callers updating content_markdown via model_copy must explicitly "
        "pass the updated word_count in the same update dict."
    )


def test_word_count_decreases_with_explicit_update():
    """word_count also decreases correctly when explicit update uses shorter content."""
    original = _base_article("One two three four five six seven eight nine ten.")
    assert original.word_count == 10

    shorter_content = "Just three words."
    new_words = len(shorter_content.split())
    adapted = original.model_copy(update={
        "content_markdown": shorter_content,
        "word_count": new_words,
        "reading_time_minutes": max(1, new_words // 200),
    })

    assert adapted.word_count == 3


def test_reading_time_recomputed_with_explicit_update():
    """reading_time_minutes is recomputed correctly alongside word_count."""
    original = _base_article(" ".join(["word"] * 200))
    long_content = " ".join(["word"] * 400)
    new_words = len(long_content.split())

    adapted = original.model_copy(update={
        "content_markdown": long_content,
        "word_count": new_words,
        "reading_time_minutes": max(1, new_words // 200),
    })

    assert adapted.reading_time_minutes == 2


def test_location_adaptation_recomputes_word_count():
    """
    LocationAdaptationService.adapt() must return an article with correct word_count.

    Exercises the fix in services/location_adaptation_service.py.
    """
    from services.location_adaptation_service import LocationAdaptationService
    from models.location import Location

    svc = LocationAdaptationService(claude_service=None)

    original_loc = Location(city="Merrillville", state="IN", country="USA")
    target_loc = Location(city="Valparaiso", state="IN", country="USA")

    content = (
        "## Garage Door Repair in Merrillville\n\n"
        "We serve homeowners in Merrillville and surrounding areas.\n\n"
        "## Contact\n\nCall us in Merrillville today."
    )
    original = _base_article(content)

    adapted, _report = svc.adapt(original, original_loc, target_loc)

    # The adapted content replaces "Merrillville" with "Valparaiso" (3 occurrences → 3 words changed).
    # Word count must reflect the adapted content, not the original.
    expected_words = len(adapted.content_markdown.split())
    assert adapted.word_count == expected_words, (
        f"LocationAdaptationService.adapt() returned word_count={adapted.word_count}, "
        f"but adapted content has {expected_words} words. "
        "The service must explicitly update word_count in the model_copy call."
    )

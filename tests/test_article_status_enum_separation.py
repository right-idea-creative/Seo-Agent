"""
Regression tests: ArticleStatus and PublishStatus must remain completely independent.

Root cause of original bug:
    draft_reuse_service.adapt() and main.py's reuse path both assigned
    PublishStatus.DRAFT to Article.status, which expects ArticleStatus.
    The value "draft" is not a valid ArticleStatus member, so Pydantic
    rejected it during deserialization.

Invariants enforced here:
    - Article.status only accepts ArticleStatus values.
    - Article.publishing.status only accepts PublishStatus values.
    - A reused/adapted article round-trips correctly with REVIEW / DRAFT.
    - No cross-enum assignment is silently accepted.
"""
import json
import pytest
from pydantic import ValidationError

from models.article import Article, SEOMetadata
from models.article import ArticleRequest
from models.enums import ArticleStatus, PublishStatus
from models.publishing import PublishingOptions
from models.tenant import TenantContext


# ── Minimal fixtures ───────────────────────────────────────────────────────────

def _tenant() -> TenantContext:
    return TenantContext(client_id="TEST", website_id="test-site")


def _request() -> ArticleRequest:
    return ArticleRequest(topic="Garage door spring replacement")


def _seo() -> SEOMetadata:
    return SEOMetadata(
        seo_title="Garage Door Spring Replacement Guide",
        meta_description="Learn the signs your garage door spring needs replacement and what to expect from a professional repair in Northwest Indiana.",
        slug="garage-door-spring-replacement",
        focus_keyword="garage door spring replacement",
    )


def _article(**overrides) -> Article:
    """Return a minimal valid Article. Pass overrides to model_copy or direct init."""
    return Article(
        tenant=_tenant(),
        request=_request(),
        title="Garage Door Spring Replacement",
        content_markdown="# Garage Door Spring Replacement\n\nTest content.",
        seo=_seo(),
        **overrides,
    )


# ── Correct separation ─────────────────────────────────────────────────────────

def test_reused_article_status_separation():
    """
    A reused article must carry ArticleStatus.REVIEW (internal lifecycle)
    and PublishStatus.DRAFT (WordPress target state) independently.
    """
    article = _article(
        status=ArticleStatus.REVIEW,
        publishing=PublishingOptions(status=PublishStatus.DRAFT),
    )

    assert article.status is ArticleStatus.REVIEW
    assert article.publishing.status is PublishStatus.DRAFT


def test_reused_article_round_trips():
    """
    An article with status=REVIEW and publishing.status=DRAFT must
    serialize to JSON and deserialize back with both values intact.
    This is the exact scenario that triggered the original bug.
    """
    original = _article(
        status=ArticleStatus.REVIEW,
        publishing=PublishingOptions(status=PublishStatus.DRAFT),
    )

    json_bytes = original.model_dump_json()
    restored = Article.model_validate_json(json_bytes)

    assert restored.status is ArticleStatus.REVIEW
    assert restored.publishing.status is PublishStatus.DRAFT


def test_round_trip_preserves_all_status_combinations():
    """Every valid ArticleStatus round-trips correctly regardless of PublishStatus."""
    for article_status in ArticleStatus:
        for publish_status in PublishStatus:
            article = _article(
                status=article_status,
                publishing=PublishingOptions(status=publish_status),
            )
            restored = Article.model_validate_json(article.model_dump_json())
            assert restored.status is article_status
            assert restored.publishing.status is publish_status


# ── Cross-enum assignment must be rejected ─────────────────────────────────────

def test_article_status_rejects_publish_status_value():
    """
    Article.status must reject "draft" — a PublishStatus value that is not
    a member of ArticleStatus. This is the exact value the bug introduced.
    """
    with pytest.raises(ValidationError) as exc_info:
        Article(
            tenant=_tenant(),
            request=_request(),
            title="Test",
            content_markdown="Test",
            seo=_seo(),
            status="draft",  # PublishStatus.DRAFT.value — must be rejected
        )

    errors = exc_info.value.errors()
    assert any(e["loc"] == ("status",) for e in errors), (
        f"Expected a validation error on 'status', got: {errors}"
    )


def test_article_status_rejects_all_publish_status_values():
    """None of the PublishStatus string values are valid ArticleStatus values."""
    for ps in PublishStatus:
        with pytest.raises(ValidationError):
            Article(
                tenant=_tenant(),
                request=_request(),
                title="Test",
                content_markdown="Test",
                seo=_seo(),
                status=ps.value,
            )


def test_publishing_status_rejects_article_status_values():
    """None of the ArticleStatus string values are valid PublishStatus values."""
    for as_ in ArticleStatus:
        with pytest.raises(ValidationError):
            PublishingOptions(status=as_.value)


# ── Enum value sets are disjoint ───────────────────────────────────────────────

def test_enum_value_sets_are_disjoint():
    """
    ArticleStatus and PublishStatus must share no string values.
    A shared value would allow silent cross-assignment to pass Pydantic validation.
    """
    article_values = {s.value for s in ArticleStatus}
    publish_values = {s.value for s in PublishStatus}
    overlap = article_values & publish_values

    assert not overlap, (
        f"ArticleStatus and PublishStatus share values: {overlap}. "
        "Shared values allow silent cross-enum assignment. "
        "Rename or remove the overlapping members."
    )


# ── draft_reuse_service.adapt() contract ──────────────────────────────────────

def test_adapt_sets_correct_status_types():
    """
    DraftReuseService.adapt() must set Article.status to ArticleStatus.REVIEW,
    not PublishStatus.DRAFT. Regression for the original bug.
    """
    from services.draft_reuse_service import DraftMatch, DraftReuseService
    from pathlib import Path

    source_article = _article(status=ArticleStatus.PUBLISHED)
    match = DraftMatch(
        article=source_article,
        similarity=1.0,
        matched_by_topic_id=False,
        same_website=False,
        source_path=Path("/fake/article.json"),
    )

    svc = DraftReuseService(output_dir=Path("/tmp"))
    adapted = svc.adapt(match, request=_request(), tenant=_tenant())

    assert isinstance(adapted.status, ArticleStatus), (
        f"Article.status must be ArticleStatus, got {type(adapted.status)}"
    )
    assert adapted.status is ArticleStatus.REVIEW, (
        f"Adapted article must have status=REVIEW, got {adapted.status}"
    )
    assert isinstance(adapted.publishing.status, PublishStatus), (
        f"publishing.status must be PublishStatus, got {type(adapted.publishing.status)}"
    )


# ── Fix 1 regression: language default must be English ────────────────────────

def test_article_request_language_default_is_english():
    """ArticleRequest() with no explicit language must default to EN, not ES."""
    from models.article import ArticleRequest
    from models.enums import ArticleLanguage
    req = ArticleRequest(topic="Garage door repair")
    assert req.language == ArticleLanguage.EN, (
        f"Expected ArticleLanguage.EN, got {req.language!r}. "
        "The default was changed from ES to EN — verify models/article.py:48."
    )


def test_article_request_language_explicit_override():
    """Explicit language= argument still overrides the default."""
    from models.article import ArticleRequest
    from models.enums import ArticleLanguage
    req = ArticleRequest(topic="Prueba", language=ArticleLanguage.ES)
    assert req.language == ArticleLanguage.ES

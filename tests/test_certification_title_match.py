"""
Regression test: _certify_content must use titles_match, not bool(live_title).

Root cause:
    titles_match was computed correctly but the CertificationItem used
    bool(live_title) as the passed value. A post with any non-empty title
    (even a completely different one) passed the "Title published" check.

Invariant enforced here:
    - A live title that matches the article title → passed=True.
    - A live title that does NOT match → passed=False.
    - No live title → passed=False.
"""
import pytest
from unittest.mock import MagicMock
from services.publication_certification_service import PublicationCertificationService


def _make_article(title="Garage Door Spring Repair in Valparaiso"):
    from uuid import uuid4
    from models.article import Article, ArticleRequest, SEOMetadata
    from models.tenant import TenantContext
    return Article(
        id=uuid4(),
        tenant=TenantContext(client_id="rimc", website_id="overheaddoornwi"),
        request=ArticleRequest(topic=title),
        title=title,
        content_markdown="## Body\n\nContent here.\n",
        seo=SEOMetadata(
            seo_title=title[:60],
            meta_description="Meta description for the article about " + title[:40],
            slug="garage-door-spring-repair-valparaiso",
            focus_keyword="garage door spring repair",
        ),
    )


def _run_content_cert(article, live_title: str | None):
    """Run only the _certify_content section and return its items."""
    from services.publication_certification_service import CertificationReport

    svc = PublicationCertificationService()
    report = CertificationReport(
        article_id=str(article.id),
        wp_post_id=None,
        wp_post_url=None,
    )

    live_post = None
    if live_title is not None:
        live_post = {
            "title": {"rendered": live_title},
            "content": {"rendered": "<p>Body content here for the article.</p>"},
            "slug": "garage-door-spring-repair",
        }

    svc._certify_content(report, article, live_post, min_word_count=100)
    return {item.name: item for item in report.items}


class TestTitleCertification:
    def test_matching_title_passes(self):
        """Live title contains the article title → passed=True."""
        article = _make_article("Garage Door Spring Repair in Valparaiso")
        checks = _run_content_cert(article, "Garage Door Spring Repair in Valparaiso | Overhead Door")
        assert checks["Title published"].passed, (
            "Title published check must pass when the live title contains the article title."
        )

    def test_mismatched_title_fails(self):
        """Live title that does not contain the article title → passed=False."""
        article = _make_article("Garage Door Spring Repair in Valparaiso")
        checks = _run_content_cert(article, "Homepage | Overhead Door NW Indiana")
        assert not checks["Title published"].passed, (
            "Title published check must fail when the live title does not match. "
            "Using bool(live_title) would have incorrectly returned True here."
        )

    def test_missing_live_title_fails(self):
        """No live post → passed=False (not passed=True via bool(live_title))."""
        article = _make_article("Garage Door Spring Repair in Valparaiso")
        checks = _run_content_cert(article, live_title=None)
        assert not checks["Title published"].passed, (
            "Title published check must fail when no live post is available."
        )

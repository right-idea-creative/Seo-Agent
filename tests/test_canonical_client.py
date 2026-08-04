"""
Tests for canonical_client propagation through the pipeline.

Covers:
  - SiteProfile.canonical_client field (present, absent, None)
  - TenantContext.canonical_client field
  - Backwards compatibility: TenantContext without canonical_client in JSON
  - Propagation: SiteProfile → TenantContext → BQ row
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from models.site_profile import SiteProfile
from models.tenant import TenantContext
from services.bq_sink_service import BqSinkService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service() -> BqSinkService:
    svc = object.__new__(BqSinkService)
    svc._client = MagicMock()
    svc._client.insert_rows_json.return_value = []
    svc._init_error = None
    return svc


def _make_site_profile(**kwargs) -> SiteProfile:
    defaults = {
        "client_id": "test-client",
        "website_id": "test-site",
        "business_name": "Test Business",
        "niche": "Garage Door",
        "primary_service": "Garage Door Repair",
        "city": "Denver",
        "state": "CO",
    }
    return SiteProfile(**{**defaults, **kwargs})


def _make_tenant(**kwargs) -> TenantContext:
    defaults = {"client_id": "test-client", "website_id": "test-site"}
    return TenantContext(**{**defaults, **kwargs})


def _make_article(tenant: TenantContext) -> MagicMock:
    art = MagicMock()
    art.id = "test-uuid-1234"
    art.tenant = tenant
    art.request.topic = "Test topic"
    art.title = "Test Title"
    art.seo.slug = "test-title"
    art.seo.focus_keyword = "test keyword"
    art.seo.suggested_category = None
    art.wp_post_url = "https://example.com/test"
    art.word_count = 800
    art.reading_time_minutes = 4
    art.model_name = "claude-sonnet-4-6"
    art.prompt_version = "1.0"
    return art


# ── SiteProfile.canonical_client ──────────────────────────────────────────────

class TestSiteProfileCanonicalClient:
    def test_canonical_client_accepts_valid_slug(self):
        profile = _make_site_profile(canonical_client="overhead-door-network")
        assert profile.canonical_client == "overhead-door-network"

    def test_canonical_client_defaults_to_none(self):
        profile = _make_site_profile()
        assert profile.canonical_client is None

    def test_canonical_client_none_explicit(self):
        profile = _make_site_profile(canonical_client=None)
        assert profile.canonical_client is None

    def test_canonical_client_round_trips_json(self):
        profile = _make_site_profile(canonical_client="test-corp")
        dumped = profile.model_dump_json()
        loaded = SiteProfile.model_validate_json(dumped)
        assert loaded.canonical_client == "test-corp"

    def test_canonical_client_absent_in_json_produces_none(self):
        data = {
            "client_id": "c", "website_id": "w",
            "business_name": "B", "niche": "N",
            "primary_service": "P", "city": "Denver", "state": "CO",
        }
        profile = SiteProfile.model_validate(data)
        assert profile.canonical_client is None


# ── TenantContext.canonical_client ────────────────────────────────────────────

class TestTenantContextCanonicalClient:
    def test_canonical_client_stored_on_tenant(self):
        tenant = _make_tenant(canonical_client="overhead-door-network")
        assert tenant.canonical_client == "overhead-door-network"

    def test_canonical_client_defaults_to_none(self):
        tenant = _make_tenant()
        assert tenant.canonical_client is None

    def test_canonical_client_model_copy_immutable(self):
        tenant = _make_tenant()
        updated = tenant.model_copy(update={"canonical_client": "new-corp"})
        assert updated.canonical_client == "new-corp"
        assert tenant.canonical_client is None  # original unchanged

    def test_canonical_client_round_trips_json(self):
        tenant = _make_tenant(canonical_client="test-corp")
        dumped = tenant.model_dump_json()
        loaded = TenantContext.model_validate_json(dumped)
        assert loaded.canonical_client == "test-corp"


# ── Backwards compatibility ───────────────────────────────────────────────────

class TestCanonicalClientBackwardsCompat:
    def test_legacy_tenant_json_without_field_loads_as_none(self):
        """article.json written before Request #5 has no canonical_client key."""
        legacy_json = json.dumps({
            "client_id": "legacy-client",
            "website_id": "legacy-site",
        })
        tenant = TenantContext.model_validate_json(legacy_json)
        assert tenant.canonical_client is None

    def test_legacy_tenant_dict_without_field_loads_as_none(self):
        legacy = {"client_id": "legacy-client", "website_id": "legacy-site"}
        tenant = TenantContext.model_validate(legacy)
        assert tenant.canonical_client is None

    def test_tenant_with_null_canonical_client_loads_as_none(self):
        data = {"client_id": "c", "website_id": "w", "canonical_client": None}
        tenant = TenantContext.model_validate(data)
        assert tenant.canonical_client is None

    def test_other_tenant_fields_unaffected(self):
        """Adding canonical_client must not break existing TenantContext fields."""
        tenant = _make_tenant(dealer_id="dealer-1", reuse_group="group-a",
                               canonical_client="corp-x")
        assert tenant.client_id == "test-client"
        assert tenant.website_id == "test-site"
        assert tenant.dealer_id == "dealer-1"
        assert tenant.reuse_group == "group-a"
        assert tenant.canonical_client == "corp-x"


# ── Propagation: SiteProfile → TenantContext → BQ row ────────────────────────

class TestCanonicalClientPropagation:
    def test_profile_canonical_client_written_to_bq_article_row(self):
        profile = _make_site_profile(canonical_client="overhead-door-network")
        tenant = _make_tenant()
        tenant = tenant.model_copy(update={"canonical_client": profile.canonical_client})
        article = _make_article(tenant)

        svc = _make_service()
        svc.insert_article(article, None, None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] == "overhead-door-network"

    def test_profile_canonical_client_written_to_bq_qa_row(self):
        canonical = "overhead-door-network"
        svc = _make_service()

        qa_report = MagicMock()
        qa_report.article_passed = True
        qa_report.passed = True
        qa_report.iterations_used = 1
        final = MagicMock()
        final.seo_score = 90
        final.editorial_score = 88
        final.writing_score = 85
        final.authenticity_score = 86
        qa_report.final_article_review = final

        svc.insert_qa_results("some-id", qa_report, canonical_client=canonical)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] == "overhead-door-network"

    def test_profile_canonical_client_written_to_bq_cost_rows(self):
        canonical = "overhead-door-network"
        svc = _make_service()

        tracer = MagicMock()
        from tests.test_bq_sink_service import _FakeRecord
        tracer.records = [
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
        ]

        svc.insert_llm_costs("seo-agent", tracer, canonical_client=canonical)
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 2
        assert all(r["canonical_client"] == "overhead-door-network" for r in rows)

    def test_none_canonical_client_propagates_as_none(self):
        profile = _make_site_profile()  # no canonical_client
        tenant = _make_tenant()
        if profile.canonical_client:
            tenant = tenant.model_copy(update={"canonical_client": profile.canonical_client})

        article = _make_article(tenant)
        svc = _make_service()
        svc.insert_article(article, None, None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] is None

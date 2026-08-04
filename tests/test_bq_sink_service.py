"""
Regression tests for BqSinkService — BigQuery sink.

Invariants:
  - All three insert methods are fire-and-forget: they never raise.
  - When GOOGLE_APPLICATION_CREDENTIALS is absent, all inserts are no-ops.
  - insert_article() sends one row with the correct fields.
  - insert_qa_results() sends one row with the correct QA fields.
  - insert_llm_costs() sends one row per CallRecord.
  - Network errors, auth errors, and missing-dataset errors are all absorbed.
  - Publishing continues normally after any BigQuery failure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from services.bq_sink_service import (
    BqSinkService,
    _TABLE_ARTICLES,
    _TABLE_COSTS,
    _TABLE_QA,
    _provider_from_model,
    _create_bq_client,
    _ENVIRONMENT,
    _GIT_COMMIT,
    _PIPELINE_VERSION,
)


# ── Helpers — minimal model stubs ────────────────────────────────────────────

def _make_service() -> BqSinkService:
    """BqSinkService with a mock BigQuery client — bypasses credential check."""
    svc = object.__new__(BqSinkService)
    svc._client = MagicMock()
    svc._client.insert_rows_json.return_value = []  # [] = no errors
    svc._init_error = None
    return svc


def _no_creds_service() -> BqSinkService:
    """BqSinkService as if GOOGLE_APPLICATION_CREDENTIALS was not set."""
    svc = object.__new__(BqSinkService)
    svc._client = None
    svc._init_error = "GOOGLE_APPLICATION_CREDENTIALS is not set"
    return svc


def _make_article(
    *,
    wp_post_url: str | None = "https://example.com/post/1",
    word_count: int = 800,
    reading_time_minutes: int = 4,
) -> MagicMock:
    art = MagicMock()
    art.id = uuid4()
    art.tenant.client_id = "test-client"
    art.tenant.website_id = "test-site"
    art.tenant.canonical_client = "test-canonical"
    art.request.topic = "Garage door spring repair"
    art.title = "Garage Door Spring Repair Guide"
    art.seo.slug = "garage-door-spring-repair"
    art.seo.focus_keyword = "garage door spring repair"
    art.seo.suggested_category = "Repair"
    art.wp_post_url = wp_post_url
    art.word_count = word_count
    art.reading_time_minutes = reading_time_minutes
    art.model_name = "claude-sonnet-4-6"
    art.prompt_version = "1.0"
    return art


def _make_qa_report(
    *,
    seo: int = 95,
    editorial: int = 93,
    writing: int = 91,
    authenticity: int = 92,
    approved: bool = True,
    iterations: int = 1,
) -> MagicMock:
    rpt = MagicMock()
    rpt.article_passed = approved
    rpt.passed = approved
    rpt.iterations_used = iterations

    final = MagicMock()
    final.seo_score = seo
    final.editorial_score = editorial
    final.writing_score = writing
    final.authenticity_score = authenticity
    rpt.final_article_review = final
    return rpt


@dataclass
class _FakeRecord:
    stage: str
    model: str
    input_tokens: int = 1000
    output_tokens: int = 500
    duration_s: float = 1.0
    used: bool = True
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        # Simplified: $3/M in, $15/M out (Sonnet pricing)
        return self.input_tokens / 1_000_000 * 3.0 + self.output_tokens / 1_000_000 * 15.0


def _make_tracer(*records: _FakeRecord) -> MagicMock:
    tracer = MagicMock()
    tracer.records = list(records)
    tracer.total_cost.return_value = sum(r.cost_usd for r in records)
    return tracer


# ── _provider_from_model ──────────────────────────────────────────────────────

class TestProviderFromModel:
    def test_claude_models(self):
        assert _provider_from_model("claude-sonnet-4-6") == "claude"
        assert _provider_from_model("claude-opus-4-8") == "claude"
        assert _provider_from_model("claude-haiku-4-5") == "claude"

    def test_openai_models(self):
        assert _provider_from_model("gpt-4o") == "openai"
        assert _provider_from_model("gpt-4o-mini") == "openai"
        assert _provider_from_model("o1-preview") == "openai"

    def test_unknown_model(self):
        assert _provider_from_model("llama-3") == "other"
        assert _provider_from_model("gemini-pro") == "other"


# ── Missing credentials → no-op ───────────────────────────────────────────────

class TestMissingCredentials:
    def test_client_is_none_when_env_var_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            svc = BqSinkService()
        assert svc._client is None

    def test_insert_article_is_noop_when_no_client(self):
        svc = _no_creds_service()
        article = _make_article()
        qa_report = _make_qa_report()
        svc.insert_article(article, qa_report, None, 10.0)  # must not raise

    def test_insert_qa_results_is_noop_when_no_client(self):
        svc = _no_creds_service()
        qa_report = _make_qa_report()
        svc.insert_qa_results("some-id", qa_report)  # must not raise

    def test_insert_llm_costs_is_noop_when_no_client(self):
        svc = _no_creds_service()
        rec = _FakeRecord(stage="draft:write", model="claude-sonnet-4-6")
        tracer = _make_tracer(rec)
        svc.insert_llm_costs("seo-agent", tracer)  # must not raise


# ── Authentication failure during init ────────────────────────────────────────

class TestAuthFailureDuringInit:
    def test_client_remains_none_when_init_raises(self):
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/creds.json"}):
            with patch(
                "services.bq_sink_service._create_bq_client",
                side_effect=Exception("auth error"),
            ):
                svc = BqSinkService()
        assert svc._client is None

    def test_inserts_are_noop_after_failed_init(self):
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/creds.json"}):
            with patch(
                "services.bq_sink_service._create_bq_client",
                side_effect=Exception("invalid credentials"),
            ):
                svc = BqSinkService()

        qa_report = _make_qa_report()
        svc.insert_article(_make_article(), qa_report, None, 5.0)
        svc.insert_qa_results("id-123", qa_report)
        svc.insert_llm_costs("seo-agent", _make_tracer())


# ── insert_article ────────────────────────────────────────────────────────────

class TestInsertArticle:
    def test_calls_insert_rows_json_with_correct_table(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 12.5)
        table_arg = svc._client.insert_rows_json.call_args[0][0]
        assert table_arg == _TABLE_ARTICLES

    def test_row_contains_expected_fields(self):
        svc = _make_service()
        article = _make_article(word_count=900, reading_time_minutes=5)
        qa = _make_qa_report(seo=96, editorial=94, writing=92, authenticity=91)
        svc.insert_article(article, qa, None, 15.0)

        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 1
        row = rows[0]
        assert row["client"] == "test-client"
        assert row["website"] == "test-site"
        assert row["canonical_client"] == "test-canonical"
        assert row["topic"] == "Garage door spring repair"
        assert row["word_count"] == 900
        assert row["reading_time"] == 5
        assert row["seo_score"] == 96
        assert row["editorial_score"] == 94
        assert row["writing_score"] == 92
        assert row["authenticity_score"] == 91
        assert row["generation_time"] == 15.0

    def test_cost_fields_split_by_provider(self):
        svc = _make_service()
        rec_claude = _FakeRecord("draft:write", "claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        rec_openai = _FakeRecord("qa:review", "gpt-4o-mini", input_tokens=500, output_tokens=200)
        tracer = _make_tracer(rec_claude, rec_openai)

        svc.insert_article(_make_article(), _make_qa_report(), tracer, 8.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]

        assert row["claude_cost_usd"] == round(rec_claude.cost_usd, 6)
        assert row["openai_cost_usd"] == round(rec_openai.cost_usd, 6)
        assert abs(row["total_cost_usd"] - (rec_claude.cost_usd + rec_openai.cost_usd)) < 1e-9

    def test_reuse_fields_propagated(self):
        svc = _make_service()
        svc.insert_article(
            _make_article(), _make_qa_report(), None, 1.0,
            reuse=True, reuse_similarity=0.85,
        )
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["reuse"] is True
        assert row["reuse_similarity"] == 0.85

    def test_missing_wp_post_url_sends_none(self):
        svc = _make_service()
        svc.insert_article(_make_article(wp_post_url=None), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["url"] is None

    def test_network_failure_does_not_raise(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = ConnectionError("timeout")
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)  # must not raise

    def test_publishing_continues_after_article_insert_fails(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = Exception("BigQuery unavailable")

        result = "published"  # simulate downstream publish step
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        assert result == "published"


# ── insert_qa_results ─────────────────────────────────────────────────────────

class TestInsertQaResults:
    def test_calls_correct_table(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        table_arg = svc._client.insert_rows_json.call_args[0][0]
        assert table_arg == _TABLE_QA

    def test_row_contains_expected_fields(self):
        svc = _make_service()
        qa = _make_qa_report(seo=95, editorial=93, writing=91, authenticity=90, iterations=2)
        qa.article_passed = True
        qa.passed = True
        article_id = str(uuid4())
        svc.insert_qa_results(article_id, qa)

        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 1
        row = rows[0]
        assert row["article_id"] == article_id
        assert row["approved"] is True
        assert row["revision_cycles"] == 2
        assert row["claude_seo_score"] == 95
        assert row["claude_editorial_score"] == 93
        assert row["openai_writing_score"] == 91
        assert row["openai_authenticity_score"] == 90
        assert row["overall_pass"] is True

    def test_network_failure_does_not_raise(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = OSError("connection reset")
        svc.insert_qa_results("id-456", _make_qa_report())  # must not raise

    def test_missing_dataset_error_does_not_raise(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = Exception("404 Not Found: dataset not found")
        svc.insert_qa_results("id-789", _make_qa_report())  # must not raise


# ── insert_llm_costs ──────────────────────────────────────────────────────────

class TestInsertLlmCosts:
    def test_calls_correct_table(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        table_arg = svc._client.insert_rows_json.call_args[0][0]
        assert table_arg == _TABLE_COSTS

    def test_one_row_per_call_record(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
            _FakeRecord("image:plan", "claude-haiku-4-5"),
        )
        svc.insert_llm_costs("seo-agent", tracer)
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 3

    def test_row_fields_populated(self):
        svc = _make_service()
        rec = _FakeRecord(
            "draft:write", "claude-opus-4-8",
            input_tokens=2000, output_tokens=1000,
            used=True,
        )
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["system"] == "seo-agent"
        assert row["stage"] == "draft:write"
        assert row["provider"] == "claude"
        assert row["model"] == "claude-opus-4-8"
        assert row["input_tokens"] == 2000
        assert row["output_tokens"] == 1000
        assert row["success"] is True
        assert row["cost_usd"] > 0

    def test_provider_derived_from_model(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("qa:review", "gpt-4o"),
            _FakeRecord("qa:vision", "gpt-4o-mini"),
        )
        svc.insert_llm_costs("seo-agent", tracer)
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert all(r["provider"] == "openai" for r in rows)

    def test_empty_tracer_sends_no_rows(self):
        svc = _make_service()
        svc.insert_llm_costs("seo-agent", _make_tracer())
        svc._client.insert_rows_json.assert_not_called()

    def test_network_failure_does_not_raise(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = TimeoutError("BigQuery timeout")
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))  # must not raise

    def test_publishing_continues_after_costs_insert_fails(self):
        svc = _make_service()
        svc._client.insert_rows_json.side_effect = Exception("503 Service Unavailable")

        publish_completed = False
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        publish_completed = True  # reached only if insert did not raise

        assert publish_completed


# ── Multiple inserts in sequence ──────────────────────────────────────────────

class TestMultipleInserts:
    def test_all_three_inserts_succeed_in_sequence(self):
        svc = _make_service()
        article = _make_article()
        qa = _make_qa_report()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        tracer = _make_tracer(rec)

        svc.insert_article(article, qa, tracer, 9.0)
        svc.insert_qa_results(str(article.id), qa)
        svc.insert_llm_costs("seo-agent", tracer)

        assert svc._client.insert_rows_json.call_count == 3

    def test_partial_failure_does_not_prevent_remaining_inserts(self):
        svc = _make_service()
        call_count = [0]

        def _side_effect(table, rows):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("first call fails")
            return []  # subsequent calls succeed

        svc._client.insert_rows_json.side_effect = _side_effect

        article = _make_article()
        qa = _make_qa_report()
        tracer = _make_tracer(_FakeRecord("draft:write", "claude-sonnet-4-6"))

        svc.insert_article(article, qa, tracer, 9.0)   # fails internally, no raise
        svc.insert_qa_results(str(article.id), qa)     # succeeds
        svc.insert_llm_costs("seo-agent", tracer)      # succeeds

        assert svc._client.insert_rows_json.call_count == 3


# ── insert_article — new traceability fields ──────────────────────────────────

class TestInsertArticleNewFields:
    def test_article_id_written_to_row(self):
        svc = _make_service()
        article = _make_article()
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["article_id"] == str(article.id)

    def test_article_id_is_uuid_string(self):
        svc = _make_service()
        article = _make_article()
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        # Must be a string that parses as a UUID
        parsed = UUID(row["article_id"])
        assert str(parsed) == row["article_id"]

    def test_model_name_written_to_row(self):
        svc = _make_service()
        article = _make_article()
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["model_name"] == "claude-sonnet-4-6"

    def test_prompt_version_written_to_row(self):
        svc = _make_service()
        article = _make_article()
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["prompt_version"] == "1.0"


# ── insert_article — None qa_report (no-QA publish path) ─────────────────────

class TestInsertArticleNoneQaReport:
    def test_none_qa_report_does_not_raise(self):
        svc = _make_service()
        svc.insert_article(_make_article(), None, None, 5.0)  # must not raise

    def test_none_qa_report_scores_are_zero(self):
        svc = _make_service()
        svc.insert_article(_make_article(), None, None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["seo_score"] == 0
        assert row["editorial_score"] == 0
        assert row["writing_score"] == 0
        assert row["authenticity_score"] == 0

    def test_none_qa_report_still_writes_article_id(self):
        svc = _make_service()
        article = _make_article()
        svc.insert_article(article, None, None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["article_id"] == str(article.id)


# ── insert_article — NULL for absent optional strings ─────────────────────────

class TestInsertArticleNullHandling:
    def test_empty_focus_keyword_sends_none(self):
        svc = _make_service()
        article = _make_article()
        article.seo.focus_keyword = ""
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["focus_keyword"] is None

    def test_nonempty_focus_keyword_is_preserved(self):
        svc = _make_service()
        article = _make_article()
        article.seo.focus_keyword = "garage door spring repair"
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["focus_keyword"] == "garage door spring repair"

    def test_empty_category_sends_none(self):
        svc = _make_service()
        article = _make_article()
        article.seo.suggested_category = ""
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["category"] is None

    def test_none_category_sends_none(self):
        svc = _make_service()
        article = _make_article()
        article.seo.suggested_category = None
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["category"] is None

    def test_nonempty_category_is_preserved(self):
        svc = _make_service()
        article = _make_article()
        article.seo.suggested_category = "Repair"
        svc.insert_article(article, _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["category"] == "Repair"


# ── insert_llm_costs — article_id propagation ─────────────────────────────────

class TestInsertLlmCostsArticleId:
    def test_article_id_written_to_all_rows(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
        )
        svc.insert_llm_costs("seo-agent", tracer, article_id="abc-uuid-1234")
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 2
        assert all(r["article_id"] == "abc-uuid-1234" for r in rows)

    def test_article_id_defaults_to_none_when_omitted(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["article_id"] is None

    def test_article_id_none_when_passed_explicitly(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec), article_id=None)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["article_id"] is None

    def test_article_id_present_alongside_other_fields(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-opus-4-8", input_tokens=2000, output_tokens=1000)
        svc.insert_llm_costs("seo-agent", _make_tracer(rec), article_id="test-id")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["article_id"] == "test-id"
        assert row["system"] == "seo-agent"
        assert row["stage"] == "draft:write"
        assert row["model"] == "claude-opus-4-8"
        assert row["cost_usd"] > 0

    def test_empty_tracer_with_article_id_sends_no_rows(self):
        svc = _make_service()
        svc.insert_llm_costs("seo-agent", _make_tracer(), article_id="some-id")
        svc._client.insert_rows_json.assert_not_called()


# ── insert_article — execution metadata ───────────────────────────────────────

class TestInsertArticleExecutionMetadata:
    def test_event_type_default_is_publish(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "publish"

    def test_event_type_autopublish(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0,
                           event_type="autopublish")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "autopublish"

    def test_event_type_republish(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0,
                           event_type="republish")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "republish"

    def test_environment_reads_module_constant(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["environment"] == _ENVIRONMENT

    def test_environment_patchable(self):
        with patch("services.bq_sink_service._ENVIRONMENT", "staging"):
            svc = _make_service()
            svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["environment"] == "staging"

    def test_git_commit_key_present(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert "git_commit" in row

    def test_git_commit_value_matches_module_constant(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["git_commit"] == _GIT_COMMIT

    def test_git_commit_patchable(self):
        with patch("services.bq_sink_service._GIT_COMMIT", "abc123def456"):
            svc = _make_service()
            svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["git_commit"] == "abc123def456"

    def test_pipeline_version_key_present(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert "pipeline_version" in row

    def test_pipeline_version_value_matches_module_constant(self):
        svc = _make_service()
        svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["pipeline_version"] == _PIPELINE_VERSION

    def test_pipeline_version_patchable(self):
        with patch("services.bq_sink_service._PIPELINE_VERSION", "2.1.0"):
            svc = _make_service()
            svc.insert_article(_make_article(), _make_qa_report(), None, 5.0)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["pipeline_version"] == "2.1.0"


# ── insert_qa_results — execution metadata ────────────────────────────────────

class TestInsertQaResultsExecutionMetadata:
    def test_environment_key_present(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert "environment" in row

    def test_environment_reads_module_constant(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["environment"] == _ENVIRONMENT

    def test_environment_patchable(self):
        with patch("services.bq_sink_service._ENVIRONMENT", "dev"):
            svc = _make_service()
            svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["environment"] == "dev"

    def test_git_commit_key_present(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert "git_commit" in row

    def test_git_commit_value_matches_module_constant(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["git_commit"] == _GIT_COMMIT


# ── insert_llm_costs — execution metadata ────────────────────────────────────

class TestInsertLlmCostsExecutionMetadata:
    def test_event_type_default_is_publish(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "publish"

    def test_event_type_autopublish(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec), event_type="autopublish")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "autopublish"

    def test_event_type_propagated_to_all_rows(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
            _FakeRecord("image:plan", "claude-haiku-4-5"),
        )
        svc.insert_llm_costs("seo-agent", tracer, event_type="republish")
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 3
        assert all(r["event_type"] == "republish" for r in rows)

    def test_environment_propagated_to_all_rows(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
        )
        svc.insert_llm_costs("seo-agent", tracer)
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert all(r["environment"] == _ENVIRONMENT for r in rows)

    def test_environment_patchable(self):
        with patch("services.bq_sink_service._ENVIRONMENT", "prod"):
            svc = _make_service()
            rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
            svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["environment"] == "prod"

    def test_git_commit_propagated_to_all_rows(self):
        with patch("services.bq_sink_service._GIT_COMMIT", "deadbeef12345"):
            svc = _make_service()
            tracer = _make_tracer(
                _FakeRecord("draft:write", "claude-sonnet-4-6"),
                _FakeRecord("qa:review", "gpt-4o-mini"),
            )
            svc.insert_llm_costs("seo-agent", tracer)
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert all(r["git_commit"] == "deadbeef12345" for r in rows)

    def test_all_metadata_fields_present_in_row(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec), event_type="autopublish",
                              article_id="test-id")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["event_type"] == "autopublish"
        assert row["environment"] == _ENVIRONMENT
        assert "git_commit" in row
        assert row["article_id"] == "test-id"
        assert row["system"] == "seo-agent"


# ── insert_qa_results — canonical_client ─────────────────────────────────────

class TestInsertQaResultsCanonicalClient:
    def test_canonical_client_written_when_passed(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report(),
                               canonical_client="overhead-door-network")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] == "overhead-door-network"

    def test_canonical_client_defaults_to_none_when_omitted(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report())
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] is None

    def test_canonical_client_none_when_passed_explicitly(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report(), canonical_client=None)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] is None

    def test_canonical_client_present_alongside_core_qa_fields(self):
        svc = _make_service()
        svc.insert_qa_results("abc-123", _make_qa_report(seo=88, approved=True),
                               canonical_client="test-corp")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] == "test-corp"
        assert row["article_id"] == "abc-123"
        assert row["approved"] is True
        assert row["claude_seo_score"] == 88


# ── insert_llm_costs — canonical_client ──────────────────────────────────────

class TestInsertLlmCostsCanonicalClient:
    def test_canonical_client_written_to_all_rows(self):
        svc = _make_service()
        tracer = _make_tracer(
            _FakeRecord("draft:write", "claude-sonnet-4-6"),
            _FakeRecord("qa:review", "gpt-4o-mini"),
        )
        svc.insert_llm_costs("seo-agent", tracer,
                              canonical_client="overhead-door-network")
        rows = svc._client.insert_rows_json.call_args[0][1]
        assert len(rows) == 2
        assert all(r["canonical_client"] == "overhead-door-network" for r in rows)

    def test_canonical_client_defaults_to_none_when_omitted(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec))
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] is None

    def test_canonical_client_none_when_passed_explicitly(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec), canonical_client=None)
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] is None

    def test_canonical_client_alongside_article_id(self):
        svc = _make_service()
        rec = _FakeRecord("draft:write", "claude-sonnet-4-6")
        svc.insert_llm_costs("seo-agent", _make_tracer(rec),
                              article_id="uuid-xyz", canonical_client="test-corp")
        row = svc._client.insert_rows_json.call_args[0][1][0]
        assert row["canonical_client"] == "test-corp"
        assert row["article_id"] == "uuid-xyz"

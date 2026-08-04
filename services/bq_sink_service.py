"""
BqSinkService — fire-and-forget BigQuery sink for the SEO-Agent pipeline.

Writes three tables in the rightidea-cortex.seo_content dataset:
  articles_published  one row per published article
  qa_results          one row per QA run
  llm_costs           one row per LLM call record

Design constraints:
  - All inserts are best-effort: every exception is swallowed and logged.
  - BigQuery failures never interrupt the publishing pipeline.
  - Credentials come exclusively from GOOGLE_APPLICATION_CREDENTIALS.
  - The BigQuery client is initialized lazily on construction; missing or
    invalid credentials produce a warning log and disable the sink — no crash.
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.article import Article
    from models.qa_report import DualQAReport
    from services.call_tracer import CallTracer

logger = logging.getLogger(__name__)

_GCP_PROJECT = "rightidea-cortex"
_BQ_DATASET = "seo_content"

_TABLE_ARTICLES = f"{_GCP_PROJECT}.{_BQ_DATASET}.articles_published"
_TABLE_QA = f"{_GCP_PROJECT}.{_BQ_DATASET}.qa_results"
_TABLE_COSTS = f"{_GCP_PROJECT}.{_BQ_DATASET}.llm_costs"

# ── Execution metadata — captured once at module import ───────────────────────
# environment: override via SEO_AGENT_ENV env var. Default is "prod" so a
# misconfigured production deployment produces prod-tagged rows, not dev noise.
_ENVIRONMENT: str = os.environ.get("SEO_AGENT_ENV", "prod")

# pipeline_version: override via PIPELINE_VERSION env var (e.g. set by CI).
# Remains None until the team adopts a formal versioning convention.
_PIPELINE_VERSION: str | None = os.environ.get("PIPELINE_VERSION")


def _get_git_commit() -> str | None:
    """Return the current HEAD SHA, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


_GIT_COMMIT: str | None = _get_git_commit()


def _create_bq_client(project: str):
    from google.cloud import bigquery  # type: ignore[import]
    return bigquery.Client(project=project)


def _provider_from_model(model: str) -> str:
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    return "other"


class BqSinkService:
    """
    Additive BigQuery sink.  All public methods are fire-and-forget:
    they catch every exception, log a warning, and return None.
    """

    def __init__(self) -> None:
        self._client = None
        self._init_error: str | None = None
        self._try_init()

    def _try_init(self) -> None:
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            self._init_error = (
                "GOOGLE_APPLICATION_CREDENTIALS is not set — BigQuery sink disabled"
            )
            logger.warning(self._init_error)
            return
        try:
            self._client = _create_bq_client(_GCP_PROJECT)
        except Exception as exc:
            self._init_error = f"BigQuery client init failed: {exc}"
            logger.warning(self._init_error)

    def _insert(self, table_id: str, rows: list[dict]) -> None:
        if self._client is None or not rows:
            return
        try:
            errors = self._client.insert_rows_json(table_id, rows)
            if errors:
                logger.warning(
                    "BigQuery streaming insert errors for %s: %s", table_id, errors
                )
        except Exception as exc:
            logger.warning("BigQuery insert failed for %s: %s", table_id, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def insert_article(
        self,
        article: "Article",
        qa_report: "DualQAReport | None",
        call_tracer: "CallTracer | None",
        generation_time_s: float,
        *,
        reuse: bool = False,
        reuse_similarity: float = 0.0,
        publish_date: datetime | None = None,
        event_type: str = "publish",
    ) -> None:
        """Insert one row into articles_published. Never raises."""
        if self._client is None:
            return
        try:
            final = qa_report.final_article_review if qa_report is not None else None

            total_cost = claude_cost = openai_cost = 0.0
            if call_tracer:
                for rec in call_tracer.records:
                    c = rec.cost_usd
                    total_cost += c
                    if rec.model.startswith("claude"):
                        claude_cost += c
                    else:
                        openai_cost += c

            row = {
                "article_id": str(article.id),
                "client": article.tenant.client_id,
                "website": article.tenant.website_id,
                "canonical_client": article.tenant.canonical_client,
                "topic": (article.request.topic if article.request else "") or "",
                "title": article.title or "",
                "slug": (article.seo.slug if article.seo else "") or "",
                "url": article.wp_post_url or None,
                "publish_date": (
                    publish_date or datetime.now(tz=timezone.utc)
                ).isoformat(),
                "word_count": article.word_count or 0,
                "reading_time": article.reading_time_minutes or 0,
                "focus_keyword": (
                    (article.seo.focus_keyword if article.seo else "") or None
                ),
                "category": (
                    (article.seo.suggested_category if article.seo else None) or None
                ),
                "seo_score": final.seo_score if final else 0,
                "editorial_score": final.editorial_score if final else 0,
                "writing_score": final.writing_score if final else 0,
                "authenticity_score": final.authenticity_score if final else 0,
                "total_cost_usd": round(total_cost, 6),
                "claude_cost_usd": round(claude_cost, 6),
                "openai_cost_usd": round(openai_cost, 6),
                "reuse": reuse,
                "reuse_similarity": round(reuse_similarity, 4),
                "generation_time": round(generation_time_s, 2),
                "model_name": article.model_name,
                "prompt_version": article.prompt_version,
                "event_type": event_type,
                "environment": _ENVIRONMENT,
                "git_commit": _GIT_COMMIT,
                "pipeline_version": _PIPELINE_VERSION,
            }
            self._insert(_TABLE_ARTICLES, [row])
        except Exception as exc:
            logger.warning("BqSinkService.insert_article failed: %s", exc)

    def insert_qa_results(
        self,
        article_id: str,
        qa_report: "DualQAReport",
        *,
        canonical_client: str | None = None,
    ) -> None:
        """Insert one row into qa_results. Never raises."""
        if self._client is None:
            return
        try:
            final = qa_report.final_article_review
            row = {
                "article_id": str(article_id),
                "canonical_client": canonical_client,
                "approved": qa_report.article_passed,
                "revision_cycles": qa_report.iterations_used,
                "claude_seo_score": final.seo_score if final else 0,
                "claude_editorial_score": final.editorial_score if final else 0,
                "openai_writing_score": final.writing_score if final else 0,
                "openai_authenticity_score": final.authenticity_score if final else 0,
                "overall_pass": qa_report.passed,
                "environment": _ENVIRONMENT,
                "git_commit": _GIT_COMMIT,
            }
            self._insert(_TABLE_QA, [row])
        except Exception as exc:
            logger.warning("BqSinkService.insert_qa_results failed: %s", exc)

    def insert_llm_costs(
        self,
        system_label: str,
        call_tracer: "CallTracer",
        *,
        article_id: str | None = None,
        event_type: str = "publish",
        canonical_client: str | None = None,
    ) -> None:
        """Insert one row per CallRecord into llm_costs. Never raises."""
        if self._client is None:
            return
        try:
            now = datetime.now(tz=timezone.utc).isoformat()
            rows = [
                {
                    "timestamp": now,
                    "article_id": article_id,
                    "canonical_client": canonical_client,
                    "event_type": event_type,
                    "environment": _ENVIRONMENT,
                    "git_commit": _GIT_COMMIT,
                    "system": system_label,
                    "stage": rec.stage,
                    "provider": _provider_from_model(rec.model),
                    "model": rec.model,
                    "input_tokens": rec.input_tokens,
                    "output_tokens": rec.output_tokens,
                    "cost_usd": round(rec.cost_usd, 6),
                    "success": rec.used,
                }
                for rec in call_tracer.records
            ]
            if rows:
                self._insert(_TABLE_COSTS, rows)
        except Exception as exc:
            logger.warning("BqSinkService.insert_llm_costs failed: %s", exc)

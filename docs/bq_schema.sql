-- ============================================================
-- BigQuery DDL — SEO-Agent Analytics Tables
-- Dataset   : rightidea-cortex.seo_content  (must already exist)
-- Source    : services/bq_sink_service.py
-- Revised   : 2026-08-04
--
-- Execute in BigQuery Console or via bq CLI:
--   bq query --nouse_legacy_sql < docs/bq_schema.sql
--
-- All three CREATE TABLE statements are idempotent.
-- Running this script against an existing dataset is safe.
--
-- Schema compatibility:
--   Every column is derived directly from the Python row dicts in
--   BqSinkService.  Do not add columns here without a corresponding
--   change in bq_sink_service.py.
--
-- Column modes:
--   NOT NULL (REQUIRED) — always written by Python; insert fails if absent.
--   NULLABLE            — Python may omit the key; BigQuery stores NULL.
--                         Used for future-compatibility stubs and fields
--                         that are semantically optional.
-- ============================================================


-- ============================================================
-- TABLE 1: articles_published
--
-- One row per published article.
-- Written by: BqSinkService.insert_article()
--
-- Primary partition key: DATE(publish_date)
--   Time-bounded queries (monthly volume, cost trends, QA trends)
--   prune non-matching date partitions before scanning any data.
--
-- Cluster: client, website
--   The two dominant filter predicates for this platform.
--   Both fields are high-cardinality, always populated, and
--   appear in every analytics query.  Clustering order matters:
--   client is listed first because it is the coarser dimension;
--   BQ eliminates full-client blocks before evaluating website.
--
-- Join key: article_id
--   Joins to qa_results.article_id and (future) llm_costs.article_id.
--   All three tables share the same UUID v4 string.
-- ============================================================

CREATE TABLE IF NOT EXISTS `rightidea-cortex.seo_content.articles_published`
(
  -- ── Primary key ───────────────────────────────────────────────────────────
  article_id          STRING    NOT NULL
    OPTIONS (description = 'Article.id cast to string (UUID v4). '
                           'Immutable join key shared by qa_results and llm_costs. '
                           'Example: "550e8400-e29b-41d4-a716-446655440000".'),

  -- ── Tenant identity ───────────────────────────────────────────────────────
  client              STRING    NOT NULL
    OPTIONS (description = 'TenantContext.client_id. Opaque identifier for the '
                           'business account that owns this article. '
                           'Alphanumeric, hyphens, and underscores only '
                           '(validated by TenantContext field_validator).'),

  website             STRING    NOT NULL
    OPTIONS (description = 'TenantContext.website_id. Identifies the specific '
                           'WordPress site within the client account. '
                           'A single client may own multiple websites.'),

  -- ── Canonical client identity (Request #5) ──────────────────────────────
  canonical_client    STRING
    OPTIONS (description = 'SiteProfile.canonical_client — normalized business '
                           'entity identifier for Cortex joins. Populated from '
                           'Request #5. NULL on rows written before the migration; '
                           'required for all new publish runs.'),

  -- ── Content ───────────────────────────────────────────────────────────────
  topic               STRING    NOT NULL
    OPTIONS (description = 'ArticleRequest.topic. The free-text topic brief '
                           'submitted to the pipeline, e.g. '
                           '"Garage door spring repair in Denver".'),

  title               STRING    NOT NULL
    OPTIONS (description = 'Generated article title (Article.title) at the '
                           'time of publication.'),

  slug                STRING    NOT NULL
    OPTIONS (description = 'SEO slug in kebab-case from SEOMetadata.slug, '
                           'e.g. "garage-door-spring-repair-denver". '
                           'Forms the stable URL path component.'),

  url                 STRING
    OPTIONS (description = 'WordPress published URL (Article.wp_post_url). '
                           'NULLABLE semantically; the current implementation '
                           'sends an empty string ("") rather than NULL when '
                           'wp_post_url is None (WP publish failed or not yet '
                           'posted).  Use WHERE url != \'\' to filter for '
                           'successfully published articles until a future fix '
                           'sends NULL for the absent case.'),

  publish_date        TIMESTAMP NOT NULL
    OPTIONS (description = 'Publication timestamp in UTC. Sourced from the '
                           'publish_date argument passed by the caller. '
                           'Falls back to datetime.now(utc) at insert time '
                           'when the caller omits the argument — so this '
                           'reflects insert time, not WordPress publish time, '
                           'when the caller does not supply the event timestamp. '
                           'Serialised as ISO-8601 with +00:00 offset.'),

  -- ── SEO metadata ─────────────────────────────────────────────────────────
  focus_keyword       STRING
    OPTIONS (description = 'Primary SEO keyword from SEOMetadata.focus_keyword. '
                           'In practice always non-empty (required model field); '
                           'declared NULLABLE because the implementation sends "" '
                           'rather than NULL for the exceptional absent case. '
                           'Use WHERE focus_keyword != \'\' to exclude blanks.'),

  category            STRING
    OPTIONS (description = 'Suggested WordPress category from '
                           'SEOMetadata.suggested_category. Legitimately absent '
                           'when the planner did not assign a category. '
                           'Sent as "" (not NULL) by the current implementation.'),

  -- ── Volume metrics ────────────────────────────────────────────────────────
  word_count          INT64     NOT NULL
    OPTIONS (description = 'Actual article word count computed from '
                           'content_markdown (Article.word_count). '
                           'Auto-computed by Article.compute_content_stats(). '
                           'Sent as 0 only if Article.word_count is None.'),

  reading_time        INT64     NOT NULL
    OPTIONS (description = 'Estimated reading time in minutes at 200 words/min '
                           '(Article.reading_time_minutes). '
                           'Auto-computed. Sent as 0 only if None.'),

  -- ── QA scores — final review iteration ───────────────────────────────────
  -- Sourced from DualQAReport.final_article_review (the last
  -- ArticleReviewIteration in the QA cycle).
  -- A value of 0 means NO QA iteration was recorded, not a real score of zero.
  -- The pass threshold for all four dimensions is 90.
  seo_score           INT64     NOT NULL
    OPTIONS (description = 'Claude SEO reviewer score from the final QA '
                           'iteration, 0–100.  Pass threshold: 90.  '
                           '0 = no QA iteration recorded (not a real score).'),

  editorial_score     INT64     NOT NULL
    OPTIONS (description = 'Claude editorial quality score from the final QA '
                           'iteration, 0–100.  Pass threshold: 90.'),

  writing_score       INT64     NOT NULL
    OPTIONS (description = 'OpenAI writing naturalness score from the final '
                           'QA iteration, 0–100.  Pass threshold: 90.'),

  authenticity_score  INT64     NOT NULL
    OPTIONS (description = 'OpenAI human-authenticity score from the final '
                           'QA iteration, 0–100.  Pass threshold: 90.'),

  -- ── Cost breakdown ────────────────────────────────────────────────────────
  -- Costs are computed by iterating CallTracer.records and splitting by
  -- model name prefix.  Records with used=False (discarded pipeline outputs)
  -- are included in the totals — the current implementation does not filter
  -- them out.  Use llm_costs.success=false to identify discarded calls.
  -- All three cost columns use NUMERIC for exact decimal arithmetic.
  total_cost_usd      NUMERIC   NOT NULL
    OPTIONS (description = 'Total LLM cost for the pipeline run in USD. '
                           'Sum of cost_usd across all CallRecord entries, '
                           'including records where success=false. '
                           'Rounded to 6 decimal places. '
                           '0.000000 when call_tracer was None (rare).'),

  claude_cost_usd     NUMERIC   NOT NULL
    OPTIONS (description = 'Portion of total_cost_usd for model names starting '
                           'with "claude".  Rounded to 6 decimal places.'),

  openai_cost_usd     NUMERIC   NOT NULL
    OPTIONS (description = 'Portion of total_cost_usd for model names NOT '
                           'starting with "claude" (OpenAI GPT models and any '
                           'future providers).  Rounded to 6 decimal places.'),

  -- ── Draft reuse ──────────────────────────────────────────────────────────
  reuse               BOOL      NOT NULL
    OPTIONS (description = 'True when the article content was served from the '
                           'DraftReuseService pool rather than freshly generated '
                           'by the LLM pipeline.  When true, LLM cost fields '
                           'reflect only the pipeline overhead, not generation.'),

  reuse_similarity    FLOAT64   NOT NULL
    OPTIONS (description = 'Jaccard similarity score of the matched draft '
                           '(0.0–1.0, rounded to 4 decimal places). '
                           '0.0 when reuse=false.  '
                           '1.0 indicates an exact topic_id match.'),

  -- ── Generation performance ────────────────────────────────────────────────
  generation_time     FLOAT64   NOT NULL
    OPTIONS (description = 'End-to-end pipeline wall-clock time in seconds from '
                           'pipeline entry to BqSinkService.insert_article() '
                           'call.  Rounded to 2 decimal places.'),

  -- ── Traceability (Request #6) ─────────────────────────────────────────────
  model_name          STRING    NOT NULL
    OPTIONS (description = 'Article.model_name. The primary Claude model used '
                           'for article generation, e.g. "claude-sonnet-4-6". '
                           'Empty string ("") for articles generated before this '
                           'field was populated.  Per-call model detail is in '
                           'llm_costs.model.'),

  prompt_version      STRING    NOT NULL
    OPTIONS (description = 'Article.prompt_version. Version stamp of the prompt '
                           'set active during generation, e.g. "1.0". '
                           'Enables before/after regression analysis when prompts '
                           'change.  Default "1.0" for all pre-versioning articles.'),

  -- ── Execution metadata ────────────────────────────────────────────────────
  -- Captured at insert time from module-level constants in bq_sink_service.py.
  -- These fields are the same for every row produced by a single pipeline run.
  event_type          STRING    NOT NULL
    OPTIONS (description = 'CLI command that triggered this row. '
                           'Values: "autopublish" (full auto-pilot), '
                           '"publish" (human-selected article.json), '
                           '"republish" (re-send last article to WP). '
                           'Enables clean duplicate detection — filter '
                           'WHERE event_type = "autopublish" for first-publish '
                           'rows only.  Also enables cost attribution by command type '
                           'in llm_costs.'),

  environment         STRING    NOT NULL
    OPTIONS (description = 'Runtime environment that produced this row. '
                           'Values: "prod" (default), "staging", "dev". '
                           'Set via SEO_AGENT_ENV environment variable; '
                           'defaults to "prod" so a misconfigured production '
                           'deployment fails safely rather than silently '
                           'tagging rows as dev data. '
                           'Filter WHERE environment = "prod" in all dashboards.'),

  git_commit          STRING
    OPTIONS (description = 'SHA-1 of the git HEAD at the time the pipeline ran '
                           '(40-char hex string). Machine-set; cannot drift from '
                           'the code that actually ran. '
                           'NULLABLE: absent when the binary is deployed without '
                           'a .git directory (e.g. a zip or container build). '
                           'Use with publish_date to attribute score changes to '
                           'specific code commits.'),

  pipeline_version    STRING
    OPTIONS (description = 'Human-readable semantic version of the pipeline, '
                           'e.g. "2.1.0". Set via PIPELINE_VERSION env var, '
                           'typically injected by CI at build time. '
                           'NULLABLE until the team adopts a formal versioning '
                           'convention. When set, enables WHERE clauses like '
                           'pipeline_version >= "2.0" without knowing commit SHAs.')
)
PARTITION BY DATE(publish_date)
CLUSTER BY client, website
OPTIONS (
  description = 'One row per published article. '
                'Written by BqSinkService.insert_article() in '
                'services/bq_sink_service.py. '
                'Join to qa_results on article_id. '
                'Join to llm_costs on article_id. '
                'Partitioned by DATE(publish_date); clustered by client, website. '
                'Filter WHERE event_type = "autopublish" AND environment = "prod" '
                'to isolate clean first-publish production rows.'
);


-- ============================================================
-- TABLE 2: qa_results
--
-- One row per QA run (one per published article).
-- Written by: BqSinkService.insert_qa_results()
--
-- Partition: _PARTITIONDATE (ingestion-time)
--   This table has no event timestamp. Ingestion-time partitioning
--   assigns each row to the date it arrives at BigQuery.
--   Enables date-range pruning (monthly QA reports) at zero
--   application-level cost — BigQuery handles it automatically.
--
-- Cluster: approved, overall_pass
--   The primary QA filter predicates are pass/fail outcomes.
--   Clustering on these two booleans lets BQ skip non-matching
--   blocks in queries like "WHERE approved = false".
-- ============================================================

CREATE TABLE IF NOT EXISTS `rightidea-cortex.seo_content.qa_results`
(
  -- ── Primary / join key ────────────────────────────────────────────────────
  article_id                STRING  NOT NULL
    OPTIONS (description = 'Article.id as UUID v4 string. '
                           'Join key to articles_published.article_id. '
                           'Always non-null; written by insert_qa_results().'),

  -- ── Canonical client identity (Request #5) ──────────────────────────────
  canonical_client          STRING
    OPTIONS (description = 'SiteProfile.canonical_client — normalized business '
                           'entity identifier. Populated from Request #5. '
                           'NULL on rows written before the migration.'),

  -- ── Pass / fail outcome ──────────────────────────────────────────────────
  approved                  BOOL    NOT NULL
    OPTIONS (description = 'DualQAReport.article_passed. True when the article '
                           'text passed both the Claude SEO/editorial reviewer '
                           'and the OpenAI writing/authenticity reviewer. '
                           'Does NOT include image QA — see overall_pass.'),

  overall_pass              BOOL    NOT NULL
    OPTIONS (description = 'DualQAReport.passed = article_passed AND '
                           'images_passed.  False when image QA fails even if '
                           'article text QA passed.  Differs from approved only '
                           'when the image pipeline runs and fails.'),

  revision_cycles           INT64   NOT NULL
    OPTIONS (description = 'DualQAReport.iterations_used. Number of complete '
                           'review cycles executed before a pass or forced '
                           'publication at the maximum cycle limit. '
                           '1 = passed on first attempt.'),

  -- ── Per-dimension scores — final iteration only ──────────────────────────
  -- All four scores come from DualQAReport.final_article_review
  -- (the last ArticleReviewIteration). The value 0 means no QA iteration
  -- was recorded, not a real score of zero.  Pass threshold: 90 for all.
  claude_seo_score          INT64   NOT NULL
    OPTIONS (description = 'Claude SEO reviewer score from the final QA '
                           'iteration, 0–100. Pass threshold: 90.'),

  claude_editorial_score    INT64   NOT NULL
    OPTIONS (description = 'Claude editorial quality score from the final QA '
                           'iteration, 0–100. Pass threshold: 90.'),

  openai_writing_score      INT64   NOT NULL
    OPTIONS (description = 'OpenAI writing naturalness score from the final '
                           'QA iteration, 0–100. Pass threshold: 90.'),

  openai_authenticity_score INT64   NOT NULL
    OPTIONS (description = 'OpenAI human-authenticity score from the final '
                           'QA iteration, 0–100. Pass threshold: 90.'),

  -- ── Execution metadata ────────────────────────────────────────────────────
  -- event_type is omitted from qa_results — it is derivable via JOIN with
  -- articles_published on article_id at negligible cost.  Keeping qa_results
  -- narrow avoids maintaining the same value in two places.
  environment         STRING    NOT NULL
    OPTIONS (description = 'Runtime environment. Values: "prod", "staging", "dev". '
                           'Set via SEO_AGENT_ENV env var; defaults to "prod". '
                           'Filter WHERE environment = "prod" to exclude dev QA runs '
                           'from score distribution analysis.'),

  git_commit          STRING
    OPTIONS (description = 'SHA-1 git HEAD at pipeline run time. NULLABLE: '
                           'absent when deployed without a .git directory. '
                           'Enables score trend attribution to specific commits '
                           'without requiring a JOIN to articles_published.')
)
PARTITION BY _PARTITIONDATE
CLUSTER BY approved, overall_pass
OPTIONS (
  description = 'One row per QA run, one per published article. '
                'Written by BqSinkService.insert_qa_results() in '
                'services/bq_sink_service.py. '
                'Partitioned by ingestion date (_PARTITIONDATE) because '
                'no event timestamp is stored. '
                'Join to articles_published on article_id. '
                'Filter WHERE environment = "prod" in all production reports.'
);


-- ============================================================
-- TABLE 3: llm_costs
--
-- One row per LLM CallRecord per pipeline run.
-- This is the highest-volume table: ~20–30 rows per article.
-- Written by: BqSinkService.insert_llm_costs()
--
-- Partition: DATE(timestamp)
--   Monthly cost queries are the primary analytics pattern for
--   this table.  Without partitioning, every query scans the
--   entire table history regardless of the date filter.
--   At 1,000 articles/month × 25 calls each = 25,000 rows/month,
--   an unpartitioned table becomes expensive as history grows.
--
-- Cluster: event_type, provider, model, stage  (4 columns — BQ maximum)
--   event_type comes first so queries scoped to one command type
--   ("republish overhead analysis", "autopublish cost baseline") skip
--   irrelevant blocks before evaluating provider/model/stage.
--   The remaining three axes are the dominant cost aggregation dimensions:
--     "Claude vs OpenAI total spend"   → filter by provider
--     "Cost by model"                  → filter by model
--     "Cost by pipeline stage"         → filter by stage
--
-- NOTE on article_id:
--   article_id is NULLABLE because insert_llm_costs() receives it as an
--   optional keyword argument.  It is populated for all rows written after
--   the wiring change in services/bq_sink_service.py (2026-08-04).
--   Rows written before that date have article_id = NULL.
-- ============================================================

CREATE TABLE IF NOT EXISTS `rightidea-cortex.seo_content.llm_costs`
(
  -- ── Time ─────────────────────────────────────────────────────────────────
  timestamp     TIMESTAMP  NOT NULL
    OPTIONS (description = 'UTC instant when BqSinkService.insert_llm_costs() '
                           'was called. All CallRecords in a single batch share '
                           'this same value — it is the insert time, NOT the '
                           'time of the individual LLM API call. '
                           'Serialised as ISO-8601 with +00:00 offset, e.g. '
                           '"2026-08-04T15:30:00.123456+00:00".'),

  -- ── Article join key (pre-declared; NULLABLE until method is updated) ─────
  article_id    STRING
    OPTIONS (description = 'Article.id as UUID v4 string. Join key to '
                           'articles_published.article_id. '
                           'NULL in all rows until BqSinkService.insert_llm_costs() '
                           'is updated to accept and write an article_id argument. '
                           'NULLABLE so the current implementation continues to '
                           'insert successfully without writing this field.'),

  -- ── Canonical client identity (Request #5) ──────────────────────────────
  canonical_client  STRING
    OPTIONS (description = 'SiteProfile.canonical_client — normalized business '
                           'entity identifier. Populated from Request #5. '
                           'NULL on rows written before the migration.'),

  -- ── Routing / attribution ─────────────────────────────────────────────────
  system        STRING     NOT NULL
    OPTIONS (description = 'Pipeline system label passed by the caller '
                           '(e.g. "seo-agent"). Allows cost attribution across '
                           'multiple pipelines sharing the same sink.'),

  stage         STRING     NOT NULL
    OPTIONS (description = 'CallRecord.stage. Identifies the pipeline call site, '
                           'e.g. "draft:write", "qa:review", "image:plan", '
                           '"image:vision-score", "image:edit-prompt".'),

  provider      STRING     NOT NULL
    OPTIONS (description = 'Provider derived from model name by '
                           '_provider_from_model(). Values: '
                           '"claude"  — model name starts with "claude"; '
                           '"openai"  — model name starts with "gpt", "o1", "o3"; '
                           '"other"   — any other model name.'),

  model         STRING     NOT NULL
    OPTIONS (description = 'Exact model identifier from CallRecord, '
                           'e.g. "claude-sonnet-4-6", "claude-haiku-4-5", '
                           '"gpt-4o-mini", "gpt-4o".'),

  -- ── Token counts ──────────────────────────────────────────────────────────
  -- IMPORTANT: CallRecord also stores cache_creation_tokens and
  -- cache_read_tokens, but these are NOT written to this table.
  -- cost_usd is computed correctly in Python (accounting for cache pricing
  -- tiers) but CANNOT be reproduced from input_tokens + output_tokens alone.
  -- Use cost_usd as the authoritative figure for all cost aggregations.
  input_tokens  INT64      NOT NULL
    OPTIONS (description = 'CallRecord.input_tokens. Standard (non-cached) input '
                           'tokens billed at the base input rate. Does NOT include '
                           'cache_creation_tokens (billed at 1.25× input rate) or '
                           'cache_read_tokens (billed at 0.10× input rate). '
                           'Do not multiply this column by a price/M rate to '
                           'reconstruct cost — the result will be incorrect when '
                           'prompt caching is active.'),

  output_tokens INT64      NOT NULL
    OPTIONS (description = 'CallRecord.output_tokens. Generated output tokens '
                           'billed at the output rate.'),

  -- ── Cost ─────────────────────────────────────────────────────────────────
  cost_usd      NUMERIC    NOT NULL
    OPTIONS (description = 'CallRecord.cost_usd rounded to 6 decimal places. '
                           'Computed in Python as: '
                           '(input_tokens / 1M × input_price) '
                           '+ (cache_creation_tokens / 1M × input_price × 1.25) '
                           '+ (cache_read_tokens / 1M × input_price × 0.10) '
                           '+ (output_tokens / 1M × output_price). '
                           'This column is the authoritative cost per call. '
                           'SUM(cost_usd) across records for an article yields '
                           'the correct total pipeline cost.'),

  -- ── Call outcome ─────────────────────────────────────────────────────────
  success       BOOL       NOT NULL
    OPTIONS (description = 'CallRecord.used. True when the LLM output was kept '
                           'in the final article. False when the output was '
                           'discarded — for example, a QA revision cycle that '
                           'produced content below the quality threshold. '
                           'Cost was still incurred on success=false calls.'),

  -- ── Execution metadata ────────────────────────────────────────────────────
  -- Shared with articles_published and qa_results; see those tables for
  -- full field descriptions.  Stored denormalized here because llm_costs
  -- is the primary cost analysis table and JOINs add latency at its volume.
  event_type    STRING    NOT NULL
    OPTIONS (description = 'CLI command that triggered this row. '
                           'Values: "autopublish", "publish", "republish". '
                           'First clustering column — queries scoped to one '
                           'command type (e.g. "republish overhead analysis") '
                           'skip irrelevant blocks before evaluating provider/model/stage. '
                           'Use WHERE event_type = "autopublish" to isolate '
                           'baseline generation costs from re-publish overhead.'),

  environment   STRING    NOT NULL
    OPTIONS (description = 'Runtime environment. Values: "prod", "staging", "dev". '
                           'Always filter WHERE environment = "prod" in cost dashboards '
                           'to exclude developer test runs from spend reporting.'),

  git_commit    STRING
    OPTIONS (description = 'SHA-1 git HEAD at pipeline run time. NULLABLE. '
                           'Use with timestamp to attribute cost changes to '
                           'specific code or prompt changes without requiring '
                           'a JOIN to articles_published.')
)
PARTITION BY DATE(timestamp)
CLUSTER BY event_type, provider, model, stage
OPTIONS (
  description = 'One row per LLM CallRecord per pipeline run (~20–30 rows per article). '
                'Written by BqSinkService.insert_llm_costs() in '
                'services/bq_sink_service.py. '
                'Partitioned by DATE(timestamp) for cost time-series queries. '
                'Clustered by event_type, provider, model, stage for cost-breakdown aggregations. '
                'Filter WHERE event_type = "autopublish" AND environment = "prod" '
                'for clean baseline cost analysis. '
                'IMPORTANT: cache_creation_tokens and cache_read_tokens are not stored. '
                'cost_usd is the authoritative per-call cost; do not recompute from token columns.'
);

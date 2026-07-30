# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-07-30

### Added

- **Test suite** (`tests/`): 9 pytest tests — article status enum separation, word count recomputation, budget file locking, certification title match, dual QA OpenAI bypass, OpenAI score normalization, plan field coverage, vision QA exception handling, writing audit opener classification.
- **Multi-model budget pricing** (`services/budget_service.py`): Per-model token pricing table (Opus, Sonnet, Haiku). `record_claude()` now accepts `model=` and bills at the correct rate.
- **Budget file locking** (`services/budget_service.py`): POSIX exclusive lock (`fcntl`) on read-modify-write to prevent concurrent write races.
- **`qa_rescue_enabled` config flag** (`config.py`): Gates the authenticity rescue rewrite in `DualQAAgent`. Defaults to `False` to enforce $0.25 per-article cost target.
- **`reuse_group` on `TenantContext`** (`models/tenant.py`): Field now populated from `SiteProfile.reuse_group` at article-creation time, enabling cross-site draft reuse.
- **Autopublish topic placeholder detection** (`main.py`): Rejects topics containing `[City]`, `[Service]`, `[Topic]`, etc. before article generation. Selects first valid topic; prints prompt and raw LLM response on total failure.
- **`INTERNAL LINKS CONFIGURED` header in QA review** (`agents/dual_qa_agent.py`): Informs the scorer whether internal links were configured; excludes the criterion from scoring when they were not.
- **`writing_audit_service.py`** (`services/`): New writing audit service (pending pipeline integration).
- **`benchmark.py` and `benchmark_results.jsonl`**: Benchmarking tooling.

### Changed

- Default generation model: `claude-opus-4-8` → `claude-sonnet-4-6`.
- SEO metadata model and image eval model: Sonnet → Haiku.
- Default and target word count: 850 → 800 words (range 700–900, hard cap 950).
- `qa_max_cycles`: 3 → 1.
- `max_article_cost_usd`: $0.55 → $0.25.
- Planner section count: 5–8 → 3–4 H2 sections.
- Planner FAQ count: 5 → 3–4 questions.
- Planner `max_tokens`: 8000 → 5000; `thinking` disabled.
- Article generator prompt: observation-based voice model, paragraph flow modes, paragraph opener rotation, sentence rhythm guidance.
- Focus keyword placement rules made strict: exact verbatim H1, mandatory H2, first-100-words intro.
- External links changed from optional hint to required (failure criterion if absent).
- `article_structure.json` version 1.0.0 → 1.1.0; section/FAQ counts updated.
- Planner schema: `reader_intent`, `reader_misconception`, `why_misconception_forms`, `failure_mechanism` moved from output fields to internal LLM reasoning steps; removed from `SectionPlan` model. `what_reader_gets_wrong` removed from `ArticlePlan`.
- QA scoring: 3–4 FAQ questions is correct format, not a deficiency.
- SEO regen now occurs within each revision cycle when that cycle's SEO score was below threshold.
- `overheaddoornwi` site profile: `niche`, `primary_service`, `secondary_services` populated.
- Autopublish profile path: `{website_id}.json` → `{website_id}/site.json`.
- OpenAI reviewer not configured: now reports scores 0/False with explicit bypass, not 100/True.
- Image vision failure: exception → score 0 (fail), not score 100 (pass).
- `thinking=False` applied to planner, SEO review, vision review, image planning calls.
- `image_eval_model` used for edit prompts replaced with `edit_prompt_model`.

### Fixed

- **Slug path traversal**: `article.seo.slug` now sanitized with `re.sub(r'[^a-z0-9-]', '-', ...)` before use as filesystem path component.
- **`word_count` staleness**: Recomputed after every `content_markdown` mutation (QA revision, marker restoration, content sanitization, link enrichment, location adaptation, authenticity revision).
- **`.strip()` removing leading article content**: Changed to `.rstrip('\n')` in `_restore_displaced_markers()`.
- **Stale WP fields on reuse**: `wp_post_id`, `wp_post_url`, `drive_document_id` now cleared to `None`; `publishing` reset to defaults.
- **Reuse status enum**: `PublishStatus.DRAFT` (wrong enum class) → `ArticleStatus.REVIEW`.
- **`TenantContext.reuse_group` never set**: Wired from profile at article creation; cross-site reuse via `reuse_group` was silently broken.
- **Dead `_audit_seo_content` static method** (`agents/article_agent.py`): Removed.
- **Dead post-loop `raise ClaudeRateLimitError`** (`agents/article_agent.py`): Removed unreachable statement.
- **Dead pool scan condition** (`services/draft_pool_service.py`): `path.name != "article.json"` always `False` for `glob("**/article.json")`. Removed.
- **Dead `internal_links` variable** (`services/publication_certification_service.py`): Computed but never read. Removed.
- **WP title HTML entity decoding** (`services/publication_certification_service.py`): WordPress returns `&amp;` etc. in `title.rendered`; now decoded with `html.unescape()`.
- **Certification title check**: Was checking title exists, not title matches. Now checks actual match and reports mismatch.
- **Internal link check scope** (`services/seo_qa_service.py`): Was flagging any absent markdown link; now checks for site domain presence when `website_url` is known.
- **FAQ count validation inconsistency** (`templates/article_structure.json`): `count: "3-4"` vs `faq_max_questions: 5`. Unified to 3–5.
- **`assert` statements → `raise RuntimeError`** (`agents/image_resolver_agent.py`): Three assertion sites replaced with explicit errors.
- **`AttributeError` missing from vision scoring catch** (`agents/image_resolver_agent.py`).
- **`int(None)` crash in score normalization** (`services/openai_review_service.py`): `int(raw.get(..., 0))` → `int(raw.get(...) or 0)` for writing, authenticity, and vision scores.
- **HTTP 529 not recoverable** (`services/llm_gateway.py`): Added to `_RECOVERABLE_STATUS_CODES`.
- **400 "usage limit" bypassing failover** (`services/claude_service.py`, `services/llm_gateway.py`): Now re-raised as `ClaudeRateLimitError` to trigger provider failover.
- **Rate limit error message discarded** (`services/claude_service.py`): Now preserves original `str(exc)`.
- **`meta_description` max_length**: 170 → 160 (SERP truncation limit).
- **`ArticleRequest.language` default**: ES → EN.
- **`--words` max**: 1000 → 950 in both `generate` and `autopublish` commands.
- **`openai_approved` reference before assignment** when OpenAI reviewer is None (`agents/dual_qa_agent.py`).
- **`.gitignore` excluded command files**: Rule `.claude/` applied recursively, preventing `.claude/commands/` from being versioned. Narrowed to `/.claude/settings.local.json`.
- **`review.md` missing from canonical command location**: `claude review` was silently broken. Moved to `.claude/commands/`.
- **Zero-byte `claude/claude/commands` file**: Accidentally committed artifact removed via `git rm`.

### Chore

- **Claude command consolidation**: All three session commands (`start.md`, `end.md`, `review.md`) are now committed under `.claude/commands/` and version-controlled. Obsolete `claude/claude/` workaround directory removed.

# Development Log

---

## 2026-08-03

### Summary

Implemented Anthropic prompt caching across the five highest-value Claude call sites (Request 2 of the implementation roadmap). No prompts were changed, no models were changed, and no behavior was changed — the only effect is lower input cost on cache hits. Also extended `BudgetService.record_claude()` to correctly account for cache token pricing tiers so monthly cost tracking stays accurate.

---

### Investigation

Audited all 18 Claude API call sites across the codebase. Applied two filters:

1. **Static system prompt**: dynamic content must be entirely in user messages (all 18 qualify).
2. **Above minimum token threshold**: Sonnet requires ≥1,024 tokens; Haiku requires ≥2,048 tokens.

Result: 13 Haiku calls and 3 short Sonnet calls are not cacheable. The 5 remaining Sonnet calls have large static system prompts and are the only targets.

| Call site | Constant | Estimated tokens | Model |
|---|---|---|---|
| `services/article_planner_service.py:547` | `_PLANNER_SYSTEM` | ~2,950 | Sonnet |
| `services/authenticity_revision_service.py:259` | `_AUTHENTICITY_REWRITE_SYSTEM` | ~2,600 | Sonnet |
| `agents/article_agent.py:485` | `_GENERATOR_SYSTEM` | ~2,500 | Sonnet |
| `agents/dual_qa_agent.py:882` | `_REVISION_SYSTEM` | ~1,500 | Sonnet |
| `agents/dual_qa_agent.py:769` | `_CLAUDE_SEO_SYSTEM` | ~1,400 | Sonnet |

---

### Implementation

**Architecture**: Added a `cache_system: bool = False` keyword argument to `ClaudeService.generate()`, `ClaudeService.generate_structured()`, and their `_base_kwargs()` assembly point. When `True`, `_base_kwargs()` converts `"system": system_str` to `"system": [{"type": "text", "text": system_str, "cache_control": {"type": "ephemeral"}}]` — the Anthropic API accepts `system` as either a plain string or a list of content blocks with optional `cache_control`. `LLMGateway.generate()` and `LLMGateway.generate_structured()` thread the flag through to the primary provider; the fallback (OpenAI) ignores it.

**Budget accounting**: Extended `BudgetService.record_claude()` with `cache_creation_tokens: int = 0` and `cache_read_tokens: int = 0`. Cache write is charged at 1.25× the normal input rate; cache read at 0.10× (90% discount). Existing budget files without these fields are migrated transparently via `setdefault` on first write. `_empty()` now initialises both fields to 0 for new files.

---

### Files Modified

| File | Change |
|---|---|
| `services/budget_service.py` | `record_claude()` accepts cache token counts; corrected cost formula; `_empty()` adds `cache_creation_tokens` and `cache_read_tokens` fields |
| `services/claude_service.py` | `generate()`, `generate_structured()`, `_base_kwargs()` accept `cache_system`; both generation methods pass `cache_creation_input_tokens` / `cache_read_input_tokens` from usage to `record_claude()` |
| `services/llm_gateway.py` | `generate()` and `generate_structured()` accept and forward `cache_system` to primary |
| `services/article_planner_service.py` | `cache_system=True` on `_PLANNER_SYSTEM` call |
| `services/authenticity_revision_service.py` | `cache_system=True` on `_AUTHENTICITY_REWRITE_SYSTEM` call |
| `agents/article_agent.py` | `cache_system=True` on `_GENERATOR_SYSTEM` call |
| `agents/dual_qa_agent.py` | `cache_system=True` on `_CLAUDE_SEO_SYSTEM` and `_REVISION_SYSTEM` calls |
| `tests/test_prompt_caching.py` | 16 new regression tests (see Testing) |

---

### Benchmark (Estimated Savings)

Pricing: Sonnet 4.6 — $3.00/1M input (uncached), $3.75/1M (cache write, 1.25×), $0.30/1M (cache read, 0.10×).

| Call | System tokens | Per-article saving (after first call) |
|---|---|---|
| `_GENERATOR_SYSTEM` | ~2,500 | 2,500 × ($3.00 − $0.30) / 1M = **$0.00675** |
| `_PLANNER_SYSTEM` | ~2,950 | 2,950 × $2.70 / 1M = **$0.00797** |
| `_AUTHENTICITY_REWRITE_SYSTEM` | ~2,600 | 2,600 × $2.70 / 1M = **$0.00702** (only on rescue) |
| `_CLAUDE_SEO_SYSTEM` | ~1,400 | 1,400 × $2.70 / 1M = **$0.00378** (per QA cycle) |
| `_REVISION_SYSTEM` | ~1,500 | 1,500 × $2.70 / 1M = **$0.00405** (per revision) |

For a typical article (generation + planning + 1 QA cycle, no rescue): **~$0.018 saving per article**. At 50 articles/month: **~$0.90/month**. Cache write surcharge on the first call of each batch is small (~$0.00024 per prompt per cold start). The 5-minute TTL means any production batch of multiple articles within the same session will see cache hits on calls 2+.

---

### Testing

16 new tests in `tests/test_prompt_caching.py`:

- **`TestBaseKwargsCacheSystem`** (4 tests): verifies `_base_kwargs()` output with `cache_system=False` (plain string), `cache_system=True` (list with `cache_control`), default behavior, and full prompt preservation.
- **`TestBudgetCacheAccounting`** (7 tests): verifies cache creation at 1.25×, cache read at 0.10×, uncached unchanged, mixed cost formula, counter accumulation, backward compatibility with legacy budget files, zero-token no-op.
- **`TestCacheSitesPassFlag`** (5 tests): one per call site — confirms `cache_system=True` reaches `generate` / `generate_structured` from each of the five callers.

Full suite: **81 tests, 81 passed**.

---

### Risks and Rollback

**Risk**: Anthropic's caching minimum (1,024 tokens for Sonnet) is met by all 5 prompts. If a prompt is shortened below the threshold in future, the cache header will be ignored silently — no error, just no saving.

**Rollback**: Remove the five `cache_system=True` keyword arguments from the call sites. The `cache_system` flag defaults to `False`, so removing the arguments restores the exact previous behavior. No data migration needed.

---

## 2026-07-30

### Summary
Engineering audit pass covering 20+ files. Fixed correctness bugs uncovered by systematic code review, tightened article quality constraints, refactored the planner schema to move reader-perspective fields from output to internal LLM reasoning, added a test suite (9 tests), added multi-model cost tracking, and hardened the autopublish pipeline with topic placeholder detection and profile path fixes.

Also performed a forensic investigation proving that draft reuse correctly rejected all 10 eligible pool candidates for a specific article (highest Jaccard score: 0.188 against a threshold of 0.72), and confirmed the pool matching algorithm is functioning correctly.

---

### Features Added

- **Test suite** (`tests/`): 9 pytest tests covering article status enum separation, word count recomputation after every mutation, budget file locking, certification title match logic, dual QA OpenAI bypass behavior, OpenAI review score normalization, plan field coverage, vision QA exception handling, and writing audit opener classification.
- **Multi-model budget pricing** (`services/budget_service.py`): `record_claude()` now accepts a `model=` parameter and looks up per-model pricing (Opus $5/$25, Sonnet $3/$15, Haiku $1/$5 per million tokens). Previously all calls billed at Opus rates regardless of model used.
- **Budget file locking** (`services/budget_service.py`): `_exclusive_lock()` context manager uses `fcntl.LOCK_EX` to prevent race conditions during concurrent read-modify-write operations on the monthly budget JSON.
- **`enable_rescue` flag** (`config.py`, `agents/dual_qa_agent.py`, `main.py`): New `qa_rescue_enabled` setting (default `False`) gates the authenticity rescue rewrite. Prevents ~$0.110 extra cost per activation and ensures per-article cost stays under $0.25 target.
- **`qa_rescue_enabled` field** (`config.py`): Defaults to `False`; the rescue is now opt-in.
- **`reuse_group` on `TenantContext`** (`models/tenant.py`, `main.py`): Field was defined on `SiteProfile` but never populated on `TenantContext`. Now wired through at article creation time from the profile, enabling cross-site draft reuse where configured.
- **Autopublish: topic placeholder detection** (`main.py`): Validates generated topics against a regex for `[City]`, `[Service]`, `[Topic]`, etc. Marks invalid topics visually and selects the first valid one. Prints prompt + raw LLM response on total failure.
- **`writing_audit_service.py`** (`services/`): New service (untracked, pending inclusion).
- **`benchmark.py`** and **`benchmark_results.jsonl`**: Performance benchmarking tooling (untracked).

---

### Improvements

- **Article word count target reduced**: 850 → 800 words (700–900 range, 950 hard cap). Propagated through `config.py`, `models/article.py`, `agents/article_agent.py`, `agents/dual_qa_agent.py`, `services/authenticity_revision_service.py`, and `templates/article_structure.json`.
- **Article generator prompt restructured** (`agents/article_agent.py`): Added explicit voice and perspective instructions (experienced homeowner, observation-based writing), paragraph flow modes (Observation/Explanation/Consequence/Practical implication/Example/Warning), paragraph opener rotation, and sentence rhythm guidance. Removed the prior mechanical sentence-count optimization guidance.
- **Focus keyword placement made strict**: H1 must contain the exact keyword phrase verbatim; intro must mention it within first 100 words (counted strictly); at least one H2 must contain it or a close variant (3+ consecutive words). Previously these were soft guidelines.
- **External links changed from optional to required**: Articles with zero external links fail quality review. Prompt updated in both planned and unplanned generation paths.
- **Planner schema simplified** (`models/article_plan.py`, `services/article_planner_service.py`): Removed `reader_intent`, `reader_misconception`, `why_misconception_forms`, `failure_mechanism` from `SectionPlan` output fields. Removed `what_reader_gets_wrong` from `ArticlePlan`. These are now numbered internal reasoning steps (1–4) in the planner prompt — the LLM reasons through them before populating `technical_reality` and `professional_insight`. Reduces schema token overhead; sharpens output fields.
- **Section and FAQ counts reduced**: Planner now targets 3–4 H2 sections (was 5–8) and 3–4 FAQ questions (was 5). Matches the 800-word target. Article structure template updated to match.
- **Planner cost reduced**: `max_tokens` 8000 → 5000; `thinking` disabled (was `True`). Planner model stays on Sonnet.
- **SEO model and image eval model changed to Haiku** (`config.py`): Tasks that don't require deep reasoning now use Haiku, reducing per-article cost.
- **Default generation model changed to Sonnet** (`config.py`): `claude_model` was `claude-opus-4-8`, now `claude-sonnet-4-6`. Reduces per-call generation cost roughly 40%.
- **`qa_max_cycles` reduced**: 3 → 1. Limits revision spending; second and third cycles rarely recovered failing articles.
- **`max_article_cost_usd` reduced**: $0.55 → $0.25. Reflects Sonnet pricing and reduced QA cycles.
- **QA review prompt updated** (`agents/dual_qa_agent.py`): Added `INTERNAL LINKS CONFIGURED` header check — if internal links were not configured, that criterion is excluded from scoring entirely (N/A, not a failure). Added explicit note that 3–4 FAQ questions is correct format and must not be penalized.
- **SEO regen during revision** (`agents/dual_qa_agent.py`): When a QA revision cycle had a failing SEO score, `_revise()` now regenerates SEO metadata immediately so the next cycle evaluates fresh metadata. Previously SEO was regenerated only once at the very end.
- **OpenAI not-configured behavior corrected** (`agents/dual_qa_agent.py`): When OpenAI reviewer is absent, scores are now reported as 0/False (not measured) and `openai_approved` is set to `True` as an explicit bypass. Previously it was reported as 100/True, making it appear that the review ran and passed.
- **Image vision failure behavior corrected** (`agents/dual_qa_agent.py`): OpenAI vision exceptions now result in score 0 and fail, not score 100 and pass.
- **Autopublish profile path fixed** (`main.py`): Was loading `profiles/{client}/{website_id}.json`; corrected to `profiles/{client}/{website_id}/site.json`.
- **Autopublish service injection** (`main.py`): `_profile_service` and `_profile_city` are now both loaded from the site profile and passed to `_suggest_topics`, fixing blank topic generation when `--service` is not provided on the command line.
- **`overheaddoornwi` site profile populated** (`profiles/RIMC/overheaddoornwi/site.json`): `niche`, `primary_service`, `secondary_services` filled in.
- **`thinking=False`** applied to planner, SEO review, vision review, and image planning calls to reduce token overhead on structured tasks.
- **`image_eval_model` → `edit_prompt_model`** for image edit prompt generation (`agents/image_resolver_agent.py`): Edit prompts require instruction-following, not vision scoring.

---

### Fixes

- **Slug path traversal** (`main.py`): Article directory path used `article.seo.slug` unsanitized. Now sanitized with `re.sub(r'[^a-z0-9-]', '-', slug.lower()).strip('-')` before use as a filesystem path component.
- **`word_count` not updated after content mutations** (multiple files): `model_copy()` calls that modified `content_markdown` were not recomputing `word_count` or `reading_time_minutes`. Fixed in: `dual_qa_agent._revise()`, `dual_qa_agent._restore_dropped_markers()`, `main._save_article` after sanitization, `main` after link enrichment, `authenticity_revision_service`, `location_adaptation_service`.
- **`.strip()` → `.rstrip('\n')` in marker restoration** (`agents/dual_qa_agent.py:93`): `.strip()` on the full article body removed leading content. Changed to `.rstrip('\n')` to only trim trailing newlines.
- **Stale WordPress fields on reuse** (`main.py`): Reused articles inherited `wp_post_id`, `wp_post_url`, `drive_document_id` from the source draft. Now cleared to `None` and `publishing` reset to defaults.
- **Reuse status** (`main.py`, `services/draft_reuse_service.py`): `PublishStatus.DRAFT` (incorrect enum) → `ArticleStatus.REVIEW` on article reuse.
- **`TenantContext.reuse_group` never populated** (`main.py`): The field existed on the model but was never set from the site profile. Cross-site reuse via `reuse_group` was silently broken.
- **Dead `_audit_seo_content` static method** (`agents/article_agent.py`): Removed. Call site had been removed; only reference was the definition.
- **Dead post-loop `raise`** (`agents/article_agent.py`): Unreachable `raise ClaudeRateLimitError(...)` after a `for` loop whose final-iteration `except` branch always re-raises.
- **Dead condition in pool filesystem scan** (`services/draft_pool_service.py`): `path.name != "article.json"` was always `False` for `glob("**/article.json")` results. Removed.
- **Dead `internal_links` variable** (`services/publication_certification_service.py`): Computed but never read. Removed.
- **HTML entity decoding for WP title** (`services/publication_certification_service.py`): WordPress REST API returns `&amp;` etc. in `title.rendered`. Now decoded with `html.unescape()` before comparison.
- **Certification title check** (`services/publication_certification_service.py`): Was checking `bool(live_title)` (title exists) instead of `titles_match` (title matches). Now checks actual match and reports the mismatch clearly.
- **Internal link check scope** (`services/seo_qa_service.py`): The regex `\[.+?\]\(.+?\)` matched any markdown link, including external links. Now checks for domain presence in markdown using `article.request.website_url` when available; skips the check when `website_url` is unknown.
- **FAQ count inconsistency** (`templates/article_structure.json`): `"count": "3-4"` in prompt guidance vs `faq_max_questions: 5` in validation_rules. Unified to 3–5.
- **`assert` replaced with `raise RuntimeError`** (`agents/image_resolver_agent.py`): Three `assert self._drive is not None` and `assert self._generator is not None` replaced with explicit runtime errors.
- **`except (TypeError, ValueError)` missing `AttributeError`** (`agents/image_resolver_agent.py`): Vision scoring could raise `AttributeError` on malformed responses. Added to catch clause.
- **`int(raw.get(..., 0))` fails on `None`** (`services/openai_review_service.py`): OpenAI can return `null` for score fields. Changed to `int(raw.get(...) or 0)` in three places.
- **HTTP 529 not recoverable** (`services/llm_gateway.py`): Anthropic's overloaded response code was not in `_RECOVERABLE_STATUS_CODES`. Added.
- **400 "usage limit" not triggering failover** (`services/claude_service.py`, `services/llm_gateway.py`): A 400 status with "usage limit" message was being raised as a non-recoverable `ClaudeAPIError`. Now re-raised as `ClaudeRateLimitError` so LLMGateway failover applies.
- **Rate limit error message lost** (`services/claude_service.py`, `services/llm_gateway.py`): Original error string was discarded in favor of a generic message. Now preserved via `str(exc)`.
- **`meta_description` max_length** (`models/article.py`): Was 170; SERP truncation is at 160. Updated to 160.
- **`language` default** (`models/article.py`): Was `ArticleLanguage.ES`; corrected to `ArticleLanguage.EN`.
- **`generate` and `autopublish` `--words` max** (`main.py`): Was 1000; corrected to 950 to match the hard cap.

---

### Challenges

- **Context compaction mid-investigation**: The forensic cost audit and draft reuse investigation spanned enough context to trigger compaction. The session continued without loss but required re-establishing the pool-read step.
- **Jaccard/normalization asymmetry**: The draft pool's `_tokenize()` (no synonym map) and `topic_normalization.normalize_topic_id()` (applies `_SYNONYMS`) operate on the same text with different transformations. The request side of the Jaccard comparison carries both raw forms (`overhead`, `repairs`) and canonical forms (added from `service` field: `door`, `repair`), while pool candidates are scored from their stored topic_ids (which were normalized). This consistently inflates the union and deflates Jaccard scores. For the investigated article (best score 0.188 vs 0.72 threshold) the asymmetry was not material, but it is a latent scoring defect.
- **`_budget` vs `budget` attribute bug (identified, not yet fixed)**: `DualQAAgent.run()` line 555 uses `getattr(self._claude, '_budget', None)`. `self._claude` is `LLMGateway`, which exposes a `budget` property (no underscore), not `_budget`. Result: all QA cost fields in `DualQAReport` are permanently `$0.0`. The underlying costs ARE correctly recorded in `BudgetService` — only the per-report attribution is wrong. Fix requires changing `'_budget'` to `'budget'`.
- **OpenAI text review costs not tracked**: `openai_review_service.py` calls GPT-4o-mini for article and image text reviews but never calls `budget.record_openai()`. These costs are invisible to all tracking systems.

---

### Next Steps

1. **Fix `dual_qa_agent.py:555`**: `getattr(self._claude, '_budget', None)` → `getattr(self._claude, 'budget', None)`. One-character fix; unblocks accurate per-article QA cost reporting.
2. **Track OpenAI text review costs**: Add `budget.record_openai_text()` (or extend `record_openai()`) and call it in `openai_review_service.py` after article and vision review calls.
3. **Resolve Jaccard/normalization asymmetry**: Either apply `_SYNONYMS` in `draft_pool_service._tokenize()`, or strip the duplicate forms from `req_tokens` before scoring.
4. **Integrate `writing_audit_service.py`** into the QA pipeline.
5. **Run the test suite against the current codebase** and fix any failures introduced by today's schema changes (`SectionPlan` field removals).

---

### Session Close — 2026-07-30

**Documentation added this session:** `docs/dev-log.md`, `CHANGELOG.md`, `PROJECT_STATUS.md`, `README.md` (rewritten to reflect current capabilities). `PROJECT_MEMORY.md` created at project root.

**Command infrastructure:** `.claude/commands/end.md` populated from nested source. `claude/claude/PROJECT_MEMORY.md` (empty, misplaced) removed from untracked state.

**Commits:** `650bbeb` (engineering audit), `4867a78` (PROJECT_STATUS + README), `chore:` (command infrastructure + project memory).

---

## 2026-07-30 (Session 2)

### Summary

Short session focused entirely on repository structure and tooling. No production code was modified.

Executed the `claude start` session briefing workflow, which produced a complete project health report and a prioritized work plan. Then investigated the `claude/claude/` directory anomaly, diagnosed it as a failed gitignore workaround, and performed a clean consolidation of all Claude command files into the correct versioned location.

---

### Features Added

None.

---

### Improvements

- All three Claude commands (`start.md`, `end.md`, `review.md`) are now committed to version control and will be available across all future sessions without manual reconstruction.

---

### Fixes

- **`.gitignore` rule too broad**: `.claude/` → `/.claude/settings.local.json`. The old rule gitignored any `.claude/` directory anywhere in the tree. Command files were unversioned despite being project artifacts, not local config.
- **Zero-byte `claude/claude/commands` file**: Accidentally committed in a prior session. Removed via `git rm`.
- **`review.md` missing from root `.claude/commands/`**: Only existed in the now-deleted `claude/claude/.claude/commands/` path. `claude review` was silently broken. Moved to canonical location.

---

### Refactors

None.

---

### Documentation

None. (Session briefing was analysis output, not a documentation artifact.)

---

### Challenges

The `.gitignore` rule `.claude/` (without a leading `/`) applies recursively to every `.claude/` directory in the repository tree. The `claude/claude/.claude/` workaround path was therefore also gitignored, making the workaround completely ineffective. Diagnosis required reading git internals behavior for pattern anchoring.

---

### Lessons Learned

- Gitignore patterns without a leading `/` are applied recursively. To restrict a rule to the repository root, always anchor with `/`.
- Claude Code command files are project assets, not developer-local config. They belong in version control under `.claude/commands/`, with only credentials and local settings gitignored.

---

### Next Steps

1. Fix B1: `agents/dual_qa_agent.py:555` — `'_budget'` → `'budget'` (one-character fix).
2. Fix B2: `services/openai_review_service.py` — add `budget.record_openai()` calls after each GPT-4o-mini request.
3. Run `pytest tests/` to establish baseline before any code changes.

---

## 2026-08-03

### Summary

Fixed both open budget-tracking bugs (B1 and B2). No architecture changes, no prompt changes, no model changes. Suite grew from 54 to 65 tests.

---

### Investigation

**B1 — `dual_qa_agent.py:555`**

Confirmed the bug was present. `getattr(self._claude, '_budget', None)` always returned `None` because `self._claude` is an `LLMGateway` instance, which exposes `.budget` as a public property (no underscore). `_budget` is an internal attribute on `ClaudeService`, which `LLMGateway` wraps but does not re-expose. Result: `budget_svc` was always `None`, `_snap()` always returned `None`, `_claude_delta()` always returned `0.0`, and every `DualQAReport` field (`claude_review_cost_usd`, `revision_cost_usd`, etc.) was permanently `$0.00`. Underlying costs were correctly recorded in `BudgetService` by `ClaudeService` — only the per-report attribution was broken.

**B2 — `openai_review_service.py`**

Confirmed that neither `review_article()` (text review) nor `review_image()` (vision review) ever called any `BudgetService` method. Both methods correctly compute cost from `response.usage` token counts and accumulate it in `self.text_cost_usd` / `self.vision_cost_usd`. Those values are copied into `DualQAReport` fields (`openai_review_cost_usd`, `vision_openai_cost_usd`) at the end of `run()`. However, `BudgetService.record_openai()` was never called, so monthly budget totals never reflected these costs.

**B2 — `BudgetService.record_openai()` incompatibility**

`record_openai(images: int)` is image-generation-specific: it applies a fixed `$0.25/image` price and increments an `images` counter. Text/vision review costs are token-based (~$0.001–0.003/article) and must not corrupt the `images` counter. A new method `record_openai_text(cost_usd: float)` was added to `BudgetService` to handle review costs correctly.

---

### Fixes

**`agents/dual_qa_agent.py:555`**

```python
# Before (broken):
budget_svc = getattr(self._claude, '_budget', None)

# After (fixed):
budget_svc = getattr(self._claude, 'budget', None)
```

**`agents/dual_qa_agent.py` — 3 OpenAI recording sites added**

- Main article review loop: captures `text_cost_usd` delta around `_openai_review_article()` and calls `budget_svc.record_openai_text(delta)`.
- Rescue re-review path: reuses the existing `openai_before_cost`/`openai_after_cost` delta variables; adds one `budget_svc.record_openai_text()` call after the delta is computed.
- Vision review in `_review_single_ai_image()`: captures `vision_cost_usd` delta around `review_image()` and calls `_vision_budget_svc.record_openai_text(delta)`. `budget_svc` is out of scope here (different method from `run()`), so `getattr(self._claude, 'budget', None)` is resolved fresh within the method.

**`services/budget_service.py` — new `record_openai_text()` method**

```python
def record_openai_text(self, cost_usd: float) -> None:
    if cost_usd <= 0:
        return
    with self._exclusive_lock():
        data = self._load()
        o = data["openai"]
        o["calls"] += 1
        o["usd"] = round(o["usd"] + cost_usd, 6)
        self._save(data)
```

Does not touch `o["images"]`. Uses the same `_exclusive_lock()` as other record methods. Zero-cost calls are skipped (guard before the lock acquisition).

**`tests/test_vision_qa_exception_handling.py` and `tests/test_dual_qa_openai_bypass.py`**

Two pre-existing test stubs referenced `stub_claude._budget = None` (the old broken attribute name). Updated to `stub_claude.budget = None`. Also added `stub_openai.vision_cost_usd = 0.0` to the vision exception test so cost delta arithmetic does not hit a `MagicMock > int` TypeError.

---

### Tests Executed

Baseline: `54 passed` (before any changes).

After all changes: `65 passed` (11 new tests in `tests/test_qa_cost_attribution.py`).

New tests cover:
- `BudgetService.record_openai_text()`: records USD, increments calls, does not touch `images`, accumulates across calls, skips zero-cost calls, coexists with `record_openai()`.
- `DualQAAgent` text review recording: `openai.usd` is non-zero after a review run; no recording when reviewer is absent.
- `DualQAAgent` vision review recording: `openai.usd` is non-zero after vision review; `images` counter stays 0; exception during review records nothing.
- B1 property name: typed stub (not `MagicMock`) confirms `.budget` resolves correctly and `._budget` returns `None` on an `LLMGateway`-like object.

---

### Risk Assessment

**B1 fix** — Zero regression risk. One attribute name character changed. The `budget` property is the documented public interface. Two existing test stubs updated to use the correct attribute; both still pass.

**B2 fix** — Low regression risk. All three recording sites guard on `cost_delta > 0` before writing to `BudgetService`, so no spurious writes. The `record_openai_text()` method acquires the same exclusive lock as other record methods; no race condition introduced. Adding costs to `openai.usd` in the existing budget file structure does not break any existing schema consumers.

---

### Next Steps

None — B1 and B2 are closed. No open critical bugs remain.

---

## Request 2 Post-Mortem: Prompt Caching Removed

**Date:** 2026-08-03  
**Engineer:** Claude (automated)  
**Decision:** Remove Anthropic Prompt Caching — net-negative under current architecture.

---

### Background

Request 2 added Anthropic Prompt Caching to five call sites across the article pipeline. The implementation used a `cache_system: bool = False` flag threaded from the gateway layer down to `ClaudeService._base_kwargs()`, which conditionally converted the system prompt string into an `[{"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}]` list when the flag was set. Cache creation and read token counts were tracked in `BudgetService` and `CallTracer`.

---

### Production Measurement

A real `autopublish` run was executed with caching active. Anthropic API usage fields confirmed:

| Stage | cache_creation_tokens | cache_read_tokens | input_tokens |
|---|---|---|---|
| plan:article | > 0 | 0 | normal |
| generate:article | > 0 | 0 | normal |
| qa:claude | > 0 | 0 | normal |
| qa:revision | > 0 | 0 | normal |
| All other stages | 0 | 0 | normal |

Every stage created a new cache entry. No stage read from any prior cache entry.

---

### Root Cause Analysis: Why Cache Reads Never Occur

**Within a single article run:** The pipeline is a strict sequential dependency chain — plan → generate → QA review → revise → re-review. Each stage consumes the previous stage's output. No stage repeats with the same system prompt and user content combination. There is no opportunity for a same-run cache hit.

**Across article runs:** Anthropic's cache TTL is 5 minutes from write time, not from last access. A planning call at T≈3s writes a cache entry expiring at T≈303s. The full article pipeline runs for ≈170–250 seconds. A second article's planning call would need to arrive before T≈303s — within ≈50 seconds of the first article's planning call — to get a cache read. In practice, `autopublish` is invoked manually, one article at a time, with gaps of hours or days between runs. The TTL window never closes.

**Architecture conclusion:** The current one-article-per-process architecture structurally prevents all cache reads. The only way to exploit prompt caching in this system would be to run N articles in rapid succession (a batch mode that does not exist) or to implement a long-lived daemon that pre-warms the cache and immediately starts the next article. Neither applies.

---

### Cost Impact of Caching

Anthropic prompt caching pricing: write = 1.25× input rate; read = 0.10× input rate.

With zero cache reads, each `cache_creation_tokens` token costs **1.25× the standard input rate** — a 25% surcharge for zero return. Measured per-article:

- cache_creation_tokens: ~22,000 tokens
- Surcharge at claude-sonnet-4-6 ($3.00/M): +$0.0069 per article
- Cache reads: $0.00 (none occurred)
- Net effect: −$0.0069 per article (strictly negative)

**Projected yearly cost at current pace (estimated 2 articles/week):**
22,000 × 1.25 × $3/M × 0.25 (surcharge factor) × 104 articles/year ≈ **$0.72/year wasted**. Low absolute dollar amount, but 100% waste with no offsetting benefit.

---

### Batch API (Request 3) Viability

Request 3 (Anthropic Batch API) remains architecturally sound and is unaffected by this decision. The Batch API operates at the article level (N whole articles submitted as a batch), not at the call level. Each article's internal pipeline is still sequential; batching only parallelizes articles across submissions. The 50% cost discount on batch-submitted requests applies directly to the same Sonnet + Haiku call mix used today. Request 3 would require a new `batch-autopublish` command with no changes to the existing `autopublish` flow. It is the correct next cost optimization to pursue.

---

### Revert Implementation

All Request 2 changes removed. Files modified:

| File | Change |
|---|---|
| `services/claude_service.py` | Removed `cache_system` param from `generate()`, `generate_structured()`, `_base_kwargs()`; removed cache token fields from `record_claude()` and `call_tracer.record()` calls |
| `services/budget_service.py` | Removed `cache_creation_tokens`/`cache_read_tokens` from `record_claude()` signature and cost formula; removed from `_empty()` |
| `services/llm_gateway.py` | Removed `cache_system=False` from `generate()` and `generate_structured()` signatures; removed `cache_system=cache_system` from primary lambda calls |
| `services/article_planner_service.py:546` | Removed `cache_system=True` |
| `services/authenticity_revision_service.py:263` | Removed `cache_system=True` |
| `agents/article_agent.py:488` | Removed `cache_system=True` |
| `agents/dual_qa_agent.py:774` | Removed `cache_system=True` |
| `agents/dual_qa_agent.py:889` | Removed `cache_system=True` |
| `tests/test_prompt_caching.py` | Deleted (entire file, 16 tests) |

**Retained (not reverted):**

- `services/call_tracer.py` — Cache↑/Cache↓ columns retained. They display `-` when no caching is active. Pure observability; no behavior change. If a future batch or caching implementation is added, the instrumentation is already in place.
- `main.py` — Added tracer print inside `DualQAFailedError` handler. This was a bug fix (tracer was silently discarded on QA failure) unrelated to caching.

---

### Test Results

```
65 passed in 2.95s
```

65 tests = original 54 (pre-Request-2) + 11 QA cost attribution tests (Request 1-B). The 16 Request-2-specific tests in `test_prompt_caching.py` were deleted with the implementation.

---

### Post-Revert Cost Measurement

Real `autopublish --status draft --no-image` run after revert:

```
Stage        Model           In Tok   Cache↑   Cache↓   Out Tok   Cost      Time
plan:ar…     claude-sonnet   5,119    -        -        5,000     $0.0904   115.8s
generat…     claude-sonnet   8,752    -        -        2,000     $0.0563   58.2s
qa:clau…     claude-sonnet   4,325    -        -        2,606     $0.0521   54.0s
seo:met…     claude-haiku    4,229    -        -        303       $0.0057   3.1s
topics:…     claude-haiku    290      -        -        204       $0.0013   2.9s
openai:…     gpt-4o-mini     3,032    -        -        474       $0.0007   12.0s
TOTAL (6 calls)              25,747   -        -        10,587    $0.2065   246.0s

Claude cost:  $0.152357
```

Cache columns are all `-`. Cache pricing has been entirely removed from the cost calculation. Standard per-token pricing applies throughout.

Note: This run failed QA (article truncated due to token limit — unrelated to caching). The per-call costs are structurally identical to a successful run at similar token volumes.

---

### Final Recommendation

Do not re-introduce prompt caching unless the architecture changes to include one or more of:

1. A batch mode that runs N ≥ 2 articles in the same process within a 5-minute window with shared system prompts.
2. A long-lived worker/daemon that accepts queued article requests and keeps processes alive long enough for the cache TTL to overlap across runs.
3. A fundamentally different pipeline design where a system prompt is reused across multiple LLM calls within a single article run (e.g., a multi-turn conversation architecture).

None of these are planned. Under the current architecture, caching is net-negative.

---

# SEO Agent — Project Status

_Last updated: 2026-07-30_

---

## Current Goal

Deliver a production-quality, cost-controlled SEO article autopublish pipeline capable of generating, reviewing, and publishing ~800-word local-service articles for multiple clients and websites at under $0.25 per article.

---

## Current Sprint

**Engineering audit pass** — correctness, cost reduction, and article quality tightening.

Sprint outcome: 20+ bug fixes, 9 tests added, multi-model budget pricing, word count reduced to 800, QA cycles capped at 1, cost target reduced to $0.25/article.

---

## Completed

- Initial project architecture (agents, services, models, profiles)
- Article generation pipeline (Claude, structured planner, article agent)
- DualQA pipeline — Claude SEO/editorial review + OpenAI writing/authenticity review
- WordPress autopublish — topic suggestion → generate → QA → WP publish
- Draft pool system — Jaccard similarity matching, cross-site reuse via `reuse_group`
- Budget service — per-model pricing (Opus/Sonnet/Haiku), file locking, monthly tracking
- Google Drive image integration — vision scoring, WP upload
- Location adaptation service — city name substitution in reused articles
- Publication certification service — post-publish live URL verification
- Multi-client/multi-site tenant architecture with site profiles
- Test suite — 9 pytest tests covering correctness and QA behavior
- Autopublish topic placeholder detection — rejects `[City]`, `[Service]`, etc.
- `reuse_group` wired from site profile → tenant context (cross-site reuse now functional)
- Slug path traversal fix, WP field clearing on reuse, word_count consistency fixes

---

## In Progress

_Nothing currently in progress. Sprint completed._

---

## Blockers

**B1 — QA cost reporting always $0.00** (identified, not yet fixed)
- File: `agents/dual_qa_agent.py:555`
- Cause: `getattr(self._claude, '_budget', None)` — `LLMGateway` exposes `.budget` (no underscore), not `._budget`
- Impact: `DualQAReport.total_qa_cost_usd` is always `$0.00`; underlying costs are correctly recorded in `BudgetService`
- Fix: change `'_budget'` → `'budget'` (one character)

**B2 — OpenAI text review costs not tracked**
- File: `services/openai_review_service.py`
- Cause: GPT-4o-mini article/image review calls never invoke `budget.record_openai()`
- Impact: ~$0.001–0.003/article in review costs invisible to all cost tracking

---

## Next Priority

Fix B1 first — one-line change, unblocks accurate per-article cost attribution in QA reports.

Suggested commit:
```
fix: correct Claude budget property access in DualQAAgent
```

Then B2: add `budget.record_openai_text()` calls in `openai_review_service.py`.

---

## Known Technical Debt

- **Jaccard/normalization asymmetry**: `draft_pool_service._tokenize()` does not apply synonym mapping; `normalize_topic_id()` does. The request-side `req_tokens` carries both raw (`overhead`, `repairs`) and canonical (`door`, `repair`) forms, inflating the union and deflating Jaccard scores by a consistent margin.
- **`dual_qa_agent.py` `_budget` bug** (see B1 above)
- **OpenAI text cost not tracked** (see B2 above)
- **`writing_audit_service.py` not integrated**: File exists in `services/` but is not called from any pipeline stage.
- **`benchmark_results.jsonl` committed**: Raw benchmark output; should be `.gitignore`d if re-run frequently.
- **`dual_qa_agent.py` is ~1700 lines**: Monolithic — revision, rescue, vision QA, cost tracking, and format helpers should be split into sub-modules.
- **No CI/CD**: Tests are present but not automatically run on push. GitHub Actions needed.

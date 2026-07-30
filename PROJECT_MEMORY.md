# SEO Agent — Project Memory

Long-term engineering decisions. Updated at session end.
Daily work lives in `docs/dev-log.md`.

---

## Architecture Decisions

### Dual QA threshold design
Two SEO score thresholds exist intentionally and must never be merged:
- `qa_min_seo = 90` — pre-publish DualQA gate. Claude scores the draft. Stricter because it drives revision cycles that cost money.
- `seo_qa_min_score = 70` — post-publish certification gate. Checks the live URL after WordPress publishes. Informational only; the article is already live.

### Model routing strategy
Each pipeline stage has an independently configurable model. Do not collapse to a single model setting.
- Generation: Sonnet (instruction-following + cost balance)
- Planning: Sonnet (structured schema output)
- SEO metadata: Haiku (template-like, deterministic)
- Image eval: Haiku (scoring, not reasoning)
- QA review: Sonnet (rubric application)
- Topic suggestion: Sonnet

### Draft pool Jaccard threshold
`SIMILARITY_THRESHOLD = 0.72` in `draft_pool_service.py`. This value was validated empirically. Do not lower it — false positives (reusing semantically different articles) cause client-facing quality failures.

**Known asymmetry**: `_tokenize()` in `draft_pool_service` does not apply synonym mapping, but `normalize_topic_id()` does. This consistently deflates Jaccard scores by a small margin. The threshold accounts for this in practice; however, fixing the asymmetry (apply `_SYNONYMS` in `_tokenize`) would make scoring cleaner.

### Word count target
800 words, range 700–900, hard cap 950. This is calibrated to pass SERP quality signals while controlling token costs. Do not increase the target without re-validating QA prompt scoring.

### QA cycle budget
`qa_max_cycles = 1`. Second and third revision cycles were observed to rarely improve passing rates while adding ~$0.10–0.15 per cycle. Authenticity rescue (`qa_rescue_enabled`) is disabled by default for the same reason.

### Cost target
`max_article_cost_usd = $0.25`. This includes generation, planning, SEO metadata, QA, and image ops. Does NOT currently include OpenAI text review costs (bug: untracked).

### Budget file locking
`BudgetService` uses POSIX `fcntl.LOCK_EX` on a `.budget.lock` file. This is correct for single-machine concurrent runs. If the system ever moves to distributed execution, replace with a database-backed counter.

---

## Prompt Engineering Principles

### Planner schema: internal vs output fields
Reader-perspective analysis fields (`reader_intent`, `reader_misconception`, `why_misconception_forms`, `failure_mechanism`) were removed from the output schema in July 2026. They are now numbered internal reasoning steps (1–4) in the planner prompt. The LLM reasons through them before populating `technical_reality` and `professional_insight`. This reduces schema token overhead while producing sharper output fields.

**Do not re-add these as schema fields.** The reason they were removed is that including them in the output forced the article generator to echo the reader-perspective framing into the prose, producing visible "structured thinking" artifacts in the article body.

### Generator voice model
The article generator is prompted to write "as if explaining to an experienced homeowner standing next to the garage door." The paragraph flow modes (Observation → Explanation → Consequence → Practical implication → Example → Warning) should be preserved in any prompt revisions. These modes were introduced to replace mechanical sentence-count instructions that produced stilted prose.

### Focus keyword enforcement
As of July 2026, keyword placement rules are non-negotiable:
1. Exact keyword phrase verbatim in H1
2. Keyword within first 100 words (counted strictly)
3. At least one H2 containing the keyword or a 3+-word consecutive variant

These are QA failure criteria, not suggestions. The QA prompt enforces them.

### External links
External links are required, not optional. An article with zero external links fails QA. Acceptable sources: CPSC, OSHA, DOE, DASMA, manufacturer associations, university extensions.

---

## QA Methodology

### DualQA design
Claude reviews SEO and editorial quality. OpenAI (GPT-4o-mini) reviews writing quality and authenticity (AI-detection proxy). Both gates must pass independently. This split exists because:
- Claude is the generator; using Claude as the sole reviewer creates conflict-of-interest (it scores its own output)
- OpenAI provides an independent signal on human-writing quality and AI artifact detection

### Internal links in QA scoring
When a site profile does not configure `internal_links_to_include`, the internal links criterion is excluded from QA scoring (N/A, not a failure). The QA review is told via the `INTERNAL LINKS CONFIGURED: No` header. Do not remove this mechanism — scoring internal links as a failure when they were never configured is a false negative.

### FAQ format
3–4 FAQ questions is the correct production format for 800-word articles. QA rubrics must not penalize 3–4 questions as "too few." The previous standard of 5 questions was retired with the word count reduction.

---

## SEO Validation Rules

### Slug sanitization
`article.seo.slug` must be sanitized before use as a filesystem path: `re.sub(r'[^a-z0-9-]', '-', slug.lower()).strip('-')`. The raw slug from Claude may contain uppercase, spaces, or special characters. This was a path traversal vulnerability before the fix.

### Meta description limit
Hard limit: 160 characters (not 170). SERP truncation occurs at 160. The Pydantic model enforces `max_length=160`.

---

## Cost Optimization History

| Date | Change | Impact |
|---|---|---|
| 2026-07-30 | Default model: Opus → Sonnet | ~40% generation cost reduction |
| 2026-07-30 | SEO/image models: Sonnet → Haiku | ~$0.02/article reduction |
| 2026-07-30 | `qa_max_cycles`: 3 → 1 | Eliminates ~$0.10–0.15 revision overhead |
| 2026-07-30 | Planner `thinking`: True → False | Eliminates thinking token overhead on structured output |
| 2026-07-30 | Word count: 850 → 800 | ~6% output token reduction |

---

## Known Open Bugs

### B1 — QA cost attribution (dual_qa_agent.py:555)
```python
# Current (broken):
budget_svc = getattr(self._claude, '_budget', None)
# Fix:
budget_svc = getattr(self._claude, 'budget', None)
```
`LLMGateway` exposes `.budget` property (no underscore). The `_budget` attribute exists on `ClaudeService`, not on `LLMGateway`. Result: all QA cost fields in `DualQAReport` are `$0.00`.

### B2 — OpenAI text review costs not tracked
`openai_review_service.py` never calls `budget.record_openai()`. GPT-4o-mini article and image text reviews are invisible to all cost tracking systems.

---

## Tooling Decisions

### Claude command files belong in version control
`.claude/commands/` contains session workflow definitions (`start.md`, `end.md`, `review.md`). These are project assets, not developer-local configuration. The `.gitignore` rule must be `/.claude/settings.local.json` (anchored, file-specific) — never `.claude/` (recursive, would gitignore the entire commands directory).

**Do not revert this rule.** A `.claude/` gitignore pattern applies to any directory named `.claude/` anywhere in the repository tree, not just the root.

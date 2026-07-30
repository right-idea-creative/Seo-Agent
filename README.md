# SEO Agent

> Production-ready pipeline for generating, reviewing, and publishing local-service SEO articles at scale.

![Status](https://img.shields.io/badge/status-active%20development-orange)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

SEO Agent is a multi-client, multi-site article generation and publishing pipeline built on Claude and OpenAI. It produces ~800-word, locally-targeted, SEO-optimized articles and publishes them directly to WordPress — with full QA, image resolution, budget tracking, and draft reuse across sites.

**Target cost:** Under $0.25 per article end-to-end.

---

## Current Capabilities

### Article Generation

- **AI planning**: Claude generates a structured technical reasoning plan (expert knowledge, local factors, counter-intuitive facts, specific numbers) before writing.
- **AI writing**: Claude writes an 800-word article from the plan with strict voice, paragraph flow, and keyword placement rules.
- **Focus keyword enforcement**: Exact match required in H1, within first 100 words, and in at least one H2.
- **External links**: At least one authoritative external link required per article.

### Quality Assurance

- **DualQA pipeline**: Claude scores SEO/editorial; OpenAI (GPT-4o-mini) scores writing quality and authenticity. Both gates must pass.
- **Automatic revision**: Up to 1 revision cycle with targeted feedback; SEO metadata regenerated mid-cycle if score is failing.
- **Vision QA**: OpenAI vision reviews each placed image for quality and relevance.
- **Content sanitization**: Removes UI artifacts before any pipeline stage.

### Publishing

- **WordPress autopublish**: Full end-to-end from topic selection to live post. Supports draft and publish status.
- **Google Drive image resolution**: Claude Vision scores candidate images from a Drive folder; winner is uploaded to WordPress Media Library.
- **Internal link enrichment**: Injects contextual internal links from existing WP posts.
- **Post-publish certification**: Verifies the live URL against expected title, content, SEO score, and image presence.

### Draft Reuse

- **Draft pool**: Persistent JSON index of all generated articles. O(1) exact topic match; Jaccard similarity fallback for near-matches.
- **Location adaptation**: Reused articles have city/state substituted for the target market.
- **Cross-site reuse**: Sites sharing a `reuse_group` in their profile may reuse each other's drafts.
- **Safety gates**: Location compatibility, client isolation, and published-article exclusion enforced before any reuse.

### Budget Tracking

- **Per-model pricing**: Separate rates for Opus, Sonnet, and Haiku; all Claude calls priced at the actual model used.
- **Monthly budget**: Configurable hard cap across Claude + OpenAI spend; pipeline stops when reached.
- **Per-article cost target**: Warning printed when cost exceeds `max_article_cost_usd` (default: $0.25).
- **File-locked writes**: POSIX exclusive lock on budget JSON prevents corruption under concurrent runs.

### Multi-Client Architecture

- **Site profiles**: Per-client, per-website JSON configs with business name, service, city, WordPress credentials, and Drive settings.
- **Tenant isolation**: All articles, budgets, and output are namespaced by `client_id / website_id`.
- **Model routing**: Each pipeline stage (generation, planning, SEO, QA, image eval) has an independently configurable model.

---

## Project Structure

```
seo-agent/
│
├── agents/                  # AI agents (article, QA, image resolver)
├── budget/                  # Monthly budget JSON files
├── docs/                    # Development log
├── models/                  # Pydantic data models
├── output/articles/         # Generated article JSON + media
├── profiles/                # Per-client site profiles
│   └── {client}/{website}/site.json
├── services/                # Core service layer
├── templates/               # Article structure spec
├── tests/                   # pytest test suite
│
├── config.py                # Settings (Pydantic BaseSettings, .env)
├── main.py                  # CLI entry point (Typer)
└── requirements.txt
```

---

## Installation

```bash
# Python 3.12+ required
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Set: ANTHROPIC_API_KEY, OPENAI_API_KEY, and per-site WP/Drive credentials
```

### Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API (generation, QA, planning) |
| `openai` | GPT-4o-mini article/image QA |
| `typer` | CLI framework |
| `rich` | Terminal output formatting |
| `pydantic` / `pydantic-settings` | Data models and config |
| `google-api-python-client` | Google Drive integration |
| `openpyxl` | Excel export support |
| `requests` | WordPress REST API |
| `pytest` | Test suite |

---

## Usage

```bash
# Generate a single article (saved locally, no publish)
python main.py generate --client RIMC --website overheaddoornwi \
  --topic "Overhead Door Spring Replacement" --city "Northwest Indiana" --state IN

# Full autopublish (topic → generate → QA → WordPress)
python main.py autopublish --client RIMC --website overheaddoornwi

# Suggest article topics
python main.py suggest --client RIMC --website overheaddoornwi --service "Overhead Door Repair"

# Publish a previously generated article
python main.py publish --client RIMC --website overheaddoornwi \
  --article output/articles/RIMC/overheaddoornwi/article-slug/article.json
```

---

## Configuration

All settings live in `config.py` and are overridable via `.env`.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `claude_model` | `claude-sonnet-4-6` | Default generation model |
| `default_word_count` | `800` | Target article word count |
| `max_article_cost_usd` | `$0.25` | Per-article cost warning threshold |
| `claude_monthly_budget_usd` | configurable | Hard monthly Claude spend cap |
| `qa_max_cycles` | `1` | Maximum QA revision cycles |
| `qa_rescue_enabled` | `False` | Enable authenticity rescue rewrite (~$0.11 extra) |
| `seo_qa_min_score` | `70` | Post-publish certification SEO gate |
| `qa_min_seo` | `90` | Pre-publish DualQA SEO gate (stricter) |

---

## Roadmap

### Phase 1 — Foundation ✅

- [x] Project architecture
- [x] Agent framework
- [x] Site profile system
- [x] Service layer

### Phase 2 — Content Generation ✅

- [x] AI article generation with structured planning
- [x] Local SEO targeting
- [x] Focus keyword enforcement
- [x] FAQ and external link requirements
- [x] Draft pool and reuse system

### Phase 3 — Integrations ✅

- [x] WordPress REST API (publish, update, media upload)
- [x] Google Drive image resolution and Vision QA
- [x] DualQA pipeline (Claude + OpenAI)
- [x] Post-publish certification

### Phase 4 — Cost Control ✅

- [x] Per-model budget tracking
- [x] Monthly hard cap
- [x] Budget file locking
- [x] Per-article cost target and warnings

### Phase 5 — Quality and Scale

- [ ] Fix QA cost attribution bug (`dual_qa_agent.py:555`)
- [ ] Track OpenAI text review costs
- [ ] Integrate `writing_audit_service.py` into pipeline
- [ ] Google Search Console integration
- [ ] Keyword clustering and topic planning
- [ ] Automated link building workflows

---

## Documentation

| Document | Description |
|---|---|
| [`docs/dev-log.md`](docs/dev-log.md) | Engineering session log |
| [`CHANGELOG.md`](CHANGELOG.md) | Feature and fix history |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Live sprint status and blockers |

---

## License

This project is licensed under the MIT License.

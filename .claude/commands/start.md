# Start Development Session

When I run:

claude start

you will prepare the development environment before any coding begins.

You are acting as:

- Senior Software Engineer
- Technical Lead
- Project Analyst

Your objective is to give me a complete, accurate picture of the project state so I can begin today's session with full context and a clear plan.

---

# RULES

Never modify code.

Never create commits.

Never update documentation.

Never invent work that was not completed.

Never guess at project state — derive everything from the repository.

Only analyze, summarize, and prepare.

---

# STEP 1 — Read project documentation

Read the following files in full:

1. `CLAUDE.md` (if it exists)
2. `README.md`
3. `PROJECT_STATUS.md`
4. `PROJECT_MEMORY.md`
5. `CHANGELOG.md`
6. Latest entry in `docs/dev-log.md`

Extract:

- What this project does
- Current goal
- Current sprint
- Known blockers
- Known bugs
- Long-term engineering decisions that must be respected

---

# STEP 2 — Inspect the repository

Run the following:

- `git status`
- `git branch`
- `git log --oneline -10`
- `git diff HEAD`
- `git stash list`

Report:

- Current branch
- Working tree status (clean / dirty)
- Uncommitted changes (list files and what changed)
- Last 10 commits
- Any stashed work

Warn immediately if:

- Working tree is dirty and contains uncommitted work
- You are not on the expected branch
- There are stashed changes that may affect today's work
- There are merge conflicts

---

# STEP 3 — Analyze project health

Based on what you have read:

Assess each dimension:

**Code health**
- Known bugs open
- Dead code or dormant services
- Files that need attention

**Test health**
- Tests exist?
- Were tests verified recently?
- CI/CD configured?

**Documentation health**
- All core files present and current?
- Inconsistencies between files?

**Cost health**
- What is the current monthly spend?
- Are there untracked costs?

**Architecture health**
- Any structural concerns?
- Technical debt that blocks new work?

---

# STEP 4 — Identify documentation inconsistencies

Compare:

README → CHANGELOG → PROJECT_STATUS → PROJECT_MEMORY → docs/dev-log

Report any conflicts or contradictions between these files.

---

# STEP 5 — Produce session briefing

Output the following structured report. Be factual and concise.

---

## Session Briefing — YYYY-MM-DD

### Project
[One-sentence description of what this project does]

### Current Goal
[From PROJECT_STATUS.md]

### Repository State
[Clean / Dirty — details]

### Last Session Summary
[What was accomplished in the most recent dev-log entry]

### Open Bugs
[List with file, line number, and fix description — sourced from PROJECT_MEMORY.md and PROJECT_STATUS.md]

### Current Blockers
[From PROJECT_STATUS.md — only real blockers, not aspirational tasks]

### Technical Debt
[Summary of highest-risk debt items]

### Documentation Health
[Consistent / Inconsistencies found — list any conflicts]

### Recommended Work Plan

For each task:

**Task:** [Name]
**Priority:** Critical / High / Medium / Low
**File(s):** [Specific files to modify]
**Effort:** [Realistic estimate]
**Why now:** [Reason this is the right task to start with]
**Suggested commit:** `[conventional commit message]`

Order tasks from highest to lowest priority.

### What NOT to do today
[List work that is tempting but should be deferred — over-engineering, premature optimization, low-priority cleanup]

### Session Risk
[Any risks that could derail today's session — API rate limits, large refactors, test suite state]

---

# STEP 6 — Final readiness check

Before ending the briefing, confirm:

✅ or ❌ Working tree is clean
✅ or ❌ On correct branch
✅ or ❌ Documentation is consistent
✅ or ❌ No unresolved merge conflicts
✅ or ❌ Open bugs documented
✅ or ❌ Priority task is clear

If any item is ❌, explain what must be resolved before coding begins.

---

# FINAL RULE

This command exists to prevent wasted sessions.

A session that starts without full context risks:
- Fixing already-fixed bugs
- Duplicating work
- Introducing regressions
- Breaking architectural decisions already made

Your job is to ensure I start every session with complete situational awareness.

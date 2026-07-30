# End Development Session

When I run:

claude end

you will perform the complete end-of-session workflow.

You are acting as:

- Senior Software Engineer
- Technical Writer
- Project Manager
- Git Maintainer

Your objective is to leave the repository in a clean, documented, and production-quality state before ending the session.

---

# STEP 1 — Analyze today's work

Inspect the repository.

Review:

- git diff
- git status
- commits made during the session
- modified files
- added files
- deleted files

Understand what was actually accomplished.

Do not infer work that wasn't completed.

---

# STEP 2 — Generate a development summary

Generate a concise technical summary.

Include:

## Session Summary

- Main objective
- Features implemented
- Improvements
- Bug fixes
- Refactors
- Documentation updates
- Tests added
- Technical debt reduced
- Known issues remaining

Be factual.

Do not exaggerate progress.

---

# STEP 3 — Update docs/dev-log.md

Append today's session.

Never overwrite previous entries.

Use this format:

---

# YYYY-MM-DD

## Summary

...

## Features

...

## Improvements

...

## Fixes

...

## Refactors

...

## Documentation

...

## Challenges

...

## Lessons Learned

...

## Next Steps

...

---

Write in professional technical English.

---

# STEP 4 — Update CHANGELOG.md

Update ONLY if the session introduced:

- new feature
- bug fix
- refactor
- performance improvement
- security improvement

Follow Keep a Changelog format.

Example:

## Unreleased

### Added

...

### Changed

...

### Fixed

...

Never invent releases.

---

# STEP 5 — Update PROJECT_STATUS.md

Maintain a live status of the project.

Structure:

# SEO Agent Status

## Current Goal

...

## Current Sprint

...

## Completed

...

## In Progress

...

## Blockers

...

## Next Priority

...

Keep this file concise.

Remove obsolete items.

Never duplicate the changelog.

---

# STEP 6 — Update README.md (ONLY if necessary)

Modify README ONLY if:

- project capabilities changed
- installation changed
- architecture changed
- dependencies changed

Do NOT update README for:

- typo fixes
- prompt tweaks
- documentation changes
- tests

---

# STEP 7 — Review documentation consistency

Verify:

README

↓

CHANGELOG

↓

PROJECT_STATUS

↓

docs/dev-log

All documentation must be consistent.

If something conflicts, fix it.

---

# STEP 8 — Generate Git Commit

Review all changes.

Generate a Conventional Commit.

Allowed prefixes:

feat:
fix:
refactor:
docs:
perf:
test:
build:
ci:
style:
chore:

Never use:

update

changes

final

work

stuff

The message should describe the most important completed work.

---

# STEP 9 — Git

Execute:

git add .

git commit

git push

If merge conflicts exist:

Resolve safely.

Do not overwrite remote work.

Only continue after the repository is clean.

---

# STEP 10 — Generate Session Metrics

Generate:

Session Metrics

Files modified:

Files added:

Files removed:

Lines added:

Lines removed:

Tests created:

Documentation updated:

Commit:

Branch:

Push:

---

# STEP 11 — Recommend Next Task

Analyze the repository.

Identify the single highest-priority next task.

Include:

Priority

Reason

Estimated effort

Expected impact

Suggested commit message

Example:

Priority:
High

Reason:
QA cost reporting still uses the wrong Claude property.

Estimated effort:
5 minutes

Expected impact:
Accurate QA cost reporting.

Suggested commit:

fix: correct Claude budget property access

---

# Rules

Never fabricate completed work.

Never remove previous development logs.

Never rewrite history.

Prefer updating documentation over creating duplicates.

Use professional engineering language.

Keep documentation concise.

Ensure the repository remains production quality after every session.

Your goal is that every completed work session leaves the repository cleaner, more maintainable, and fully documented than before.
# STEP 12 — Engineering Review

Before ending the session, perform a senior engineering review of the repository.

Analyze the project as if you were the lead engineer responsible for its long-term quality.

Review:

- Architecture
- Code quality
- Technical debt
- Performance
- Maintainability
- Security
- Scalability
- Documentation
- Testing
- AI prompt quality
- Cost efficiency

Identify:

- Potential bugs
- Hidden risks
- Duplicate logic
- Dead code
- Over-engineering
- Missing abstractions
- Missing tests
- Performance bottlenecks
- Security concerns
- Prompt engineering improvements

Do not modify code.

Only report findings.

---

# STEP 13 — Prioritized Engineering Roadmap

Generate the next engineering roadmap.

Classify tasks into:

## Critical

Must be fixed before new features.

## High Priority

Should be completed soon.

## Medium Priority

Useful improvements.

## Nice to Have

Future enhancements.

For every task include:

- Why it matters
- Estimated effort
- Expected impact
- Dependencies
- Suggested Conventional Commit

---

# STEP 14 — Technical Debt Report

Generate a technical debt report.

Classify debt as:

- Architecture
- Code
- Testing
- Documentation
- AI prompts
- Performance

Estimate:

- Risk
- Cost to fix
- Long-term impact

---

# STEP 15 — AI Improvement Opportunities

Because this repository is an AI project, always analyze opportunities to improve:

- Prompt engineering
- Cost reduction
- Token efficiency
- Model quality
- Reliability
- Determinism
- SEO quality
- Human writing quality
- Evaluation metrics
- Benchmark quality

Suggest concrete improvements.

Never invent metrics.

---

# STEP 16 — Release Readiness

Estimate the current maturity of the repository.

Return:

Project Completion:
XX%

Architecture:
X/10

Code Quality:
X/10

Documentation:
X/10

Testing:
X/10

Maintainability:
X/10

Scalability:
X/10

AI Quality:
X/10

Production Readiness:
X/10

Explain every score.

---

# STEP 17 — Project Memory

Update PROJECT_MEMORY.md.

Store only long-term engineering decisions.

Never include daily work.

Include only decisions that future development should remember.

Examples:

- Architectural decisions
- Prompt engineering principles
- QA methodology
- SEO validation rules
- Cost optimization strategy
- Coding standards

Avoid duplicates.

Keep the document concise.

---

# STEP 18 — Repository Health Score

Generate a final repository health report.

Example:

Repository Health

Working Tree:
✅ Clean

Tests:
98 Passed

Documentation:
Up to date

Architecture:
Healthy

Technical Debt:
Low

Critical Issues:
0

Security Issues:
0

Overall Health:
9.4/10

Confidence:
High

---

# FINAL RULE

Act as the Lead Engineer of this repository.

Your responsibility is not only to execute work, but to continuously improve the project.

Every session should leave the repository in a cleaner, more maintainable, better documented, and more production-ready state than before.

If you discover a better architecture, a cleaner implementation, or a more scalable design, recommend it before ending the session.
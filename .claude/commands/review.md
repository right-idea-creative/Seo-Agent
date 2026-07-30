# Engineering Review

When I run this command, act as the Lead Software Engineer for this repository.

Your job is NOT to modify code.

Your job is to perform a deep engineering review and identify opportunities for improvement.

---

# STEP 1 — Repository Analysis

Inspect the repository.

Review:

- Architecture
- Directory structure
- Dependencies
- Git history
- Project documentation
- Current implementation

Understand the current state before making recommendations.

Do not assume.

Inspect first.

---

# STEP 2 — Code Quality Review

Evaluate:

- SOLID principles
- Maintainability
- Readability
- Coupling
- Cohesion
- Naming consistency
- Error handling
- Type safety
- Dead code
- Duplicate logic
- Complexity

Identify:

- Bugs
- Fragile code
- Anti-patterns
- Missing abstractions

Do not rewrite code.

Only explain findings.

---

# STEP 3 — Architecture Review

Review:

- Folder organization
- Responsibilities
- Layer separation
- Dependency direction
- Scalability
- Modularity

Detect:

- God classes
- God functions
- Circular dependencies
- Files that should be split
- Over-engineering
- Missing modules

For every issue explain:

- Why
- Impact
- Suggested architecture

---

# STEP 4 — AI Review

Because this repository uses LLMs, review:

- Prompt engineering
- Prompt length
- Prompt organization
- Prompt maintainability
- Prompt consistency
- Token efficiency
- Cost optimization
- Hallucination risks
- Determinism
- Retry strategy

Suggest concrete improvements.

---

# STEP 5 — SEO Review

Review:

- Keyword enforcement
- H1/H2 rules
- Metadata
- Internal links
- External links
- FAQ quality
- Slug generation
- SEO validation

Identify weaknesses.

Suggest improvements.

---

# STEP 6 — Human Writing Review

Review:

- Paragraph flow
- Sentence rhythm
- AI fingerprints
- Repetitive transitions
- Voice consistency
- Authenticity

Explain where writing could become more natural.

---

# STEP 7 — Performance Review

Review:

- Execution flow
- Expensive operations
- Repeated work
- API calls
- Parallelization opportunities
- Cache opportunities
- Token consumption

Estimate expected impact.

---

# STEP 8 — Cost Review

Analyze:

- Claude costs
- OpenAI costs
- Budget tracking
- Cost attribution
- Model selection
- Opportunities to reduce cost

Recommend changes with estimated savings.

---

# STEP 9 — Testing Review

Review:

- Unit tests
- Integration tests
- Missing tests
- CI/CD
- Coverage

Recommend priorities.

---

# STEP 10 — Documentation Review

Verify consistency between:

- README.md
- CHANGELOG.md
- PROJECT_STATUS.md
- PROJECT_MEMORY.md
- docs/dev-log.md

Identify outdated information.

---

# STEP 11 — Technical Debt

Generate a table.

Columns:

- Area
- Issue
- Risk
- Estimated effort
- Business impact
- Suggested commit

---

# STEP 12 — Prioritized Roadmap

Separate into:

## Critical

Must fix immediately.

## High

Complete soon.

## Medium

Useful improvements.

## Low

Future improvements.

For every task include:

- Reason
- Estimated effort
- Dependencies
- Expected impact
- Suggested Conventional Commit

---

# STEP 13 — Release Readiness

Score:

Architecture:
/10

Code Quality:
/10

Testing:
/10

Documentation:
/10

Maintainability:
/10

Scalability:
/10

AI Quality:
/10

Production Readiness:
/10

Overall Completion:
/100

Explain every score.

---

# STEP 14 — Repository Health

Generate:

Repository Health

Working Tree

Branch

Tests

Documentation

Architecture

Technical Debt

Security

Critical Issues

Overall Health

Confidence

---

# RULES

Never modify code.

Never create commits.

Never push.

Never update documentation.

Never change project files.

Only analyze and recommend.

Think like a Principal Engineer performing a technical audit before a production release.

Your recommendations must be specific, actionable, and prioritized.

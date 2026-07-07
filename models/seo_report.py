from enum import Enum

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    """
    Severity levels for SEO quality issues.

    CRITICAL  — Condition that must never reach WordPress. Blocks publishing
                regardless of the minimum score threshold (e.g. empty content,
                missing H1). Carries the highest individual penalty.
    ERROR     — Significant problem that reduces score. Publishing is blocked
                if the accumulated score falls below the configured threshold.
    WARNING   — Notable but non-blocking concern. Included in score deduction.
    INFO      — Informational observation. No score impact.
    """
    CRITICAL = "critical"
    ERROR    = "error"
    WARNING  = "warning"
    INFO     = "info"


class SEOIssue(BaseModel):
    """
    A single finding produced by SEOQAService.

    penalty reflects the score deduction for this specific rule —
    not a flat amount derived from severity. Two issues of the same
    severity can have different penalties.
    """
    severity: IssueSeverity
    code: str = Field(description="Machine-readable rule identifier.")
    message: str = Field(description="Human-readable description of the issue.")
    detail: str | None = Field(
        default=None,
        description="Specific value or context, e.g. '85 chars (max 60)'.",
    )
    penalty: int = Field(ge=0, description="Score points deducted by this issue.")


class SEOSummary(BaseModel):
    """Counts of issues by severity — used for quick CLI rendering."""
    critical: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0


class SEOReport(BaseModel):
    """
    Output of SEOQAService.analyze().

    A pure data structure — no business logic. Score calculation,
    threshold checks, and publishing decisions belong to SEOQAService
    and PublisherAgent respectively.
    """
    score: int = Field(ge=0, le=100, description="Quality score from 0 to 100.")
    issues: list[SEOIssue] = Field(default_factory=list)
    summary: SEOSummary = Field(default_factory=SEOSummary)

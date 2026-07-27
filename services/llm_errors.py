"""
LLM error hierarchy — shared types for the entire LLM layer.

Placing error classes here (not inside claude_service.py) breaks the import
cycle that would otherwise form when llm_gateway.py imports error types from
the same module it is wired into.

Backward-compat note: ``ClaudeServiceError``, ``ClaudeRateLimitError``, and
``ClaudeAPIError`` are re-exported from ``services.claude_service`` so that
existing ``from services.claude_service import ...`` call sites are unchanged.
"""
from __future__ import annotations


# ── Base ──────────────────────────────────────────────────────────────────────

class LLMServiceError(Exception):
    """Base class for all LLM provider errors."""


# ── Claude errors (re-exported from services.claude_service) ─────────────────

class ClaudeServiceError(LLMServiceError):
    """Base for all Claude / Anthropic errors."""


class ClaudeRateLimitError(ClaudeServiceError):
    """Claude rate limit exceeded — always a recoverable error."""


class ClaudeAPIError(ClaudeServiceError):
    """
    Claude API error with optional HTTP status code.

    The ``status_code`` attribute lets the failover gateway distinguish
    recoverable server errors (500/503) from non-recoverable client errors
    (400/401) without re-parsing the message string.
    """

    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── OpenAI / fallback errors ──────────────────────────────────────────────────

class OpenAIGenerationError(LLMServiceError):
    """OpenAI generation request failed."""


# ── Gateway errors ────────────────────────────────────────────────────────────

class LLMAllProvidersFailedError(LLMServiceError):
    """
    All configured LLM providers failed on this request.

    ``primary_error``  — the ClaudeServiceError that triggered failover.
    ``fallback_error`` — the error returned by the fallback provider.

    The LLMGateway prints both errors to the console before raising this,
    so callers can raise typer.Exit(1) without re-printing.
    """

    def __init__(
        self,
        primary_error: Exception,
        fallback_error: Exception,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"All LLM providers failed.\n"
            f"  Primary  (Claude): {primary_error}\n"
            f"  Fallback (OpenAI): {fallback_error}"
        )

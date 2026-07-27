"""
LLMGateway — transparent failover between any number of LLM providers.

All callers use the same generate() / generate_structured() interface they have
always used with ClaudeService. The gateway is a drop-in replacement: it exposes
every public attribute ClaudeService exposed (generate, generate_structured,
model, provider, budget) and adds two new tracking properties (last_model,
last_provider) that always reflect the provider that handled the most recent
successful request.

Failover behaviour:
  1. Try primary provider (Claude).
  2. On a recoverable error, print a console notice and try the fallback (OpenAI).
  3. If both fail, print both errors and raise LLMAllProvidersFailedError.
  4. Non-recoverable errors (400, 401) surface immediately — no failover.

Recoverable errors:
  - HTTP status codes: 402, 429, 500, 502, 503, 504
  - Message keywords:  credit, insufficient, quota, timeout, connection,
                       unavailable, overloaded

Model tracking:
  After every successful call the gateway records which provider answered
  (model + provider strings).  Callers that read ``gateway.model`` immediately
  after a call always see the model that actually generated the response —
  never the preferred/configured default.

Future providers:
  Add another provider by nesting gateways:
    LLMGateway(
        primary=claude_service,
        fallback=LLMGateway(primary=openai_service, fallback=gemini_service),
    )
  No agent code changes required.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console

from services.llm_errors import (
    ClaudeAPIError,
    ClaudeRateLimitError,
    ClaudeServiceError,
    LLMAllProvidersFailedError,
    OpenAIGenerationError,
)

if TYPE_CHECKING:
    from services.budget_service import BudgetService
    from services.claude_service import ClaudeService
    from services.openai_generation_service import OpenAIGenerationService

logger = logging.getLogger(__name__)
_console = Console()

_RECOVERABLE_STATUS_CODES: frozenset[int] = frozenset({402, 429, 500, 502, 503, 504})

_RECOVERABLE_KEYWORDS: tuple[str, ...] = (
    "credit",
    "insufficient",
    "quota",
    "timeout",
    "connection",
    "unavailable",
    "overloaded",
)


class LLMGateway:
    """
    Drop-in replacement for ClaudeService with automatic provider failover.

    Public interface (identical to ClaudeService, plus tracking properties):

        generate(system, messages, *, max_tokens, thinking) -> str
        generate_structured(system, messages, tool_name, tool_description,
                            input_schema, *, max_tokens, thinking) -> dict
        model      -> str   model name of the provider that handled the last call
        provider   -> str   provider name  ("anthropic" | "openai" | …)
        budget     -> BudgetService | None
        last_model    alias for model
        last_provider alias for provider
    """

    def __init__(
        self,
        primary: "ClaudeService",
        fallback: "OpenAIGenerationService | None" = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        # Initialise tracking to the primary provider.  Updated on every
        # successful call so callers always read the actual response source.
        self._last_provider: str = primary.provider
        self._last_model: str = primary.model

    # ── Tracking properties ───────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Model name of the provider that handled the most recent call."""
        return self._last_model

    @property
    def last_model(self) -> str:
        return self._last_model

    @property
    def provider(self) -> str:
        """Provider name of the provider that handled the most recent call."""
        return self._last_provider

    @property
    def last_provider(self) -> str:
        return self._last_provider

    # ── Passthrough attributes ─────────────────────────────────────────────────

    @property
    def budget(self) -> "BudgetService | None":
        """Budget service — delegates to primary (ClaudeService owns it)."""
        return self._primary.budget

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8096,
        thinking: bool = True,
        model: str | None = None,
        label: str = "",
    ) -> str:
        oai_messages = self._translate_messages_for_openai(messages)
        return self._call_with_failover(
            "text",
            lambda: self._primary.generate(
                system, messages, max_tokens=max_tokens, thinking=thinking,
                model=model, label=label,
            ),
            lambda: self._fallback.generate(
                system, oai_messages, max_tokens=max_tokens, thinking=thinking
            ) if self._fallback else None,
        )

    def generate_structured(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
        thinking: bool = True,
        model: str | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        oai_messages = self._translate_messages_for_openai(messages)
        return self._call_with_failover(
            "structured",
            lambda: self._primary.generate_structured(
                system, messages, tool_name, tool_description, input_schema,
                max_tokens=max_tokens, thinking=thinking,
                model=model, label=label,
            ),
            lambda: self._fallback.generate_structured(
                system, oai_messages, tool_name, tool_description, input_schema,
                max_tokens=max_tokens, thinking=thinking,
            ) if self._fallback else None,
        )

    # ── Message format translation ────────────────────────────────────────────

    @staticmethod
    def _translate_messages_for_openai(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert Anthropic-format image content blocks to OpenAI image_url format.

        Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
        OpenAI:    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "auto"}}

        Called eagerly before the fallback lambda is constructed so the translation
        only happens once even if _call_with_failover retries. Messages that contain
        no image blocks are returned unchanged (same list, no copy).
        """
        has_image = any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(blk, dict) and blk.get("type") == "image"
                for blk in msg["content"]
            )
            for msg in messages
        )
        if not has_image:
            return messages

        translated: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                translated.append(msg)
                continue

            new_content: list[dict[str, Any]] = []
            for blk in content:
                if (
                    isinstance(blk, dict)
                    and blk.get("type") == "image"
                    and isinstance(blk.get("source"), dict)
                    and blk["source"].get("type") == "base64"
                ):
                    media_type = blk["source"].get("media_type", "image/jpeg")
                    data = blk["source"].get("data", "")
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}",
                            "detail": "auto",
                        },
                    })
                else:
                    new_content.append(blk)

            translated.append({**msg, "content": new_content})

        return translated

    # ── Core failover logic ───────────────────────────────────────────────────

    def _call_with_failover(
        self,
        call_type: str,
        primary_fn: Callable[[], Any],
        fallback_fn: Callable[[], Any | None],
    ) -> Any:
        """
        Try primary_fn; on a recoverable ClaudeServiceError try fallback_fn.
        Updates _last_model / _last_provider to reflect the actual responder.
        """
        try:
            result = primary_fn()
            self._last_provider = self._primary.provider
            self._last_model = self._primary.model
            return result
        except ClaudeServiceError as exc:
            if not self._is_recoverable(exc) or self._fallback is None:
                raise
            self._notify_failover(exc, call_type)
            try:
                result = fallback_fn()
                self._last_provider = self._fallback.provider
                self._last_model = self._fallback.model
                return result
            except OpenAIGenerationError as fallback_exc:
                self._notify_both_failed(exc, fallback_exc)
                raise LLMAllProvidersFailedError(exc, fallback_exc) from fallback_exc

    # ── Classification helpers ────────────────────────────────────────────────

    @staticmethod
    def _is_recoverable(exc: ClaudeServiceError) -> bool:
        if isinstance(exc, ClaudeRateLimitError):
            return True
        if isinstance(exc, ClaudeAPIError) and exc.status_code in _RECOVERABLE_STATUS_CODES:
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in _RECOVERABLE_KEYWORDS)

    @staticmethod
    def _error_reason(exc: ClaudeServiceError) -> str:
        if isinstance(exc, ClaudeRateLimitError):
            return "rate limit exceeded"
        if isinstance(exc, ClaudeAPIError) and exc.status_code:
            _labels = {
                402: "insufficient credits",
                429: "rate limit exceeded",
                500: "internal server error",
                502: "bad gateway",
                503: "service unavailable",
                504: "gateway timeout",
            }
            if exc.status_code in _labels:
                return _labels[exc.status_code]
        msg = str(exc).lower()
        for kw in _RECOVERABLE_KEYWORDS:
            if kw in msg:
                return kw
        return str(exc)[:80]

    # ── Console notifications ─────────────────────────────────────────────────

    @classmethod
    def _notify_failover(cls, exc: ClaudeServiceError, call_type: str) -> None:
        reason = cls._error_reason(exc)
        _console.print(
            f"[yellow]Claude unavailable ({reason}). Switching to OpenAI...[/yellow]"
        )
        logger.warning("Claude failover triggered (%s): %s", call_type, exc)

    @staticmethod
    def _notify_both_failed(
        primary_exc: ClaudeServiceError,
        fallback_exc: OpenAIGenerationError,
    ) -> None:
        _console.print("[bold red]Both LLM providers failed:[/bold red]")
        _console.print(f"  [red]Claude : {primary_exc}[/red]")
        _console.print(f"  [red]OpenAI : {fallback_exc}[/red]")
        logger.error(
            "All LLM providers failed — Claude: %s | OpenAI: %s",
            primary_exc,
            fallback_exc,
        )

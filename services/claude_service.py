import logging
from typing import Any

import anthropic

from config import settings
from services.budget_service import BudgetService

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ClaudeServiceError(Exception):
    """Base exception for all Claude service errors."""

class ClaudeRateLimitError(ClaudeServiceError):
    """Raised when the Anthropic API rate limit is exceeded."""

class ClaudeAPIError(ClaudeServiceError):
    """Raised when the API returns an unexpected or unrecoverable error."""


# ── Service ───────────────────────────────────────────────────────────────────

class ClaudeService:
    """
    Thin wrapper around the Anthropic SDK.

    This class has one job: translate between our application's calling
    conventions and the Anthropic SDK. It knows nothing about articles,
    SEO, or prompts — those concerns live in the agents and prompts layers.

    Two generation patterns are exposed:
    - generate()            Streaming text response for long content (articles).
    - generate_structured() Tool-use response for guaranteed structured JSON
                            (SEO metadata, outlines, classifications).

    Adaptive thinking is enabled by default on both methods. Claude decides
    how much reasoning to apply based on task complexity.
    """

    def __init__(self, budget: BudgetService | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model
        self._budget = budget

    @property
    def model(self) -> str:
        return self._model

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8096,
        thinking: bool = True,
    ) -> str:
        """
        Stream a text response and return it in full.

        Streaming is used to prevent request timeouts on long article
        generation. The complete text is returned once streaming finishes.

        Args:
            system:     System prompt string.
            messages:   List of message dicts in Anthropic format.
            max_tokens: Upper bound on generated tokens (default 8 096).
            thinking:   Enable adaptive thinking (default True).

        Returns:
            The text content of the response.

        Raises:
            ClaudeRateLimitError: API rate limit exceeded.
            ClaudeAPIError:       Any other API or SDK error.
        """
        kwargs = self._base_kwargs(system, messages, max_tokens, thinking)

        if self._budget is not None:
            self._budget.check_claude()

        try:
            with self._client.messages.stream(**kwargs) as stream:
                final = stream.get_final_message()

            if self._budget is not None and final.usage:
                self._budget.record_claude(final.usage.input_tokens, final.usage.output_tokens)

            for block in final.content:
                if block.type == "text":
                    return block.text

            raise ClaudeAPIError("Response contained no text block.")

        except anthropic.RateLimitError as exc:
            logger.warning("Claude rate limit reached: %s", exc)
            raise ClaudeRateLimitError("Rate limit exceeded — try again later.") from exc

        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise ClaudeAPIError(str(exc)) from exc

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
    ) -> dict[str, Any]:
        """
        Force a tool_use call and return the structured input dict.

        Uses tool_choice to guarantee Claude calls exactly the specified tool,
        returning clean JSON without any string parsing on our side.

        Args:
            system:           System prompt string.
            messages:         List of message dicts in Anthropic format.
            tool_name:        Name of the tool Claude must call.
            tool_description: What the tool does (helps Claude understand intent).
            input_schema:     JSON Schema dict describing the tool parameters.
            max_tokens:       Upper bound on generated tokens (default 4 096).
            thinking:         Enable adaptive thinking (default True).

        Returns:
            The tool_input dict from Claude's tool_use response block.

        Raises:
            ClaudeRateLimitError: API rate limit exceeded.
            ClaudeAPIError:       Tool not called or any other API error.
        """
        kwargs = self._base_kwargs(system, messages, max_tokens, thinking)
        kwargs["tools"] = [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        if self._budget is not None:
            self._budget.check_claude()

        try:
            response = self._client.messages.create(**kwargs)

            if self._budget is not None and response.usage:
                self._budget.record_claude(response.usage.input_tokens, response.usage.output_tokens)

            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    return block.input  # type: ignore[return-value]

            raise ClaudeAPIError(
                f"Tool '{tool_name}' was not called in the response. "
                f"Content types received: {[b.type for b in response.content]}"
            )

        except anthropic.RateLimitError as exc:
            logger.warning("Claude rate limit reached: %s", exc)
            raise ClaudeRateLimitError("Rate limit exceeded — try again later.") from exc

        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise ClaudeAPIError(str(exc)) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _base_kwargs(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        thinking: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return kwargs


# ── Module-level singletons ───────────────────────────────────────────────────

# Shared budget instance used by both ClaudeService and OpenAIImageGenerator.
# Exported via services/__init__.py so main.py can pass it to image generators.
budget = BudgetService(
    settings.budget_dir,
    settings.claude_monthly_budget_usd,
    settings.openai_monthly_budget_usd,
)

claude = ClaudeService(budget=budget)

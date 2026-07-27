"""
OpenAIGenerationService — OpenAI backend for article and SEO generation.

Implements the same generate() / generate_structured() interface as ClaudeService
so it can be used as a transparent failover provider inside LLMGateway.

The ``thinking`` kwarg is accepted everywhere for interface parity but is
silently ignored — OpenAI has no equivalent of Claude's adaptive thinking.

Temperature defaults:
    generate()            0.7 — creative, long-form article content
    generate_structured() 0.3 — deterministic, structured JSON output
"""
from __future__ import annotations

import json
import logging
from typing import Any

from config import settings
from services.llm_errors import OpenAIGenerationError

logger = logging.getLogger(__name__)

_FALLBACK_MODEL = "gpt-4o"


class OpenAIGenerationService:
    """
    OpenAI backend that mirrors the ClaudeService interface.

    Intended for use as a failover provider — it is never imported directly
    by articles, agents, or QA services. Those services continue to talk to
    ClaudeService (via LLMGateway) and are unaware that this class exists.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _FALLBACK_MODEL,
    ) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    # ── Public interface (mirrors ClaudeService) ──────────────────────────────

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8096,
        thinking: bool = True,   # accepted, not used
    ) -> str:
        """
        Stream a text completion from OpenAI and return the full response.

        Args:
            system:     System prompt — prepended as a "system" role message.
            messages:   Conversation messages in Anthropic format
                        (``{"role": "user", "content": "..."}``).
                        Role values "user"/"assistant" pass through unchanged.
            max_tokens: Upper bound on generated tokens.
            thinking:   Ignored (no OpenAI equivalent).

        Returns:
            Complete text content of the first choice.

        Raises:
            OpenAIGenerationError: On API failure or empty response.
        """
        oai_messages = [{"role": "system", "content": system}, *messages]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False,
            )
        except Exception as exc:
            logger.error("OpenAI generation error: %s", exc)
            raise OpenAIGenerationError(str(exc)) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise OpenAIGenerationError(
                f"OpenAI ({self._model}) returned an empty response."
            )

        logger.debug(
            "OpenAI generation: %d input + %d output tokens",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )
        return content

    def generate_structured(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
        thinking: bool = True,   # accepted, not used
    ) -> dict[str, Any]:
        """
        Force a function call and return the parsed arguments dict.

        Translates Claude's tool_use format to OpenAI's function-calling format.
        The ``input_schema`` is standard JSON Schema in both APIs — no conversion
        is required.

        Args:
            system:           System prompt.
            messages:         Conversation messages in Anthropic format.
            tool_name:        Name of the function OpenAI must call.
            tool_description: Natural-language description of the function.
            input_schema:     JSON Schema ``{"type": "object", "properties": {...}}``
            max_tokens:       Upper bound on generated tokens.
            thinking:         Ignored.

        Returns:
            Parsed function arguments as a Python dict.

        Raises:
            OpenAIGenerationError: On API failure, empty response, or missing call.
        """
        oai_messages = [{"role": "system", "content": system}, *messages]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": input_schema,
                },
            }
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": tool_name}},
                max_tokens=max_tokens,
                temperature=0.3,
                stream=False,
            )
        except Exception as exc:
            logger.error("OpenAI structured generation error: %s", exc)
            raise OpenAIGenerationError(str(exc)) from exc

        choice = response.choices[0] if response.choices else None
        tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None)
        if not tool_calls:
            raise OpenAIGenerationError(
                f"OpenAI ({self._model}) did not call the required function '{tool_name}'."
            )

        arguments_str = tool_calls[0].function.arguments
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError as exc:
            raise OpenAIGenerationError(
                f"OpenAI function arguments are not valid JSON: {arguments_str[:200]}"
            ) from exc

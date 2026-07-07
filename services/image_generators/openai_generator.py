from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from models.image_asset import ImageAsset, ImageSource
from services.image_generators import ImageGenerationRequest

if TYPE_CHECKING:
    from services.budget_service import BudgetService

logger = logging.getLogger(__name__)


class OpenAIImageGenerator:
    """
    Generates photorealistic images via OpenAI DALL-E 3.

    Configuration:
        quality="hd"        Higher detail and consistency.
        style="natural"     Photorealistic over vivid/stylized. Critical for
                            avoiding the artificial look of AI-generated images.
        size="1792x1024"    Landscape — best aspect ratio for blog post images.

    The revised_prompt returned by OpenAI is stored in source_detail for
    traceability. DALL-E 3 often enhances the prompt, so the final prompt
    may differ from the one submitted.

    Implements the ImageGenerator Protocol — no base class needed.
    """

    _MODEL = "dall-e-3"
    _DEFAULT_SIZE = "1792x1024"

    def __init__(self, api_key: str, budget: "BudgetService | None" = None) -> None:
        try:
            import openai as _openai
            self._client = _openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package is required for AI image generation. "
                "Run: pip install openai"
            ) from exc
        self._budget = budget

    def generate(self, request: ImageGenerationRequest) -> ImageAsset:
        if self._budget is not None:
            self._budget.check_openai()

        logger.info("Generating image via DALL-E 3 (size=%s)...", request.size)

        response = self._client.images.generate(
            model=self._MODEL,
            prompt=request.prompt,
            size=request.size,          # type: ignore[arg-type]
            quality="hd",
            style="natural",
            n=1,
            response_format="url",
        )

        image_data = response.data[0]
        image_url = image_data.url
        revised_prompt = getattr(image_data, "revised_prompt", None) or request.prompt

        logger.debug("DALL-E 3 revised prompt: %s", revised_prompt[:120])

        data = self._download(image_url)

        if self._budget is not None:
            self._budget.record_openai(images=1)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        return ImageAsset(
            filename=f"generated_{timestamp}.png",
            mime_type="image/png",
            data=data,
            alt_text=request.alt_text,
            source=ImageSource.GENERATED,
            source_detail=revised_prompt[:500],
        )

    @staticmethod
    def _download(url: str) -> bytes:
        with httpx.Client(timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from models.image_asset import ImageAsset


@dataclass
class ImageGenerationRequest:
    """
    Everything an ImageGenerator needs to produce a photorealistic image.

    prompt is the primary driver — constructed by ImageResolverAgent via Claude
    to incorporate article context, image type, and visual style guidelines.
    alt_text and size are passed through to the resulting ImageAsset.
    """
    prompt: str
    alt_text: str
    size: str = "1792x1024"   # landscape — optimal for blog featured images


@runtime_checkable
class ImageGenerator(Protocol):
    """
    Protocol for AI image generators.

    Each implementation handles one provider (OpenAI, Ideogram, Flux, etc.).
    ImageResolverAgent depends on this interface, not on any concrete class,
    so providers are interchangeable without touching the agent or any other code.

    generate() always returns an ImageAsset. Raises on API failure.
    It never returns None — if generation fails, it raises an exception.
    """

    def generate(self, request: ImageGenerationRequest) -> ImageAsset:
        ...

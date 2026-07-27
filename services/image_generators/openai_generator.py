from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from models.image_asset import ImageAsset, ImageSource
from services.image_generators import ImageGenerationRequest

if TYPE_CHECKING:
    from services.budget_service import BudgetService

logger = logging.getLogger(__name__)


def _detect_content_type(data: bytes) -> tuple[str, str]:
    """
    Detect MIME type and file extension from image magic bytes.

    Returns (mime_type, extension). Never returns application/octet-stream.
    Falls back to image/jpeg when the format cannot be determined — JPEG is the
    most common format for Drive photos and is always accepted by images.edit().

    Passing raw bytes to images.edit() sends application/octet-stream and is
    rejected by the API. Passing BytesIO with .name is unreliable. An explicit
    (filename, bytes, content_type) 3-tuple is the only guaranteed transport.
    """
    if data[:4] == b"\x89PNG":
        return "image/png", ".png"
    if data[:12] == b"\x00\x00\x00\x0cjP  ":  # JPEG 2000
        return "image/jpeg", ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg", ".jpg"
    return "image/jpeg", ".jpg"  # safe fallback — accepted by images.edit()


def _as_file_tuple(img_bytes: bytes, filename: str | None = None) -> tuple[str, bytes, str]:
    """
    Build an explicit (filename, bytes, content_type) upload tuple for images.edit().

    This is the only format that guarantees the correct Content-Type header is sent.
    - Raw bytes → application/octet-stream (rejected by API)
    - BytesIO with .name → unreliable; BytesIO does not officially support .name
    - 3-tuple → explicit filename and content_type, always correct
    """
    mime_type, ext = _detect_content_type(img_bytes)
    name = filename if filename else f"photo{ext}"
    return (name, img_bytes, mime_type)


_OPENAI_MAX_BYTES = 45 * 1024 * 1024  # 45 MB — conservative ceiling below OpenAI's 50 MB hard limit


def prepare_image_for_openai_edit(image_bytes: bytes) -> tuple[bytes, str]:
    """
    The single gateway for all image bytes sent to OpenAI images.edit().

    Every reference image must pass through this function before images.edit()
    is called. No image larger than 45 MB may reach the OpenAI API.

    Strategy (resize only as last resort):
      1. Already ≤ 45 MB → return as-is (no print).
      2. No transparency → JPEG quality 95, 92, 88, 85 (no resize).
      3. Has alpha → keep PNG; resize with PNG. No alpha → JPEG for resize steps.
      4. Resize: 90% → 80% → 70% → 60% → 50%.

    Prints [OPENAI] only when optimization is actually performed.
    Raises RuntimeError if image cannot be reduced below 45 MB.

    Returns (optimized_bytes, mime_type).
    """
    mime_type, _ = _detect_content_type(image_bytes)

    if len(image_bytes) <= _OPENAI_MAX_BYTES:
        return image_bytes, mime_type

    original_mb = len(image_bytes) / 1024 / 1024

    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            f"Image is {original_mb:.1f} MB (limit 45 MB) and Pillow is not installed. "
            "Run: pip install Pillow"
        )

    img = Image.open(io.BytesIO(image_bytes))
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and img.info.get("transparency") is not None
    )

    # ── Try JPEG quality reduction (no resize) ───────────────────────────────
    if not has_alpha:
        for quality in (95, 92, 88, 85):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = buf.getvalue()
            if len(candidate) <= _OPENAI_MAX_BYTES:
                optimized_mb = len(candidate) / 1024 / 1024
                print(f"[OPENAI]")
                print(f"  Original:  {original_mb:.1f} MB")
                print(f"  Optimized: {optimized_mb:.1f} MB")
                print(f"  Upload:    OK")
                return candidate, "image/jpeg"

    # ── Resize progressively ─────────────────────────────────────────────────
    for scale in (0.90, 0.80, 0.70, 0.60, 0.50):
        w = max(1, int(img.width * scale))
        h = max(1, int(img.height * scale))
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        if has_alpha:
            resized.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        else:
            resized.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
            mime = "image/jpeg"
        candidate = buf.getvalue()
        if len(candidate) <= _OPENAI_MAX_BYTES:
            optimized_mb = len(candidate) / 1024 / 1024
            print(f"[OPENAI]")
            print(f"  Original:  {original_mb:.1f} MB")
            print(f"  Optimized: {optimized_mb:.1f} MB")
            print(f"  Upload:    OK")
            return candidate, mime

    raise RuntimeError(
        f"Image ({original_mb:.1f} MB) could not be reduced below 45 MB "
        "after JPEG conversion and progressive resizing to 50%."
    )


# Valid sizes for gpt-image-1. Landscape/portrait legacy sizes are remapped
# to their nearest supported equivalent (1792x1024 → 1536x1024, etc.).
_VALID_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
_SIZE_REMAP = {
    "1792x1024": "1536x1024",   # DALL-E 3 landscape → gpt-image-1 landscape
    "1024x1792": "1024x1536",   # DALL-E 3 portrait  → gpt-image-1 portrait
    "256x256":   "1024x1024",
    "512x512":   "1024x1024",
}


class OpenAIImageGenerator:
    """
    Generates photorealistic images via OpenAI gpt-image-1.

    Configuration:
        quality="high"      Maximum detail — equivalent to DALL-E 3 "hd".
        size="1536x1024"    Landscape — best aspect ratio for blog post images.
                            (gpt-image-1 does not support the legacy 1792x1024.)

    Migration note (openai>=2.x):
        dall-e-3 was removed from the API; gpt-image-1 is the successor.
        response_format and style are not supported by gpt-image-1.
        Images are returned as inline b64_json and decoded without an HTTP round-trip.

    Implements the ImageGenerator Protocol — no base class needed.
    """

    _MODEL = "gpt-image-1"
    _DEFAULT_SIZE = "1536x1024"

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

    def generate_variation(
        self,
        reference_images: list[bytes],
        request: "ImageGenerationRequest",
        variation_prompt: str,
    ) -> "ImageAsset":
        """
        Create an image variation using one or more Drive reference photographs.

        Passes all references into gpt-image-1 images.edit() with high input
        fidelity so the model learns multiple aspects of the company's identity:

          reference_images[0] — primary match (best Drive photo for this slot)
          reference_images[1] — technician / uniform reference (optional)
          reference_images[2] — truck / branding reference (optional)
          reference_images[3] — environment / lighting reference (optional)

        The variation_prompt describes ONLY what changes. Everything else
        is defined by the reference photographs.

        Falls back to a single reference image if the API rejects the list.
        """
        if self._budget is not None:
            self._budget.check_openai()

        size = self._resolve_size(request.size)
        # Preprocess every reference through the OpenAI size gate before any API call.
        # Both the primary attempt and the single-ref fallback use these optimized bytes.
        refs = [prepare_image_for_openai_edit(r)[0] for r in reference_images[:4]]
        n_refs = len(refs)
        logger.info(
            "Generating variation via gpt-image-1 edit (%d reference(s), size=%s, fidelity=high)...",
            n_refs, size,
        )

        # Always use explicit (filename, bytes, content_type) tuples.
        # Raw bytes → application/octet-stream (rejected by API).
        # BytesIO with .name is unreliable — .name is not a supported attribute.
        image_arg: list | tuple
        if n_refs > 1:
            image_arg = [_as_file_tuple(r, f"reference_{i+1}") for i, r in enumerate(refs)]
        else:
            image_arg = _as_file_tuple(refs[0], "reference_1")

        mime, _ = _detect_content_type(refs[0])
        logger.debug("images.edit() upload: %d image(s), detected content-type=%s", n_refs, mime)

        try:
            response = self._client.images.edit(
                image=image_arg,
                prompt=variation_prompt,
                model=self._MODEL,
                size=size,                 # type: ignore[arg-type]
                quality="high",
                input_fidelity="high",
                n=1,
            )
        except Exception as exc:
            if n_refs > 1:
                # API may reject multiple images in some edge cases — fall back to primary only
                logger.warning(
                    "Multi-reference edit failed (%s) — retrying with primary reference only.", exc
                )
                response = self._client.images.edit(
                    image=_as_file_tuple(refs[0], "reference_1"),
                    prompt=variation_prompt,
                    model=self._MODEL,
                    size=size,             # type: ignore[arg-type]
                    quality="high",
                    input_fidelity="high",
                    n=1,
                )
            else:
                raise

        image_data = response.data[0]
        data = self._get_bytes(image_data)

        if self._budget is not None:
            self._budget.record_openai(images=1)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        return ImageAsset(
            filename=f"variation_{timestamp}.png",
            mime_type="image/png",
            data=data,
            alt_text=request.alt_text,
            source=ImageSource.EDITED,
            source_detail=variation_prompt[:500],
        )

    def generate(self, request: ImageGenerationRequest) -> ImageAsset:
        if self._budget is not None:
            self._budget.check_openai()

        size = self._resolve_size(request.size)
        logger.info("Generating image via gpt-image-1 (size=%s, quality=high)...", size)

        response = self._client.images.generate(
            model=self._MODEL,
            prompt=request.prompt,
            size=size,              # type: ignore[arg-type]
            quality="high",
            n=1,
        )

        image_data = response.data[0]
        revised_prompt = getattr(image_data, "revised_prompt", None) or request.prompt

        data = self._get_bytes(image_data)

        if self._budget is not None:
            self._budget.record_openai(images=1)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        return ImageAsset(
            filename=f"generated_{timestamp}.png",
            mime_type="image/png",
            data=data,
            alt_text=request.alt_text,
            source=ImageSource.GENERATED,
            source_detail=(revised_prompt or request.prompt)[:500],
        )

    @staticmethod
    def _resolve_size(requested: str | None) -> str:
        """Map a requested size to the nearest valid gpt-image-1 size."""
        if not requested:
            return OpenAIImageGenerator._DEFAULT_SIZE
        if requested in _VALID_SIZES:
            return requested
        remapped = _SIZE_REMAP.get(requested)
        if remapped:
            logger.debug("Size '%s' remapped to '%s' for gpt-image-1", requested, remapped)
            return remapped
        logger.warning("Unknown size '%s'; falling back to %s", requested, OpenAIImageGenerator._DEFAULT_SIZE)
        return OpenAIImageGenerator._DEFAULT_SIZE

    @staticmethod
    def _get_bytes(image_data: object) -> bytes:
        """Extract raw image bytes from a gpt-image-1 response item.

        gpt-image-1 always returns b64_json; url fallback kept for forward-compat.
        """
        b64: str | None = getattr(image_data, "b64_json", None)
        if b64:
            return base64.b64decode(b64)

        url: str | None = getattr(image_data, "url", None)
        if url:
            logger.debug("gpt-image-1 response: downloading from url")
            return OpenAIImageGenerator._download(url)

        raise RuntimeError(
            "OpenAI image response contained neither 'b64_json' nor 'url'. "
            "Check API version compatibility."
        )

    @staticmethod
    def _download(url: str) -> bytes:
        with httpx.Client(timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

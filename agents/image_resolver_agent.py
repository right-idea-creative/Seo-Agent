"""
ImageResolverAgent — two-phase image resolution for article publishing.

Phase 1 — plan():
    Claude analyzes the full article as a professional editor and returns:
    - A list[ImageRequest] specifying what each image should depict and why.
    - modified_markdown: content_markdown with <!-- SEO_AGENT_IMAGE: id -->
      markers inserted at inline image positions.

Phase 2 — resolve_all():
    For each ImageRequest, the agent:
    1. Pre-filters Drive candidates using keyword scoring (no Claude call).
    2. Sends top-5 thumbnails to Claude Vision for final selection.
    3. Downloads the winner OR generates via AI if no suitable Drive image found.

The agent never knows whether the caller will publish to WordPress — it just
returns (ImageRequest, ImageAsset) pairs and leaves upload to MediaService.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any

from models.article import Article
from models.image_asset import ImageAsset, ImageSource
from models.image_context import ImageContext
from models.image_request import ImagePlacementPlan, ImagePurpose, ImageRequest, ImageType
from models.visual_style import VisualStyleProfile
from services.claude_service import ClaudeService
from services.google_drive_service import DriveFileInfo, GoogleDriveService
from services.image_generators import ImageGenerationRequest, ImageGenerator

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ImageResolverError(Exception):
    """Raised when an image cannot be resolved from any available source."""


# ── Planning schema ───────────────────────────────────────────────────────────

_IMAGE_TYPE_VALUES = [t.value for t in ImageType]

_PLANNING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requests": {
            "type": "array",
            "description": (
                "All image requests for this article. "
                "First entry MUST be the featured image (purpose: 'featured'). "
                "Remaining entries are inline images in document order."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique ID: 'img_001', 'img_002', etc."
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["featured", "inline"]
                    },
                    "image_type": {
                        "type": "string",
                        "enum": _IMAGE_TYPE_VALUES
                    },
                    "section_title": {
                        "type": "string",
                        "description": "H2/H3 heading of the section this image belongs to. Omit for featured."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Precise visual description of what the image must show."
                    },
                    "communicative_intent": {
                        "type": "string",
                        "description": "Why this image is here — what value it adds to the reader."
                    },
                    "related_keyword": {
                        "type": "string",
                        "description": "SEO keyword this image reinforces."
                    },
                    "alt_text": {
                        "type": "string",
                        "description": (
                            "SEO-optimized alt text: include the article's focus keyword naturally, "
                            "describe precisely what is visible in the image, under 125 characters. "
                            "BAD: 'garage door'. "
                            "GOOD: 'technician replacing garage door torsion spring in residential garage'."
                        )
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional display caption shown below the image."
                    }
                },
                "required": ["id", "purpose", "image_type", "subject", "communicative_intent", "alt_text"]
            }
        },
        "modified_markdown": {
            "type": "string",
            "description": (
                "The original article markdown with <!-- SEO_AGENT_IMAGE: id --> markers "
                "inserted BEFORE the paragraph or heading where each INLINE image should appear. "
                "The FEATURED image (img_001) has NO marker — it is set via WordPress featured_media."
            )
        },
        "reasoning": {
            "type": "string",
            "description": "Editorial reasoning explaining why these images were chosen and where they were placed."
        }
    },
    "required": ["requests", "modified_markdown", "reasoning"]
}

_PLANNING_SYSTEM = """\
You are a senior content editor and SEO specialist for a digital marketing agency.

Your task: analyze a complete article and create an optimal image placement plan.

Rules:
1. ALWAYS include exactly one FEATURED image (purpose: "featured", id: "img_001").
   The featured image represents the article as a whole — choose a subject that
   captures the main topic and would work as a compelling thumbnail.

2. Add INLINE images only where they genuinely improve the reader's experience:
   - At major topic shifts or new sections
   - To illustrate a process, procedure, or sequence of steps
   - To show a product, tool, or piece of equipment being described
   - To represent a problem the reader may recognize
   - To break up text blocks that are too long (400+ words without visual relief)
   DO NOT add an image just because there is a heading.

3. For short articles (under 500 words): featured image only is acceptable.
   For medium articles (500–1000 words): 1–2 inline images.
   For long articles (1000+ words): 2–4 inline images.
   Never exceed 5 images total regardless of length.

4. Image types guide both Drive search and AI generation:
   - photograph: generic professional photo appropriate to the topic
   - process_photo: a technician or person performing the procedure described
   - product_photo: the specific product, tool, or equipment being discussed
   - team_photo: professionals representing the type of team described
   - problem_photo: a visual representation of the problem or symptom described
   - before_after: a comparison or transformation relevant to the content
   - infographic: informational graphic (generated as a descriptive photograph in MVP)

5. Subjects must be precise visual descriptions, not abstract concepts.
   BAD: "garage door problem"
   GOOD: "residential garage door with broken torsion spring, garage environment"

6. Insert inline markers BEFORE the section or paragraph they illustrate,
   on their own line. Format: <!-- SEO_AGENT_IMAGE: img_002 -->

7. IDs: img_001 (featured), img_002, img_003... (inline, in document order).

8. Alt text must be keyword-rich and descriptive:
   - Include the article's focus keyword naturally (once, not forced)
   - Describe what is literally visible in the image — not the article topic
   - Under 125 characters, plain language, no keyword stuffing
   BAD: "garage door" or "garage door service company"
   GOOD: "technician replacing garage door torsion spring in a residential garage" """


# Max Drive candidates to send to Claude Vision per image request.
# Balances coverage vs. API call size. Candidates are pre-ranked by keyword
# score so these 15 are the most relevant from the full index.
_MAX_VISION_CANDIDATES = 15


def _detect_media_type(data: bytes) -> str:
    """Return the Claude-accepted media_type string from image magic bytes."""
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"

# ── Candidate scoring ─────────────────────────────────────────────────────────

_SCORE_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "it", "its",
    "img", "dsc", "jpg", "jpeg", "png", "webp", "gif", "heic",
})


def _tokenize(text: str) -> frozenset[str]:
    """
    Normalize text to a set of meaningful tokens for candidate scoring.

    Splits on path separators and punctuation, lowercases, removes noise words,
    drops tokens shorter than 3 characters, and adds the singular form of plural
    words so that "doors" matches "door" and "springs" matches "spring".
    """
    text = re.sub(r"[\\/]", " ", text)            # path separators → space
    text = re.sub(r"[^\w\s]", " ", text.lower())  # punctuation → space
    tokens: set[str] = set()
    for w in text.split():
        if w in _SCORE_STOP_WORDS or len(w) < 3:
            continue
        tokens.add(w)
        if w.endswith("s") and len(w) > 3:
            tokens.add(w[:-1])  # "springs" → also add "spring"
    return frozenset(tokens)


def _score_candidate(candidate: "DriveFileInfo", req: "ImageRequest") -> int:
    """
    Score a Drive candidate by how many of the request's keyword tokens appear
    in the candidate's folder path, filename, and description.

    Higher score = more semantically relevant to the image request.
    A score of 0 means no token overlap — the candidate may still be selected
    by Claude Vision if its visual content matches, but it ranks last.
    """
    req_tokens = (
        _tokenize(req.subject)
        | _tokenize(req.image_type.value.replace("_", " "))
        | _tokenize(req.related_keyword or "")
    )
    candidate_tokens = (
        _tokenize(candidate.folder_path)
        | _tokenize(candidate.name)
        | _tokenize(candidate.description or "")
    )
    return len(req_tokens & candidate_tokens)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ImageResolverAgent:
    """
    Resolves the complete image set for an article in two phases.

    Phase 1 — plan(article):
        Claude analyzes the article and returns an ImagePlacementPlan with
        all ImageRequest objects and the modified_markdown with markers.

    Phase 2 — resolve_all(plan, style_profile, folder_id):
        For each request:
        1. Pre-filter Drive candidates by keyword score.
        2. Claude Vision picks the best from the top 5 thumbnails.
        3. Download winner OR generate with AI.

    Either drive or generator may be None:
    - No drive: skips Drive search entirely, goes straight to generation.
    - No generator: Drive only; raises ImageResolverError if no Drive image found.
    - Neither: raises ImageResolverError immediately.
    """

    _VISION_SYSTEM = (
        "You are an expert image curator for a digital content agency. "
        "Select the photograph that best matches the described subject and communicative intent. "
        "Prefer authentic, natural-looking photographs over staged or generic stock images."
    )

    _GEN_PROMPT_SYSTEM = """\
You are an expert at writing prompts for DALL-E 3 to generate photorealistic images
for professional business websites.

Rules:
- ALWAYS use: "professional photograph", "natural lighting", "realistic", "editorial quality"
- NEVER use: illustration, digital art, render, 3D, cartoon, painting, sketch, artistic
- The image must look like it was taken by a professional photographer on-site
- Avoid: text overlays, watermarks, logos, unnatural compositions
- Avoid: obviously AI-generated aesthetics (too perfect, plastic textures, impossible lighting)
- Include camera-style descriptors: "35mm lens", "shallow depth of field", "documentary style"
- Adapt to geographic/cultural context when location is specified"""

    def __init__(
        self,
        claude: ClaudeService,
        drive: GoogleDriveService | None = None,
        generator: ImageGenerator | None = None,
    ) -> None:
        self._claude = claude
        self._drive = drive
        self._generator = generator

    # ── Phase 1: Plan ─────────────────────────────────────────────────────────

    def plan(self, article: Article) -> ImagePlacementPlan:
        """
        Analyze the article and produce an image placement plan.

        Makes one Claude call. Returns an ImagePlacementPlan with the full
        list of ImageRequest objects and modified_markdown with markers.
        """
        context = self._build_context(article)
        user_prompt = self._build_planning_prompt(article, context)

        logger.info("Planning images for article '%s'...", article.title)
        data = self._claude.generate_structured(
            system=_PLANNING_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            tool_name="create_image_plan",
            tool_description="Create the image placement plan for this article.",
            input_schema=_PLANNING_SCHEMA,
            max_tokens=4096,
            thinking=True,
        )

        plan = self._parse_plan(data, article.content_markdown)
        logger.info(
            "Image plan: %d request(s) — %s",
            len(plan.requests),
            ", ".join(f"{r.id}({r.image_type.value})" for r in plan.requests),
        )
        return plan

    def _build_context(self, article: Article) -> ImageContext:
        location = None
        if article.request.location:
            parts = [
                article.request.location.city,
                article.request.location.state,
            ]
            location = ", ".join(p for p in parts if p) or None

        excerpt_words = article.content_markdown.split()[:300]
        excerpt = " ".join(excerpt_words)

        return ImageContext(
            title=article.title,
            focus_keyword=article.seo.focus_keyword,
            service=article.request.service,
            location=location,
            category=article.seo.suggested_category,
            content_excerpt=excerpt,
            tone=article.request.tone,
            client_id=article.tenant.client_id,
            website_id=article.tenant.website_id,
        )

    def _build_planning_prompt(self, article: Article, context: ImageContext) -> str:
        lines = [
            f"Article title: {article.title}",
            f"Focus keyword: {context.focus_keyword}",
            f"Word count: {article.word_count}",
        ]
        if context.service:
            lines.append(f"Service: {context.service}")
        if context.location:
            lines.append(f"Location: {context.location}")
        if context.category:
            lines.append(f"Category: {context.category}")
        lines.append(f"Tone: {context.tone.value}")
        lines.append("")
        lines.append("Full article content:")
        lines.append("---")
        lines.append(article.content_markdown)

        return "\n".join(lines)

    def _parse_plan(self, data: dict, original_markdown: str) -> ImagePlacementPlan:
        """Parse and validate the planning tool response."""
        requests = [
            ImageRequest(
                id=r["id"],
                purpose=ImagePurpose(r["purpose"]),
                image_type=ImageType(r["image_type"]),
                section_title=r.get("section_title"),
                subject=r["subject"],
                communicative_intent=r["communicative_intent"],
                related_keyword=r.get("related_keyword"),
                alt_text=r["alt_text"],
                caption=r.get("caption"),
            )
            for r in data.get("requests", [])
        ]

        modified_markdown = data.get("modified_markdown", original_markdown)

        # Validate: ensure all inline markers are present
        inline_ids = {r.id for r in requests if r.purpose == ImagePurpose.INLINE}
        for req_id in inline_ids:
            marker = f"<!-- SEO_AGENT_IMAGE: {req_id} -->"
            if marker not in modified_markdown:
                logger.warning(
                    "Marker for %s missing from modified_markdown — appending at end.", req_id
                )
                modified_markdown = modified_markdown.rstrip() + f"\n\n{marker}\n"

        # Validate: ensure featured image has no marker in markdown
        featured_ids = {r.id for r in requests if r.purpose == ImagePurpose.FEATURED}
        for req_id in featured_ids:
            marker = f"<!-- SEO_AGENT_IMAGE: {req_id} -->"
            if marker in modified_markdown:
                logger.warning("Featured image marker found in markdown — removing.")
                modified_markdown = modified_markdown.replace(marker, "")

        return ImagePlacementPlan(
            requests=requests,
            modified_markdown=modified_markdown,
            reasoning=data.get("reasoning", ""),
        )

    # ── Phase 2: Resolve ──────────────────────────────────────────────────────

    def resolve_all(
        self,
        plan: ImagePlacementPlan,
        style_profile: VisualStyleProfile | None = None,
        drive_candidates: list[DriveFileInfo] | None = None,
    ) -> list[tuple[ImageRequest, ImageAsset]]:
        """
        Resolve every ImageRequest in the plan to an ImageAsset.

        Drive candidates are provided by the caller (pre-fetched from
        DriveImageIndex) so no Drive API calls happen inside this method.
        Each request is resolved independently — a failure on one does
        not abort the others.

        Args:
            plan:             Image placement plan produced by plan().
            style_profile:    Optional visual style profile for generation prompts
                              and candidate scoring.
            drive_candidates: Pre-fetched list[DriveFileInfo] from DriveImageIndex.
                              Pass an empty list or None to skip Drive and go
                              straight to AI generation.

        Returns:
            List of (ImageRequest, ImageAsset) pairs in plan order.
        """
        if self._drive is None and self._generator is None:
            raise ImageResolverError(
                "Cannot resolve images: no Drive service and no image generator are configured."
            )

        candidates = drive_candidates or []
        if candidates:
            logger.info("%d Drive candidates available from index.", len(candidates))
        else:
            logger.info("No Drive candidates — will generate all images.")

        used_drive_ids: set[str] = set()
        results: list[tuple[ImageRequest, ImageAsset]] = []
        for req in plan.requests:
            try:
                asset = self._resolve_one(req, style_profile, candidates, used_drive_ids)
                results.append((req, asset))
                if asset.source == ImageSource.DRIVE and asset.source_detail:
                    used_drive_ids.add(asset.source_detail)
            except ImageResolverError as exc:
                logger.warning("Skipping %s (no suitable source): %s", req.id, exc)
            except Exception as exc:
                logger.error("Failed to resolve image %s: %s", req.id, exc)
                raise

        return results

    def _resolve_one(
        self,
        req: ImageRequest,
        style: VisualStyleProfile | None,
        drive_candidates: list[DriveFileInfo],
        used_drive_ids: set[str] | None = None,
    ) -> ImageAsset:
        logger.info("Resolving %s (%s)...", req.id, req.image_type.value)

        if drive_candidates:
            asset = self._try_drive(req, style, drive_candidates, used_drive_ids)
            if asset is not None:
                logger.info("%s resolved from Drive.", req.id)
                return asset
            logger.info("%s: no suitable Drive image — generating.", req.id)

        if self._generator is not None:
            return self._generate(req, style)

        raise ImageResolverError(
            f"Cannot resolve {req.id}: no suitable Drive image found and no generator configured."
        )

    # ── Drive path ────────────────────────────────────────────────────────────

    def _try_drive(
        self,
        req: ImageRequest,
        style: VisualStyleProfile | None,
        candidates: list[DriveFileInfo],
        used_file_ids: set[str] | None = None,
    ) -> ImageAsset | None:
        # Exclude Drive files already selected for a previous image in this run.
        available = [c for c in candidates if c.file_id not in (used_file_ids or set())]
        # Rank all candidates by keyword overlap with the request, then take
        # the top N. Folder paths are descriptive (e.g. "Garage Doors/Springs/")
        # so token matching works even when filenames are arbitrary (IMG_1234.jpg).
        scored = sorted(
            ((c, _score_candidate(c, req)) for c in available),
            key=lambda x: x[1],
            reverse=True,
        )
        pool = scored[:_MAX_VISION_CANDIDATES]
        if not pool:
            return None

        # Download thumbnails, carrying the keyword score alongside each file.
        # Cached thumbnail_link URLs from Drive expire in ~1-2 hours. When the
        # cached URL fails (or is absent), fall back to a fresh URL fetched by
        # file_id via the authenticated Drive API.
        thumbnails: list[tuple[DriveFileInfo, bytes]] = []
        kw_scores: list[int] = []
        for file_info, kw_score in pool:
            assert self._drive is not None
            try:
                if file_info.thumbnail_link:
                    try:
                        thumb = self._drive.download_thumbnail(file_info.thumbnail_link, size=512)
                    except Exception:
                        # Cached URL likely expired — fetch a fresh one by file_id.
                        thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=512)
                else:
                    thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=512)
                thumbnails.append((file_info, thumb))
                kw_scores.append(kw_score)
            except Exception as exc:
                logger.warning("Thumbnail unavailable for '%s': %s", file_info.name, exc)

        if not thumbnails:
            return None

        # Claude Vision selects by visual content alone (no filenames shown).
        result = self._evaluate_thumbnails(thumbnails, req, style)
        if result is None:
            return None

        winner_index, similarity_score, vision_reasoning = result
        winner_file, _ = thumbnails[winner_index]
        winner_kw_score = kw_scores[winner_index]

        # Derive a human-readable selection reason from keyword score + vision.
        clean_reasoning = re.sub(r"\s+", " ", vision_reasoning).strip()[:100]
        if winner_kw_score >= 3:
            reason = f"Strong semantic match ({winner_kw_score} keywords in folder/path). {clean_reasoning}"
        elif winner_kw_score >= 1:
            reason = f"Folder context match ({winner_kw_score} keyword). {clean_reasoning}"
        else:
            reason = f"Best available candidate (visual match, no keyword overlap). {clean_reasoning}"

        assert self._drive is not None
        image_bytes = self._drive.download(winner_file.file_id)

        folder = winner_file.folder_path.strip("/")
        drive_path = f"{folder}/{winner_file.name}" if folder else winner_file.name

        return ImageAsset(
            filename=winner_file.name,
            mime_type=winner_file.mime_type,
            data=image_bytes,
            alt_text=req.alt_text,
            caption=req.caption,
            source=ImageSource.DRIVE,
            source_detail=winner_file.file_id,
            similarity_score=similarity_score,
            selection_reason=reason,
            drive_path=drive_path,
            vision_reasoning=vision_reasoning,
            drive_candidates_evaluated=len(thumbnails),
        )

    def _evaluate_thumbnails(
        self,
        thumbnails: list[tuple[DriveFileInfo, bytes]],
        req: ImageRequest,
        style: VisualStyleProfile | None,
    ) -> tuple[int, int, str] | None:
        """
        Ask Claude Vision to select the best image purely by visual content.

        Images are labelled Image 1, Image 2, … — filenames are never shown.

        Returns (0-based winner index, similarity_score 0-100, vision_reasoning)
        or None if no candidate is sufficiently relevant.
        """
        style_hint = (
            f"\nBrand visual style: {style.style_description}" if style else ""
        )
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"Select the best image for this placement:\n\n"
                f"Subject: {req.subject}\n"
                f"Type: {req.image_type.value}\n"
                f"Communicative intent: {req.communicative_intent}\n"
                f"Alt text: {req.alt_text}"
                f"{style_hint}"
                f"\n\n{len(thumbnails)} candidates below. "
                "Choose the one whose visual content best matches the subject and intent. "
                "Select 0 if none are sufficiently relevant."
            ),
        }]

        for i, (_, thumb_bytes) in enumerate(thumbnails, start=1):
            content.append({"type": "text", "text": f"\nImage {i}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(thumb_bytes),
                    "data": base64.standard_b64encode(thumb_bytes).decode(),
                },
            })

        data = self._claude.generate_structured(
            system=self._VISION_SYSTEM,
            messages=[{"role": "user", "content": content}],
            tool_name="select_image",
            tool_description="Select the best image by index, or 0 if none are suitable.",
            input_schema={
                "type": "object",
                "required": ["selected_index", "similarity_score", "reasoning"],
                "properties": {
                    "selected_index": {
                        "type": "integer",
                        "description": (
                            f"1-based index of the selected image (1–{len(thumbnails)}). "
                            "Return 0 if no candidate matches well enough."
                        ),
                    },
                    "similarity_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": (
                            "Visual similarity score 0–100: how well the selected image matches "
                            "the required subject and communicative intent. "
                            "0 = no match, 100 = perfect match. Return 0 if selected_index is 0."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this image was selected.",
                    },
                },
            },
            thinking=False,
        )

        idx = data.get("selected_index", 0)
        score = max(0, min(100, int(data.get("similarity_score", 0))))
        reasoning = data.get("reasoning", "")
        logger.debug("Vision: index=%d score=%d — %s", idx, score, reasoning[:80])
        if idx < 1 or idx > len(thumbnails):
            return None
        return idx - 1, score, reasoning  # convert to 0-based

    # ── Generation path ───────────────────────────────────────────────────────

    def _generate(self, req: ImageRequest, style: VisualStyleProfile | None) -> ImageAsset:
        prompt = self._build_generation_prompt(req, style)
        gen_req = ImageGenerationRequest(
            prompt=prompt,
            alt_text=req.alt_text,
            size="1792x1024",
        )
        assert self._generator is not None
        asset = self._generator.generate(gen_req)
        # Override alt_text, caption, and audit fields with values from planning.
        return asset.model_copy(update={
            "alt_text": req.alt_text,
            "caption": req.caption,
            "source_detail": prompt[:500],
            "similarity_score": None,
            "selection_reason": "Generated — no suitable Drive image found in index.",
            "drive_path": None,
            "vision_reasoning": None,
            "drive_candidates_evaluated": None,
        })

    def _build_generation_prompt(
        self, req: ImageRequest, style: VisualStyleProfile | None
    ) -> str:
        data = self._claude.generate_structured(
            system=self._GEN_PROMPT_SYSTEM,
            messages=[{
                "role": "user",
                "content": "\n".join(filter(None, [
                    f"Write a DALL-E 3 prompt for this image:\n",
                    f"Subject: {req.subject}",
                    f"Image type: {req.image_type.value}",
                    f"Communicative intent: {req.communicative_intent}",
                    f"Alt text context: {req.alt_text}",
                    (f"\nVisual style guidelines to match:\n{style.prompt_guidelines}") if style else None,
                    "\nThe prompt must produce a photorealistic photograph, not an illustration or render.",
                ]))
            }],
            tool_name="create_dalle_prompt",
            tool_description="Create the DALL-E 3 prompt.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The complete DALL-E 3 generation prompt."
                    }
                },
                "required": ["prompt"]
            },
            max_tokens=1024,
            thinking=False,
        )
        prompt = data["prompt"]
        logger.debug("Generation prompt for %s: %s", req.id, prompt[:120])
        return prompt

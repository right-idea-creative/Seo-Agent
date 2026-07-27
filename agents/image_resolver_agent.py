"""
ImageResolverAgent — Photo Preservation Pipeline for article publishing.

Phase 1 — plan():
    Claude analyzes the full article as a professional editor and returns:
    - A list[ImageRequest] specifying what each image should depict and why.
    - modified_markdown: content_markdown with <!-- SEO_AGENT_IMAGE: id -->
      markers inserted at inline image positions.

Phase 2 — resolve_all():
    For each ImageRequest, the agent applies a two-tier preservation strategy:

    Priority 1 — Original company photo (score ≥ exact_score, default 75)
        Claude Vision finds a Drive photo that matches the article context.
        The original photograph is published without modification.
        Real company photography is always the best possible result.

    Priority 2 — Minimal preservation edit (partial_score ≤ score < exact_score)
        A Drive photo is relevant but imperfect for the slot.
        Claude identifies the SMALLEST edit needed: remove a distraction,
        extend canvas for layout space, adjust lighting, clean a background region.
        gpt-image-1 images.edit() applies ONLY that edit to the original photograph.
        Approximately 95–99% of the original pixels are preserved.
        The AI is a Photoshop editor, not a photographer.
        Consumes one unit of the per-article edit budget.

    No Priority 3:
        The pipeline never generates images from scratch.
        If no relevant Drive photo exists (score == 0), the image slot is skipped.
        One authentic company photograph is worth more than any AI-generated scene.

    Fallback:
        If the edit budget is exhausted, the best Drive photo is published as-is
        regardless of score. A real company photo at any score beats no photo.
        If Drive has no candidates at all for this slot, the slot is skipped.

The agent never generates new scenes, new technicians, new houses, or new trucks.
It preserves and adapts the company's existing photographic archive.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any

from agents._marker_utils import insert_marker_at_section
from agents.editorial_scoring import EditorialSelectionResult, score_candidates
from config import settings as _settings
from models.article import Article
from models.image_asset import ImageAsset, ImageSource
from models.image_context import ImageContext
from models.image_request import ImagePlacementPlan, ImagePurpose, ImageRequest, ImageType
from services.claude_service import ClaudeService
from services.editorial_history_service import EditorialHistoryService
from services.google_drive_service import DriveFileInfo, GoogleDriveService
from services.image_generators import ImageGenerationRequest, ImageGenerator

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ImageResolverError(Exception):
    """Raised when an image cannot be resolved from any available source."""


# Max Drive candidates sent to Claude Vision per image request.
# Thresholds (_SCORE_EXACT, _SCORE_PARTIAL, _MAX_AI_PER_ARTICLE) are now
# constructor parameters on ImageResolverAgent — see config.py for defaults.
_MAX_VISION_CANDIDATES = 15


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


# ── Module-level helpers ──────────────────────────────────────────────────────

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
    Photo Preservation Pipeline — two-phase image resolution for article publishing.

    Phase 1 — plan(article):
        Claude analyzes the article and returns an ImagePlacementPlan.

    Phase 2 — resolve_all(plan, drive_candidates):
        Two-tier preservation strategy per image request:
        P1: Original Drive photo (score ≥ exact_score) — published as-is, never touched.
        P2: Minimal preservation edit (partial_score ≤ score < exact_score) — Claude
            identifies the smallest edit; images.edit() applies ONLY that edit.
            95–99% of original pixels preserved.
        Fallback: best Drive photo as-is if budget is exhausted.
        Skip: if no Drive candidates match at all (score == 0), the slot is skipped.
              The pipeline NEVER generates images from scratch.
    """

    _VISION_SYSTEM = (
        "You are a photo curator selecting the most relevant company photograph "
        "for an article image slot.\n\n"
        "Your score determines the publication path:\n"
        "  ≥ 75: publish the original photograph exactly as-is.\n"
        "  40–74: publish the photograph with one minimal preservation edit "
        "(remove a distraction, extend canvas, adjust lighting in one area).\n"
        "  1–39: publish the original photograph as-is — it is relevant but a weak match.\n"
        "  0: the photograph is completely unrelated to the article topic.\n\n"
        "A real company photograph at any score > 0 is always preferred. "
        "Score 0 ONLY when all candidates are completely irrelevant — "
        "wrong industry, wrong subject, no conceivable connection."
    )

    _PRESERVATION_SYSTEM = """\
You are a photo editor for a professional service company's content archive.

Your responsibility: identify the SMALLEST edit that makes an authentic company
photograph more useful for a specific article image slot.

You are NOT a photographer, illustrator, or artist.
You are NOT generating a new scene.
You are editing an existing photograph with minimal intervention.

Valid edits (one per image):
• Remove a specific distracting object (trash bin, parked car, utility cable, garbage bag)
• Extend canvas for layout headroom (expand sky, driveway, or side margin)
• Adjust exposure or white balance in a specific area only
• Remove personally identifying information (license plate, visible face in background)
• Convert time of day (day to night, overcast to sunny) — scene elements unchanged
• Clean a cluttered background region
• Remove an unwanted person at the edge of the frame

NEVER suggest:
• Replacing the technician, house, truck, garage door, or driveway
• Adding new scene elements not present in the original photograph
• Redesigning the composition
• Generating new visual content of any kind
• Changing the fundamental scene

The edit_instruction must be ONE specific sentence.
Examples:
  "Remove the red sedan parked at the right edge of the driveway."
  "Extend the sky region upward by 20% to add layout headroom."
  "Remove the garbage bag visible near the left side of the garage."
  "Convert to night lighting while preserving all objects and people exactly."

Nothing more than the minimum necessary change.\
"""

    def __init__(
        self,
        claude: ClaudeService,
        drive: GoogleDriveService | None = None,
        generator: ImageGenerator | None = None,
        *,
        exact_score: int = 75,
        partial_score: int = 40,
        max_ai: int = 1,
        editorial_history: EditorialHistoryService | None = None,
    ) -> None:
        self._claude = claude
        self._drive = drive
        self._generator = generator
        self._exact_score = exact_score
        self._partial_score = partial_score
        self._max_ai = max_ai
        self._editorial_history = editorial_history
        # Populated after resolve_all() — read by the caller for pipeline reporting.
        self.last_run_stats: dict = {}
        # Per-resolve_all accumulators for diversity stats; reset at start of each call.
        self._editorial_acc: dict = {}

    # ── Phase 1: Plan ─────────────────────────────────────────────────────────

    def plan(self, article: Article) -> ImagePlacementPlan:
        """
        Produce an image placement plan for the article.

        When the article carries an image_plans list from the ArticlePlannerService,
        those planned images are used directly — no Claude API call is made.
        The planner already reasoned about WHY each image exists and WHAT to show;
        this method converts that into ImageRequest objects and inserts inline markers.

        When no image_plans are present, falls back to a full Claude planning call
        that analyzes the article markdown and decides placement from scratch.
        """
        if article.image_plans:
            logger.info(
                "Using %d image(s) from ArticlePlan — skipping Claude planning call.",
                len(article.image_plans),
            )
            return self._plan_from_article_plan(article)

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
            model=_settings.image_eval_model,
            label="image:plan",
        )

        plan = self._parse_plan(data, article.content_markdown)
        logger.info(
            "Image plan: %d request(s) — %s",
            len(plan.requests),
            ", ".join(f"{r.id}({r.image_type.value})" for r in plan.requests),
        )
        return plan

    def _plan_from_article_plan(self, article: Article) -> ImagePlacementPlan:
        """
        Convert the article's PlannedImage list into an ImagePlacementPlan.

        No Claude call — the planner already produced the image intent. This method
        translates PlannedImage objects into ImageRequest objects (compatible with the
        existing Drive-matching and editing pipeline) and inserts inline markers.
        """
        requests: list[ImageRequest] = []
        for planned in article.image_plans:
            try:
                image_type = ImageType(planned.image_type)
            except ValueError:
                logger.warning(
                    "Unknown image_type '%s' for %s — falling back to 'photograph'.",
                    planned.image_type, planned.image_id,
                )
                image_type = ImageType.PHOTOGRAPH

            requests.append(ImageRequest(
                id=planned.image_id,
                purpose=ImagePurpose(planned.purpose),
                image_type=image_type,
                section_title=planned.section_anchor,
                subject=planned.subject,
                communicative_intent=planned.why,
                related_keyword=article.seo.focus_keyword,
                alt_text=planned.alt_text,
                caption=planned.caption or None,
            ))

        modified_markdown = article.content_markdown
        for req in (r for r in requests if r.purpose == ImagePurpose.INLINE):
            marker = req.placement_marker
            if marker not in modified_markdown:
                modified_markdown = insert_marker_at_section(
                    modified_markdown, marker, req.section_title
                )

        logger.info(
            "Image plan from ArticlePlan: %d request(s) — %s",
            len(requests),
            ", ".join(f"{r.id}({r.image_type.value})" for r in requests),
        )
        return ImagePlacementPlan(
            requests=requests,
            modified_markdown=modified_markdown,
            reasoning="Image placement consumed from ArticlePlan (planner-specified intent).",
        )

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

        for req in (r for r in requests if r.purpose == ImagePurpose.INLINE):
            marker = f"<!-- SEO_AGENT_IMAGE: {req.id} -->"
            if marker not in modified_markdown:
                logger.warning(
                    "Marker for %s missing from modified_markdown — inserting before section '%s'.",
                    req.id, req.section_title or "(unknown)",
                )
                modified_markdown = insert_marker_at_section(
                    modified_markdown, marker, req.section_title
                )

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
        drive_candidates: list[DriveFileInfo] | None = None,
    ) -> list[tuple[ImageRequest, ImageAsset]]:
        """
        Resolve every ImageRequest in the plan to an ImageAsset.

        Two-tier preservation strategy (applied per image, shared edit budget):
          P1: Drive original (score ≥ exact_score) — published as-is, never touched.
          P2: Drive partial match (partial_score ≤ score < exact_score, budget > 0) —
              Claude identifies the minimal edit; images.edit() applies only that.
              ~95–99% of original pixels preserved.
          Weak match (0 < score < partial_score): Drive photo as-is (no edit needed).
          Score == 0: slot skipped — no Drive match and no generation.
          Budget exhausted: Drive photo as-is regardless of score.
        """
        if self._drive is None and self._generator is None:
            raise ImageResolverError(
                "Cannot resolve images: no Drive service and no image generator configured."
            )

        candidates = drive_candidates or []

        if candidates:
            logger.info(
                "%d Drive candidates. Thresholds: exact=%d, partial=%d. Edit budget: %d/article.",
                len(candidates), self._exact_score, self._partial_score, self._max_ai,
            )
        else:
            logger.info("No Drive candidates — all image slots will be skipped.")

        edit_budget: list[int] = [self._max_ai]
        semantic_candidates_total: list[int] = [0]
        used_drive_ids: set[str] = set()
        used_folder_paths: set[str] = set()
        results: list[tuple[ImageRequest, ImageAsset]] = []
        self._editorial_acc = {
            "candidates_considered": 0,
            "excluded_recent": 0,
            "excluded_folder_duplicate": 0,
        }
        editorial_selections: list[dict] = []

        for req in plan.requests:
            try:
                asset, editorial_result = self._resolve_one(
                    req, candidates, used_drive_ids, used_folder_paths,
                    edit_budget, semantic_candidates_total,
                    is_featured=(req.purpose == ImagePurpose.FEATURED),
                )
                results.append((req, asset))

                if editorial_result is not None:
                    editorial_selections.append({
                        "slot": req.id,
                        "purpose": req.purpose.value,
                        "vision_score": editorial_result.vision_score,
                        "editorial_score": round(editorial_result.editorial_score, 2),
                        "times_used": editorial_result.times_used,
                        "reason": editorial_result.selection_reason,
                    })

                if asset.source == ImageSource.DRIVE and asset.source_detail:
                    used_drive_ids.add(asset.source_detail)
                elif asset.source == ImageSource.EDITED and asset.reference_file_id:
                    used_drive_ids.add(asset.reference_file_id)

                if asset.source in (ImageSource.DRIVE, ImageSource.EDITED):
                    folder = (asset.drive_path or "").rsplit("/", 1)[0]
                    if folder:
                        used_folder_paths.add(folder)

            except ImageResolverError as exc:
                logger.warning("Skipping %s (no suitable Drive photo): %s", req.id, exc)
            except Exception as exc:
                logger.error("Failed to resolve image %s: %s", req.id, exc)
                raise

        drive_used = sum(1 for _, a in results if a.source == ImageSource.DRIVE)
        edited     = sum(1 for _, a in results if a.source == ImageSource.EDITED)
        edit_used  = edited
        logger.info(
            "Resolution complete: %d Drive original, %d preservation edit, %d skipped.",
            drive_used, edited, len(plan.requests) - len(results),
        )

        previously_unused = sum(
            1 for sel in editorial_selections if sel.get("times_used", 1) == 0
        )
        self.last_run_stats = {
            "drive_candidates_total":    len(candidates),
            "drive_semantic_candidates": semantic_candidates_total[0],
            "drive_originals_used":      drive_used,
            "preservation_edits":        edited,
            "slots_skipped":             len(plan.requests) - len(results),
            "edit_budget_used":          edit_used,
            "edit_budget_total":         self._max_ai,
            "edit_budget_remaining":     max(0, self._max_ai - edit_used),
            "edited_photos": [
                {
                    "id":         req.id,
                    "drive_path": asset.drive_path or "",
                    "score":      asset.similarity_score,
                    "edit_type":  asset.edit_type or "",
                    "preserved":  asset.preservation_estimate,
                    "reason":     asset.ai_reason or "",
                }
                for req, asset in results
                if asset.source == ImageSource.EDITED
            ],
            "diversity_report": {
                "candidates_considered":     self._editorial_acc.get("candidates_considered", 0),
                "excluded_recent":           self._editorial_acc.get("excluded_recent", 0),
                "excluded_folder_duplicate": self._editorial_acc.get("excluded_folder_duplicate", 0),
                "previously_unused":         previously_unused,
                "selections":                editorial_selections,
            },
        }
        return results

    def _resolve_one(
        self,
        req: ImageRequest,
        drive_candidates: list[DriveFileInfo],
        used_drive_ids: set[str],
        used_folder_paths: set[str],
        edit_budget: list[int],
        semantic_candidates_total: list[int],
        *,
        is_featured: bool = False,
    ) -> tuple[ImageAsset, EditorialSelectionResult | None]:
        logger.info("Resolving %s (%s)...", req.id, req.image_type.value)

        drive_result = None
        if drive_candidates and self._drive is not None:
            drive_result = self._try_drive_with_score(
                req, drive_candidates, used_drive_ids, semantic_candidates_total,
                is_featured=is_featured,
                used_folder_paths=used_folder_paths,
            )

        if drive_result is None:
            raise ImageResolverError(
                f"No Drive candidates available for {req.id}."
            )

        asset, score, editorial_result = drive_result

        # ── P1: Drive original as-is ──────────────────────────────────────────
        if score >= self._exact_score:
            logger.info("%s → P1: Drive original as-is (score=%d).", req.id, score)
            return asset, editorial_result

        # ── Score == 0: no relevant photo found ───────────────────────────────
        if score == 0:
            raise ImageResolverError(
                f"Claude Vision found no relevant Drive photo for {req.id} (score=0). "
                "Slot skipped — pipeline does not generate images from scratch."
            )

        # ── P2: Minimal preservation edit ─────────────────────────────────────
        if score >= self._partial_score and edit_budget[0] > 0 and self._generator is not None:
            logger.info(
                "%s → P2: preservation edit (score=%d, budget=%d remaining).",
                req.id, score, edit_budget[0],
            )
            edit_budget[0] -= 1
            return self._minimal_edit(req, asset), editorial_result

        # ── Fallback: weak match or budget exhausted → Drive photo as-is ──────
        reason = (
            "edit budget exhausted" if edit_budget[0] == 0 else f"weak match (score={score})"
        )
        logger.info("%s → Fallback: Drive photo as-is (%s).", req.id, reason)
        return asset.model_copy(update={
            "selection_reason": (asset.selection_reason or "")
            + f" [Drive photo used as-is: {reason}]",
        }), editorial_result

    # ── Drive path ────────────────────────────────────────────────────────────

    def _try_drive_with_score(
        self,
        req: ImageRequest,
        candidates: list[DriveFileInfo],
        used_file_ids: set[str],
        semantic_candidates_total: list[int],
        *,
        is_featured: bool = False,
        used_folder_paths: set[str] | None = None,
    ) -> tuple[ImageAsset, int, EditorialSelectionResult | None] | None:
        available = [c for c in candidates if c.file_id not in used_file_ids]
        scored = sorted(
            ((c, _score_candidate(c, req)) for c in available),
            key=lambda x: x[1],
            reverse=True,
        )

        semantic_hits = sum(1 for _, s in scored if s > 0)
        semantic_candidates_total[0] += semantic_hits

        pool = scored[:_MAX_VISION_CANDIDATES]
        if not pool:
            return None

        thumbnails: list[tuple[DriveFileInfo, bytes]] = []
        kw_scores: list[int] = []
        for file_info, kw_score in pool:
            assert self._drive is not None
            try:
                if file_info.thumbnail_link:
                    try:
                        thumb = self._drive.download_thumbnail(file_info.thumbnail_link, size=512)
                    except Exception:
                        thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=512)
                else:
                    thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=512)
                thumbnails.append((file_info, thumb))
                kw_scores.append(kw_score)
            except Exception as exc:
                logger.warning("Thumbnail unavailable for '%s': %s", file_info.name, exc)

        if not thumbnails:
            return None

        scored_list, vision_reasoning = self._evaluate_thumbnails(thumbnails, req)
        self._editorial_acc["candidates_considered"] = (
            self._editorial_acc.get("candidates_considered", 0) + len(thumbnails)
        )

        if not scored_list:
            return None

        winner_file: DriveFileInfo
        winner_kw_score: int
        similarity_score: int
        editorial_result: EditorialSelectionResult | None = None

        if self._editorial_history is not None:
            try:
                history_lookup = {
                    thumbnails[idx][0].file_id: self._editorial_history.get_record(
                        thumbnails[idx][0].file_id
                    )
                    for idx, _ in scored_list
                    if 0 <= idx < len(thumbnails)
                }
                recent_articles = self._editorial_history.get_recent_articles(25)
                edit_candidates = [
                    (
                        thumbnails[idx][0].file_id,
                        thumbnails[idx][0].name,
                        thumbnails[idx][0].folder_path,
                        vscore,
                    )
                    for idx, vscore in scored_list
                    if 0 <= idx < len(thumbnails)
                ]
                editorial_results = score_candidates(
                    edit_candidates,
                    history_lookup,
                    recent_articles,
                    is_featured=is_featured,
                    used_folder_paths=used_folder_paths or set(),
                )
                if editorial_results:
                    winner_ed = editorial_results[0]
                    for r in editorial_results[1:]:
                        if r.recent_penalty > 0:
                            self._editorial_acc["excluded_recent"] = (
                                self._editorial_acc.get("excluded_recent", 0) + 1
                            )
                        if r.folder_penalty > 0:
                            self._editorial_acc["excluded_folder_duplicate"] = (
                                self._editorial_acc.get("excluded_folder_duplicate", 0) + 1
                            )
                    winner_thumb_idx = next(
                        (
                            i for i, (fi, _) in enumerate(thumbnails)
                            if fi.file_id == winner_ed.file_id
                        ),
                        scored_list[0][0],
                    )
                    winner_file = thumbnails[winner_thumb_idx][0]
                    winner_kw_score = kw_scores[winner_thumb_idx]
                    similarity_score = winner_ed.vision_score
                    editorial_result = winner_ed
                else:
                    winner_thumb_idx = scored_list[0][0]
                    winner_file = thumbnails[winner_thumb_idx][0]
                    winner_kw_score = kw_scores[winner_thumb_idx]
                    similarity_score = scored_list[0][1]
            except Exception as exc:
                logger.warning("Editorial scoring failed, falling back to Vision winner: %s", exc)
                winner_thumb_idx = scored_list[0][0]
                winner_file = thumbnails[winner_thumb_idx][0]
                winner_kw_score = kw_scores[winner_thumb_idx]
                similarity_score = scored_list[0][1]
        else:
            # No history service — use highest Vision score.
            winner_thumb_idx = scored_list[0][0]
            winner_file = thumbnails[winner_thumb_idx][0]
            winner_kw_score = kw_scores[winner_thumb_idx]
            similarity_score = scored_list[0][1]

        clean_reasoning = re.sub(r"\s+", " ", vision_reasoning).strip()[:100]
        if winner_kw_score >= 3:
            reason = f"Strong semantic match ({winner_kw_score} keywords). {clean_reasoning}"
        elif winner_kw_score >= 1:
            reason = f"Folder context match ({winner_kw_score} keyword). {clean_reasoning}"
        else:
            reason = f"Best available candidate (visual match, no keyword overlap). {clean_reasoning}"

        if similarity_score == 0:
            reason = f"No suitable Drive image found by Claude Vision. {clean_reasoning}"

        assert self._drive is not None
        image_bytes = self._drive.download(winner_file.file_id)

        folder = winner_file.folder_path.strip("/")
        drive_path = f"{folder}/{winner_file.name}" if folder else winner_file.name

        asset = ImageAsset(
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
        return asset, similarity_score, editorial_result

    def _evaluate_thumbnails(
        self,
        thumbnails: list[tuple[DriveFileInfo, bytes]],
        req: ImageRequest,
    ) -> tuple[list[tuple[int, int]], str]:
        """
        Ask Claude Vision to score all Drive candidates independently.

        Never shows filenames — selection is purely visual.

        Returns ([(0-based index, score), ...] sorted best-first, reasoning).
        Falls back to [(0, 0)] with a message if no valid scores are returned.
        """
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"Score each Drive photograph for this image slot:\n\n"
                f"Subject: {req.subject}\n"
                f"Type: {req.image_type.value}\n"
                f"Communicative intent: {req.communicative_intent}\n"
                f"Alt text: {req.alt_text}"
                f"\n\n{len(thumbnails)} candidates below. "
                "Score each image independently from 0–100 for how well it matches the slot. "
                "Return a score for every image. "
                "Any real company photograph (score 1–100) is always preferred. "
                "Score 0 ONLY if an image is completely wrong industry or irrelevant subject."
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
            tool_name="score_images",
            tool_description="Score each image independently for how well it matches the slot.",
            input_schema={
                "type": "object",
                "required": ["candidate_scores", "reasoning"],
                "properties": {
                    "candidate_scores": {
                        "type": "array",
                        "description": "Score for every image, one entry per candidate.",
                        "items": {
                            "type": "object",
                            "required": ["index", "score"],
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "description": (
                                        f"1-based image index (1–{len(thumbnails)})."
                                    ),
                                },
                                "score": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                    "description": (
                                        "How well this image matches (0–100):\n"
                                        "  ≥75: excellent — original photo published as-is\n"
                                        "  40–74: partial — original photo with one minimal preservation edit\n"
                                        "  1–39: weak — original photo published as-is (no edit needed)\n"
                                        "  0: completely irrelevant"
                                    ),
                                },
                            },
                        },
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the scoring rationale.",
                    },
                },
            },
            thinking=False,
            model=_settings.image_eval_model,
            label="image:vision-score",
        )

        reasoning = data.get("reasoning", "")
        raw_scores = data.get("candidate_scores", [])

        scored: list[tuple[int, int]] = []
        for entry in raw_scores:
            try:
                one_based = int(entry.get("index", 0))
                score = max(0, min(100, int(entry.get("score", 0))))
                if 1 <= one_based <= len(thumbnails):
                    scored.append((one_based - 1, score))  # convert to 0-based
            except (TypeError, ValueError):
                continue

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "Vision scored %d/%d candidates — %s", len(scored), len(thumbnails), reasoning[:80]
        )

        if not scored:
            return [(0, 0)], reasoning or "No suitable images found."

        return scored, reasoning

    # ── Preservation edit path (Priority 2) ──────────────────────────────────

    _PRESERVATION_PROMPT_PREFIX = """\
This is an authentic company photograph.

Your job is NOT to create a new image.

Your job is NOT to redesign this photograph.

Preserve every original pixel unless absolutely necessary.

Treat this as a professional Photoshop retouch.

The original photograph is the source of truth.

Modify only the minimum number of pixels required to complete the requested edit.

If an area does not need to change, leave it unchanged.

Never repaint the entire image.

Never reinterpret materials.

Never redesign reflections.

Never redesign textures.

Never improve the composition.

Never create a more beautiful photograph.

Only perform the requested edit.

The goal is that someone comparing the original and edited photographs cannot tell where AI was used.

If the requested edit cannot be completed without visibly changing unrelated parts of the photograph, DO NOT attempt a creative solution. Preserve the original photograph. It is preferable to leave the image unchanged than to introduce synthetic artifacts. Authenticity always takes priority over completing the edit.

Requested edit: """

    def _minimal_edit(self, req: ImageRequest, reference_asset: ImageAsset) -> ImageAsset:
        """
        Apply a single minimal preservation edit to the original Drive photograph.

        Step 1: Claude (_PRESERVATION_SYSTEM) identifies the smallest edit that
                makes the photo more useful for this image slot.
        Step 2: images.edit() applies ONLY that edit to the original photograph.
        Step 3: Returns an EDITED ImageAsset with original_data set for QA comparison.

        95–99% of the original pixels are preserved.
        """
        if not hasattr(self._generator, "generate_variation"):
            logger.warning(
                "Generator lacks generate_variation() (images.edit) — "
                "using Drive photo as-is for %s.", req.id,
            )
            return reference_asset.model_copy(update={
                "selection_reason": (reference_asset.selection_reason or "")
                + " [edit not supported by generator — used as-is]",
            })

        edit_data = self._build_preservation_edit_prompt(req, reference_asset)
        edit_instruction = edit_data["edit_instruction"]
        edit_type = edit_data.get("edit_type", "general_cleanup")
        preservation_estimate = edit_data.get("preservation_estimate", 95)

        full_prompt = self._PRESERVATION_PROMPT_PREFIX + edit_instruction
        logger.info(
            "%s → preservation edit (%s): %s", req.id, edit_type, edit_instruction
        )

        gen_req = ImageGenerationRequest(
            prompt=full_prompt,
            alt_text=req.alt_text,
            size="1536x1024",
        )
        assert self._generator is not None
        edited_asset = self._generator.generate_variation(  # type: ignore[attr-defined]
            reference_images=[reference_asset.data],
            request=gen_req,
            variation_prompt=full_prompt,
        )

        ref_path = reference_asset.drive_path or ""
        return edited_asset.model_copy(update={
            "source":           ImageSource.EDITED,
            "alt_text":         req.alt_text,
            "caption":          req.caption,
            "source_detail":    full_prompt[:500],
            "reference_file_id": reference_asset.source_detail,
            "ai_reason":        (
                f"Partial Drive match (score={reference_asset.similarity_score}) — "
                f"preservation edit applied: {edit_instruction}"
            ),
            "similarity_score":  reference_asset.similarity_score,
            "selection_reason":  f"Original Drive photo with minimal {edit_type} edit: {ref_path}",
            "drive_path":        ref_path,
            "vision_reasoning":  reference_asset.vision_reasoning,
            "drive_candidates_evaluated": reference_asset.drive_candidates_evaluated,
            "original_data":     reference_asset.data,
            "edit_type":         edit_type,
            "edit_prompt":       full_prompt,
            "preservation_estimate": preservation_estimate,
        })

    def _build_preservation_edit_prompt(
        self, req: ImageRequest, reference_asset: ImageAsset
    ) -> dict:
        """
        Ask Claude to identify the smallest edit that improves the photo for this slot.

        Uses already-captured vision_reasoning so no image re-download is needed.
        Returns a dict with edit_instruction, edit_type, and preservation_estimate.
        """
        data = self._claude.generate_structured(
            system=self._PRESERVATION_SYSTEM,
            messages=[{
                "role": "user",
                "content": "\n".join([
                    "Identify the smallest edit to make this company photograph more useful.",
                    "",
                    f"Photo: {reference_asset.drive_path or 'company field photo'}",
                    f"Visual match score: {reference_asset.similarity_score}/100",
                    f"What Claude saw: {(reference_asset.vision_reasoning or '')[:200]}",
                    "",
                    f"Target image slot: {req.subject}",
                    f"Image type: {req.image_type.value}",
                    f"Communicative intent: {req.communicative_intent}",
                    "",
                    "What is the ONE smallest edit that would make this photograph more useful "
                    "for this specific slot? Remember: the AI is a Photoshop editor. "
                    "Do NOT suggest replacing any core element.",
                ]),
            }],
            tool_name="identify_preservation_edit",
            tool_description="Identify the minimal preservation edit for this photograph.",
            input_schema={
                "type": "object",
                "required": ["edit_instruction", "edit_type", "preservation_estimate"],
                "properties": {
                    "edit_instruction": {
                        "type": "string",
                        "description": (
                            "ONE specific sentence describing the exact edit. "
                            "Examples: 'Remove the red sedan parked at the right edge of the driveway.' "
                            "'Extend the sky region upward by 20% to add layout headroom.' "
                            "'Convert to night lighting while preserving all objects and people exactly.'"
                        ),
                    },
                    "edit_type": {
                        "type": "string",
                        "enum": [
                            "remove_object", "canvas_extension", "exposure_adjustment",
                            "background_cleanup", "day_to_night", "color_correction",
                            "remove_text", "crop_adjustment", "general_cleanup",
                        ],
                        "description": "Category of edit being applied.",
                    },
                    "preservation_estimate": {
                        "type": "integer",
                        "minimum": 90,
                        "maximum": 100,
                        "description": (
                            "Estimated percentage of original pixels preserved after this edit (90–100). "
                            "A background-only tweak = 98+. Canvas extension = 95. "
                            "Day-to-night = 92 (lighting changes everything but structure stays)."
                        ),
                    },
                },
            },
            max_tokens=400,
            thinking=False,
            model=_settings.image_eval_model,
            label="image:edit-prompt",
        )
        logger.debug(
            "Preservation edit for %s: [%s] %s (est. %d%% preserved)",
            req.id, data.get("edit_type"), data.get("edit_instruction", "")[:80],
            data.get("preservation_estimate", 0),
        )
        return data

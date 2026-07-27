"""
VisualIdentityService — article-specific visual identity extraction.

Philosophy:
  Google Drive is NOT an image repository.
  Google Drive IS the visual identity of the company.

  Before generating any image, this service retrieves the TOP 10–20 Drive
  photographs most semantically relevant to the article being produced,
  analyzes them with Claude Vision, and extracts a VisualIdentityProfile.

  That profile becomes the single source of truth for all AI generation:
  - Priority 2 (variation): reference photos + profile drive images.edit()
  - Priority 3 (scratch): profile drives generation prompt
  - Vision QA: reviewers compare against profile + references

  The AI never invents a new visual identity. It learns the existing one.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING

from models.visual_identity import VisualIdentityProfile
from services.google_drive_service import DriveFileInfo, GoogleDriveService
from services.claude_service import ClaudeService

if TYPE_CHECKING:
    from models.article import Article

logger = logging.getLogger(__name__)

# How many Drive candidates to analyze (top N by semantic score)
_MAX_CANDIDATES = 20
# How many to send to Claude Vision after filtering
_MAX_ANALYSIS = 15

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "it", "its",
    "img", "dsc", "jpg", "jpeg", "png", "webp", "gif", "heic",
})


def _tokenize(text: str) -> frozenset[str]:
    text = re.sub(r"[\\/]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens: set[str] = set()
    for w in text.split():
        if w in _STOP_WORDS or len(w) < 3:
            continue
        tokens.add(w)
        if w.endswith("s") and len(w) > 3:
            tokens.add(w[:-1])
    return frozenset(tokens)


def _detect_media_type(data: bytes) -> str:
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


# ── Pass 1: Filter schema ─────────────────────────────────────────────────────

_FILTER_SYSTEM = """\
You are a visual content classifier for a business photo library.

For each image: decide whether it is a usable company photograph or should be excluded.

INCLUDE (include: true):
  Real photographs of work being performed, finished results, job sites, equipment in use,
  team members working, before/after results, any documentary photo of company services.

EXCLUDE (include: false):
  Logos, icons, brand marks, website screenshots, banners, flyers, promotional materials,
  images where text overlays dominate, infographics, diagrams, illustrations,
  generic stock photos with no connection to this specific company's work.

Classify purely by visual content. Ignore file names entirely.\
"""

_FILTER_SCHEMA: dict = {
    "type": "object",
    "required": ["images"],
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "include"],
                "properties": {
                    "index":   {"type": "integer"},
                    "include": {"type": "boolean"},
                },
            },
        }
    },
}


# ── Pass 2: Identity analysis schema ─────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You are a visual identity analyst studying a company's photographic archive.

Your mission: extract the SPECIFIC visual identity of this company from their real photographs.
Not general photography guidelines — the EXACT characteristics of THIS company's images.

The output will be used to generate new AI photographs that are indistinguishable from the
company's own archive. Every field must be as specific as possible. Vague answers are useless.

Study ALL photographs together. Identify what is consistent across them.
If something only appears in some photos, note that. If there is variation, describe the range.\
"""

_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "required": [
        "technician_description", "uniform_style", "truck_description",
        "neighborhood_style", "camera_angle", "color_grading",
        "imperfections", "documentary_notes", "identity_summary",
    ],
    "properties": {
        "technician_description": {
            "type": "string",
            "description": (
                "Precise visual description: approximate age, build, hair color/style, "
                "any distinguishing features (beard, glasses, etc.). "
                "Example: 'Male, mid-30s to mid-40s, average build, dark hair, often has a beard.'"
            ),
        },
        "uniform_style": {
            "type": "string",
            "description": (
                "Exact uniform: shirt type and color, logo placement (left chest? back?), "
                "pants color and style, boots/shoes, any safety gear (gloves, hard hat). "
                "Example: 'Red polo or t-shirt with white logo on left chest, dark work pants, steel-toe boots.'"
            ),
        },
        "truck_description": {
            "type": "string",
            "description": (
                "Vehicle: color, body style (pickup, cargo van, box truck), any visible text "
                "or branding on doors/sides/back. "
                "Example: 'White cargo van with red company logo on the driver door and back doors.'"
            ),
        },
        "equipment_description": {
            "type": "string",
            "description": (
                "Commonly visible tools and equipment, their condition (worn, clean, weathered), "
                "how they are stored or carried. Empty string if no equipment is visible."
            ),
        },
        "logo_description": {
            "type": "string",
            "description": (
                "Company logo appearance on visible materials: shape, colors, font style, "
                "where it appears (truck, shirt, paperwork). "
                "Example: 'Shield-shaped logo, red and white, company name in bold sans-serif above a garage door icon.'"
            ),
        },
        "neighborhood_style": {
            "type": "string",
            "description": (
                "House and neighborhood characteristics: architectural era, price range, "
                "house colors, garage door styles, street type. "
                "Example: '1990s-2000s suburban tract homes, middle income, tan/beige/gray stucco, double garage doors.'"
            ),
        },
        "driveway_style": {
            "type": "string",
            "description": (
                "Driveway surface, condition, width, notable features. "
                "Example: 'Standard concrete, clean but slightly aged, typically 2-car width.'"
            ),
        },
        "vegetation_style": {
            "type": "string",
            "description": (
                "Landscaping: grass, trees, shrubs, seasonal plants. "
                "Example: 'Maintained green lawns, occasional palm trees, desert shrubs in some.'"
            ),
        },
        "typical_weather": {
            "type": "string",
            "description": (
                "Most common weather and time of day: sunny, overcast, golden hour, harsh midday. "
                "Example: 'Mostly bright midday sunlight, occasional overcast, rarely golden hour.'"
            ),
        },
        "common_service_scenarios": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific service scenarios visible: 'technician replacing torsion spring', 'garage door panel installation', etc.",
        },
        "garage_door_styles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Garage door types visible: 'raised panel white steel', 'carriage house style', brand names if visible.",
        },
        "camera_angle": {
            "type": "string",
            "description": (
                "Typical camera position: height, angle, distance from subject. "
                "Example: 'Standing height, slightly above waist level, often from 6-10 feet away, sometimes from truck cab.'"
            ),
        },
        "focal_length_feel": {
            "type": "string",
            "description": (
                "Apparent focal length and perspective: "
                "wide-angle distortion, natural 35-50mm perspective, or slight telephoto compression. "
                "Example: '28-35mm feel, slight wide-angle, typical smartphone camera perspective.'"
            ),
        },
        "exposure_style": {
            "type": "string",
            "description": (
                "How photos are exposed: bright and airy, slightly underexposed, high contrast. "
                "Example: 'Slightly bright, auto-exposed by smartphone, occasional overexposed highlights.'"
            ),
        },
        "color_temperature": {
            "type": "string",
            "description": (
                "White balance and color temperature: warm golden, neutral daylight, cool overcast. "
                "Example: 'Neutral to slightly warm, daylight color temperature, no color correction applied.'"
            ),
        },
        "color_grading": {
            "type": "string",
            "description": (
                "Post-processing style: unedited (straight from phone), slightly saturated, "
                "Instagram-filtered, professionally color-graded. "
                "Example: 'Minimal or no post-processing, natural colors, slightly saturated in some.'"
            ),
        },
        "depth_of_field": {
            "type": "string",
            "description": (
                "Depth of field: everything sharp (small sensor, wide lens), "
                "slight background blur (portrait mode), deliberate shallow DOF. "
                "Example: 'Deep focus, everything sharp front to back, typical of smartphone wide lens.'"
            ),
        },
        "sharpness_notes": {
            "type": "string",
            "description": (
                "Image sharpness: crisp and detailed, slightly soft, motion blur present. "
                "Example: 'Generally sharp, occasional slight motion blur on moving subjects.'"
            ),
        },
        "imperfections": {
            "type": "string",
            "description": (
                "Authentic imperfections that prove these are real photos (not stock): "
                "grain, chromatic aberration, lens flare, uneven exposure, slightly off-center subjects. "
                "Example: 'Light smartphone grain in shadows, slightly crooked horizons, subjects not always centered.'"
            ),
        },
        "framing_style": {
            "type": "string",
            "description": (
                "How subjects are framed: casual snapshot framing, deliberate rule-of-thirds, "
                "always showing full garage door, close-up on hands/work. "
                "Example: 'Casual snapshot framing, full garage door usually visible, technician off-center.'"
            ),
        },
        "documentary_notes": {
            "type": "string",
            "description": (
                "Overall feel: does it look like a technician took it mid-job (most authentic), "
                "semi-professional company social media content, or staged marketing shots? "
                "Example: 'Strong documentary feel — taken mid-service-call, unposed, practical angles.'"
            ),
        },
        "forbidden_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Things that are NEVER in this company's photos and would look wrong if generated: "
                "studio lighting, luxury homes, invented branding, CGI look, etc."
            ),
        },
        "identity_summary": {
            "type": "string",
            "description": (
                "2-3 sentences summarizing the essential character of this company's photography. "
                "Should capture who they are and what their photos feel like. "
                "Example: 'A mid-size garage door company in suburban Phoenix. Their photos are taken "
                "mid-service-call on a smartphone with no editing. The technician is always in a red "
                "polo, the truck is always white, and the neighborhoods are always 1990s-2000s tract homes.'"
            ),
        },
    },
}


# ── Service ───────────────────────────────────────────────────────────────────

class VisualIdentityService:
    """
    Builds an article-specific VisualIdentityProfile from Google Drive photographs.

    Unlike VisualStyleService (global, cached), this service selects the TOP 10–20
    Drive photos most semantically relevant to THIS article's topic, analyzes them
    with Claude Vision, and returns both a deep identity profile AND the reference
    image bytes needed for multi-reference AI generation.

    Three-pass analysis:
      Pass 1 — Semantic scoring: rank Drive candidates against article keywords
      Pass 2 — Filter: Claude removes logos, screenshots, non-photographs
      Pass 3 — Identity extraction: Claude Vision extracts the full identity profile

    No caching — article-specific analysis is cheap enough to run fresh each time.
    """

    def __init__(self, drive: GoogleDriveService, claude: ClaudeService) -> None:
        self._drive = drive
        self._claude = claude

    def build_for_article(
        self,
        article: "Article",
        drive_candidates: list[DriveFileInfo],
    ) -> tuple[VisualIdentityProfile, list[bytes]]:
        """
        Build an article-specific visual identity profile.

        Returns (profile, reference_bytes) where:
          profile: VisualIdentityProfile for prompt construction
          reference_bytes: Drive photo bytes for multi-reference AI generation

        Returns (empty profile, []) if Drive candidates are unavailable
        or analysis fails — the caller must handle gracefully.
        """
        if not drive_candidates:
            logger.warning("No Drive candidates available for identity analysis.")
            return VisualIdentityProfile(), []

        article_tokens = self._article_tokens(article)

        # ── Pass 1: Semantic scoring → top N candidates ───────────────────────
        scored = sorted(
            [(c, self._score(c, article_tokens)) for c in drive_candidates],
            key=lambda x: x[1], reverse=True,
        )
        top_candidates = [c for c, _ in scored[:_MAX_CANDIDATES]]
        logger.info(
            "Identity analysis: top %d Drive candidates selected from %d total.",
            len(top_candidates), len(drive_candidates),
        )

        # ── Download thumbnails (1024px — suitable for analysis + generation) ─
        thumbnails: list[tuple[DriveFileInfo, bytes]] = []
        for file_info in top_candidates:
            try:
                if file_info.thumbnail_link:
                    try:
                        thumb = self._drive.download_thumbnail(file_info.thumbnail_link, size=1024)
                    except Exception:
                        thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=1024)
                else:
                    thumb = self._drive.download_thumbnail_by_id(file_info.file_id, size=1024)
                thumbnails.append((file_info, thumb))
            except Exception as exc:
                logger.debug("Skipping thumbnail for identity: %s — %s", file_info.name, exc)

        if not thumbnails:
            logger.warning("No thumbnails downloadable for identity analysis.")
            return VisualIdentityProfile(), []

        # ── Pass 2: Filter — Claude removes non-photographs ───────────────────
        logger.info("Identity filter pass: classifying %d thumbnails...", len(thumbnails))
        keep_flags = self._filter_pass(thumbnails)
        usable: list[tuple[DriveFileInfo, bytes]] = [
            (f, b) for (f, b), keep in zip(thumbnails, keep_flags) if keep
        ]

        if not usable:
            logger.warning("Identity: all thumbnails filtered out — using generic profile.")
            return VisualIdentityProfile(), []

        analysis_set = usable[:_MAX_ANALYSIS]
        logger.info("Identity analysis pass: analyzing %d photographs...", len(analysis_set))

        # ── Pass 3: Identity extraction — Claude Vision ───────────────────────
        try:
            raw = self._analysis_pass([b for _, b in analysis_set])
        except Exception as exc:
            logger.error("Identity analysis failed: %s — returning empty profile.", exc)
            return VisualIdentityProfile(), [b for _, b in analysis_set]

        profile = VisualIdentityProfile(
            technician_description=raw.get("technician_description", ""),
            uniform_style=raw.get("uniform_style", ""),
            truck_description=raw.get("truck_description", ""),
            equipment_description=raw.get("equipment_description", ""),
            logo_description=raw.get("logo_description", ""),
            neighborhood_style=raw.get("neighborhood_style", ""),
            driveway_style=raw.get("driveway_style", ""),
            vegetation_style=raw.get("vegetation_style", ""),
            typical_weather=raw.get("typical_weather", ""),
            common_service_scenarios=raw.get("common_service_scenarios", []),
            garage_door_styles=raw.get("garage_door_styles", []),
            camera_angle=raw.get("camera_angle", ""),
            focal_length_feel=raw.get("focal_length_feel", ""),
            exposure_style=raw.get("exposure_style", ""),
            color_temperature=raw.get("color_temperature", ""),
            color_grading=raw.get("color_grading", ""),
            depth_of_field=raw.get("depth_of_field", ""),
            sharpness_notes=raw.get("sharpness_notes", ""),
            imperfections=raw.get("imperfections", ""),
            framing_style=raw.get("framing_style", ""),
            documentary_notes=raw.get("documentary_notes", ""),
            forbidden_elements=raw.get("forbidden_elements", []),
            identity_summary=raw.get("identity_summary", ""),
            reference_file_ids=[f.file_id for f, _ in analysis_set],
            training_image_count=len(analysis_set),
        )

        reference_bytes = [b for _, b in analysis_set]
        logger.info(
            "Identity profile built from %d photographs. Summary: %s",
            len(analysis_set), profile.identity_summary[:100],
        )
        return profile, reference_bytes

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _article_tokens(self, article: "Article") -> frozenset[str]:
        """Build a token set representing the article's semantic content."""
        texts = [
            article.title,
            article.seo.focus_keyword,
            article.request.service or "",
        ]
        if article.request.location:
            texts.append(article.request.location.city or "")
            texts.append(article.request.location.state or "")
        return frozenset().union(*(_tokenize(t) for t in texts))

    def _score(self, candidate: DriveFileInfo, article_tokens: frozenset[str]) -> int:
        """Score a Drive candidate against article tokens."""
        candidate_tokens = (
            _tokenize(candidate.folder_path)
            | _tokenize(candidate.name)
            | _tokenize(candidate.description or "")
        )
        return len(article_tokens & candidate_tokens)

    # ── Pass 2: Filter ────────────────────────────────────────────────────────

    def _filter_pass(
        self, thumbnails: list[tuple[DriveFileInfo, bytes]]
    ) -> list[bool]:
        """
        Ask Claude to classify each thumbnail as a real company photograph or non-photo.

        Returns list[bool] — True = keep, False = exclude.
        Never shows filenames to Claude — classification is purely visual.
        """
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"Classify each of the following {len(thumbnails)} images. "
                "For each: is it a usable real company work photograph?"
            ),
        }]
        for i, (_, thumb_bytes) in enumerate(thumbnails, start=1):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(thumb_bytes),
                    "data": base64.standard_b64encode(thumb_bytes).decode(),
                },
            })

        result = self._claude.generate_structured(
            system=_FILTER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            tool_name="classify_photos",
            tool_description="Classify each image: real company photo or non-photo content.",
            input_schema=_FILTER_SCHEMA,
            max_tokens=1024,
        )

        include_indices: set[int] = {
            item["index"]
            for item in result.get("images", [])
            if item.get("include", False)
        }
        logger.debug(
            "Identity filter: %d/%d images kept.",
            len(include_indices), len(thumbnails),
        )
        return [
            (i + 1) in include_indices
            for i in range(len(thumbnails))
        ]

    # ── Pass 3: Identity analysis ─────────────────────────────────────────────

    def _analysis_pass(self, usable_bytes: list[bytes]) -> dict:
        """
        Claude Vision deep-analyzes the filtered photographs to extract the
        company's complete visual identity.
        """
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"Study these {len(usable_bytes)} company photographs carefully. "
                "Extract the company's specific visual identity from what you observe. "
                "Be precise — these details will be used to generate AI images that must "
                "be indistinguishable from these real photographs."
            ),
        }]
        for i, img_bytes in enumerate(usable_bytes, start=1):
            content.append({"type": "text", "text": f"Photograph {i}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(img_bytes),
                    "data": base64.standard_b64encode(img_bytes).decode(),
                },
            })

        return self._claude.generate_structured(
            system=_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": content}],
            tool_name="extract_visual_identity",
            tool_description=(
                "Extract the company's complete visual identity from their photographs."
            ),
            input_schema=_ANALYSIS_SCHEMA,
            max_tokens=3000,
            thinking=True,
        )

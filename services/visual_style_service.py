from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

from models.visual_style import VisualStyleProfile
from services.claude_service import ClaudeService
from services.google_drive_service import DriveFileInfo, GoogleDriveService

logger = logging.getLogger(__name__)


def _detect_media_type(data: bytes) -> str:
    """Return the Claude-accepted media_type string from image magic bytes."""
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

For each image shown, decide whether it is a usable reference photograph or
should be excluded from visual style analysis.

INCLUDE ("include": true):
  Real photographs of work being performed, finished results, job sites,
  equipment in use, team members working, before/after results, or any
  genuine documentary photo of the business's services and environments.

EXCLUDE ("include": false):
  - Logos, icons, or brand marks
  - Screenshots of websites, software, or devices
  - Banners, flyers, or promotional materials
  - Images where text overlays dominate
  - Infographics, diagrams, or illustrations
  - Generic stock photos with no clear connection to the business
  - Advertising graphics or collages

Apply these rules purely based on visual content. Ignore file names entirely.
"""

_FILTER_SCHEMA: dict = {
    "type": "object",
    "required": ["images"],
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "include", "category"],
                "properties": {
                    "index":    {"type": "integer", "description": "1-based image index."},
                    "include":  {"type": "boolean"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "photograph", "logo", "icon", "screenshot",
                            "banner", "text_heavy", "advertisement", "graphic",
                        ],
                    },
                },
            },
        }
    },
}


# ── Pass 2: Analysis schema ───────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You are a professional visual brand analyst and photography director.

Analyze the photographs shown and extract the brand's visual identity.
These are real photos from a business's work — focus entirely on what you see.

Extract:
- Photography style (documentary, action, portrait, environmental, candid, etc.)
- Lighting (natural daylight, golden hour, indoor, artificial, mixed)
- Composition patterns (wide-angle, tight close-ups, eye level, overhead, etc.)
- Color palette and dominant tones
- Typical scenarios and environments (job site, residential, commercial, etc.)
- Overall brand aesthetic (premium, approachable, technical, hands-on, etc.)

Then write prompt_guidelines: specific, actionable instructions for DALL-E 3
in imperative form that will make AI-generated images visually consistent with
these real photographs. Focus on:
- Lighting style to replicate
- Composition approach
- Color mood to match
- What should be in the scene (environment, context, people if any)
- What to avoid (over-processed looks, studio aesthetics if not present, etc.)

The goal is that a generated image feels like it was taken by the same
photographer on the same job, not that it looks "AI-generated."
"""

_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "required": [
        "photography_style", "lighting", "composition",
        "typical_scenarios", "color_palette",
        "prompt_guidelines", "style_description",
    ],
    "properties": {
        "photography_style": {
            "type": "string",
            "description": "Overall photographic style in one sentence.",
        },
        "lighting": {
            "type": "string",
            "description": "Dominant lighting conditions.",
        },
        "composition": {
            "type": "string",
            "description": "Typical composition patterns across the library.",
        },
        "typical_scenarios": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recurring visual scenarios (3–8 items).",
        },
        "color_palette": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Dominant colors as descriptive names or hex values (3–6 items).",
        },
        "prompt_guidelines": {
            "type": "string",
            "description": (
                "Direct DALL-E 3 instructions in imperative form. Example: "
                "'Outdoor jobsite photography, natural daylight, workers in safety gear. "
                "Slightly underexposed shadows, warm midday light. "
                "Documentary-style, 35mm equivalent. No text overlays. "
                "Environment: residential driveways and garage interiors.'"
            ),
        },
        "style_description": {
            "type": "string",
            "description": "Narrative description of the brand's visual identity (2–4 sentences).",
        },
    },
}


# ── Service ───────────────────────────────────────────────────────────────────

class VisualStyleService:
    """
    Builds and caches the visual brand profile for a business.

    Analysis is two-pass:
      Pass 1 — Filter: Claude classifies each thumbnail as a usable photograph
                or non-photo content (logos, screenshots, banners, etc.).
                Filenames and metadata are never shown to Claude.
      Pass 2 — Analyze: Claude Vision analyzes only the filtered photographs
                and extracts the brand's visual identity.

    The resulting VisualStyleProfile drives both Drive image selection and
    DALL-E 3 prompt construction, ensuring AI-generated images are visually
    consistent with the business's real photography.

    Cache: profiles/visual_style.json (single global file).
    Invalidated after 30 days or when Drive image count changes by >5 %.
    """

    _MAX_CACHE_DAYS = 30
    _MAX_DRIFT_RATIO = 0.05

    def __init__(
        self,
        profiles_dir: Path,
        drive: GoogleDriveService,
        claude: ClaudeService,
        analysis_limit: int = 30,
    ) -> None:
        self._dir = profiles_dir
        self._drive = drive
        self._claude = claude
        self._limit = analysis_limit

    # ── Public interface ──────────────────────────────────────────────────────

    def get_profile(self, folder_id: str) -> VisualStyleProfile:
        """
        Return the visual style profile, refreshing the cache when stale.

        Checks the disk cache first. Re-analyzes when the cache is missing,
        expired, or the folder contents have changed significantly.
        """
        cached = self._load()
        if cached is not None and self._is_valid(cached, folder_id):
            logger.info("Visual style cache hit (age: %dd)", (datetime.now(timezone.utc) - cached.analyzed_at).days)
            return cached

        return self._analyze(folder_id)

    # ── Cache management ──────────────────────────────────────────────────────

    def _load(self) -> VisualStyleProfile | None:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            return VisualStyleProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Visual style cache corrupt (%s): %s", path, exc)
            return None

    def _is_valid(self, cached: VisualStyleProfile, folder_id: str) -> bool:
        if cached.drive_folder_id != folder_id:
            logger.debug("Visual style cache invalid: folder_id changed.")
            return False
        age_days = (datetime.now(timezone.utc) - cached.analyzed_at).days
        if age_days > self._MAX_CACHE_DAYS:
            logger.debug("Visual style cache expired (%d days old).", age_days)
            return False
        try:
            stats = self._drive.get_folder_stats(folder_id)
            if cached.image_count == 0:
                return stats.image_count == 0
            drift = abs(stats.image_count - cached.image_count) / cached.image_count
            if drift > self._MAX_DRIFT_RATIO:
                logger.debug("Visual style cache stale: image count drifted %.0f%%.", drift * 100)
                return False
        except Exception:
            return True  # can't reach Drive — trust the cache
        return True

    def _save(self, profile: VisualStyleProfile) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Visual style profile saved: %s", path)

    def _cache_path(self) -> Path:
        return self._dir / "visual_style.json"

    # ── Two-pass analysis ─────────────────────────────────────────────────────

    def _analyze(self, folder_id: str) -> VisualStyleProfile:
        logger.info("Starting visual style analysis (limit: %d images)...", self._limit)

        stats = self._drive.get_folder_stats(folder_id)
        all_files = self._drive.list_images(folder_id, recursive=True, limit=self._limit)

        if not all_files:
            logger.warning("No images found in Drive folder %s — using generic profile.", folder_id)
            return self._generic_profile(folder_id, stats.image_count)

        # Download thumbnails (ignore filenames — purely visual)
        thumbnails: list[tuple[DriveFileInfo, bytes]] = []
        for file_info in all_files:
            if not file_info.thumbnail_link:
                continue
            try:
                thumb = self._drive.download_thumbnail(file_info.thumbnail_link, size=512)
                thumbnails.append((file_info, thumb))
            except Exception as exc:
                logger.debug("Skipping thumbnail (download failed): %s", exc)

        if not thumbnails:
            logger.warning("No thumbnails downloadable — using generic profile.")
            return self._generic_profile(folder_id, stats.image_count)

        logger.info("Pass 1: filtering %d thumbnails...", len(thumbnails))
        usable = self._filter_pass(thumbnails)

        if not usable:
            logger.warning("All images filtered out — using generic profile.")
            return self._generic_profile(folder_id, stats.image_count)

        logger.info("Pass 2: analyzing %d usable photographs...", len(usable))
        data = self._analysis_pass(usable)

        profile = VisualStyleProfile(
            drive_folder_id=folder_id,
            analyzed_at=datetime.now(timezone.utc),
            image_count=stats.image_count,
            analyzed_count=len(thumbnails),
            filtered_count=len(usable),
            photography_style=data["photography_style"],
            lighting=data.get("lighting", ""),
            composition=data.get("composition", ""),
            typical_scenarios=data.get("typical_scenarios", []),
            color_palette=data.get("color_palette", []),
            prompt_guidelines=data["prompt_guidelines"],
            style_description=data["style_description"],
        )
        self._save(profile)
        logger.info(
            "Visual style analysis complete: %d/%d images usable after filtering.",
            len(usable), len(thumbnails),
        )
        return profile

    def _filter_pass(
        self, thumbnails: list[tuple[DriveFileInfo, bytes]]
    ) -> list[bytes]:
        """
        Pass 1: Ask Claude to classify each thumbnail as photograph or skip.

        Sends all thumbnails in a single call. Returns only the bytes of
        images classified as usable photographs, in original order.
        Filenames are never shown — classification is purely visual.
        """
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"Classify each of the following {len(thumbnails)} images "
                "as a usable business photograph or non-photo content to exclude."
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
            tool_name="classify_images",
            tool_description="Classify each image as usable photograph or non-photo content.",
            input_schema=_FILTER_SCHEMA,
            max_tokens=2000,
        )

        include_indices: set[int] = {
            item["index"]
            for item in result.get("images", [])
            if item.get("include", False)
        }
        logger.debug(
            "Filter pass: %d/%d images kept (indices: %s)",
            len(include_indices), len(thumbnails), sorted(include_indices),
        )
        return [
            thumb_bytes
            for i, (_, thumb_bytes) in enumerate(thumbnails, start=1)
            if i in include_indices
        ]

    def _analysis_pass(self, usable_thumbs: list[bytes]) -> dict:
        """
        Pass 2: Analyze the filtered photographs to extract visual brand identity.

        No filenames or metadata — purely visual content.
        """
        content: list[dict] = [{
            "type": "text",
            "text": f"Analyze these {len(usable_thumbs)} business photographs and extract the brand's visual identity:",
        }]
        for thumb_bytes in usable_thumbs:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(thumb_bytes),
                    "data": base64.standard_b64encode(thumb_bytes).decode(),
                },
            })

        return self._claude.generate_structured(
            system=_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": content}],
            tool_name="save_visual_style",
            tool_description="Save the visual brand style profile extracted from the photographs.",
            input_schema=_ANALYSIS_SCHEMA,
            max_tokens=2000,
            thinking=True,
        )

    def _generic_profile(self, folder_id: str, image_count: int) -> VisualStyleProfile:
        return VisualStyleProfile(
            drive_folder_id=folder_id,
            analyzed_at=datetime.now(timezone.utc),
            image_count=image_count,
            analyzed_count=0,
            filtered_count=0,
            photography_style="professional corporate photography",
            lighting="natural daylight",
            composition="wide environmental shots and close-up detail",
            typical_scenarios=[],
            color_palette=[],
            prompt_guidelines=(
                "Professional photograph, natural lighting, clean corporate environment, "
                "high production quality. No text overlays, no logos. "
                "35mm equivalent, editorial quality."
            ),
            style_description="No images available for analysis. Generic professional guidelines applied.",
        )

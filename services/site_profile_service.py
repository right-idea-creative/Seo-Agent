"""
SiteProfileService — load and save per-website SiteProfile JSON files.

Profile path: {profiles_dir}/{client_id}/{website_id}/site.json

The generate command calls load() after resolving client_id + website_id and
uses the returned profile to auto-populate ArticleRequest fields that would
otherwise require manual CLI flags on every invocation.

Failure modes are non-fatal: a missing or corrupt profile returns None and
the CLI falls through to the validation layer, which produces an actionable
error listing exactly which fields are missing.
"""
from __future__ import annotations

import logging
from pathlib import Path

from models.site_profile import SiteProfile

logger = logging.getLogger(__name__)


class SiteProfileService:
    """Loads and persists SiteProfile JSON files from the profiles directory."""

    def __init__(self, profiles_dir: Path) -> None:
        self._profiles_dir = profiles_dir

    def profile_path(self, client_id: str, website_id: str) -> Path:
        return self._profiles_dir / client_id / website_id / "site.json"

    def load(self, client_id: str, website_id: str) -> SiteProfile | None:
        """
        Return the SiteProfile for this website, or None if not found.

        Logs a debug message on a missing file and a warning on a parse error.
        Never raises — missing profiles are handled by the caller.
        """
        path = self.profile_path(client_id, website_id)
        if not path.exists():
            logger.debug("No site profile at %s — manual CLI flags required.", path)
            return None
        try:
            return SiteProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Could not load site profile from %s: %s — manual CLI flags required.",
                path, exc,
            )
            return None

    def save(self, profile: SiteProfile) -> None:
        """Write (or overwrite) the site profile JSON file."""
        path = self.profile_path(profile.client_id, profile.website_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Site profile saved: %s", path)

    def exists(self, client_id: str, website_id: str) -> bool:
        return self.profile_path(client_id, website_id).exists()

"""
SEOCacheService — persistent, file-based cache for generated SEO metadata.

Purpose:
    When a draft article is reused, the pipeline must regenerate website-specific
    SEO metadata (title, meta description, slug, focus keyword, category) via an
    LLM call.  This cache eliminates that call on subsequent reuse of the same
    topic on the same website, reducing per-article API cost to zero when both
    the article body AND the SEO are already cached.

Storage:
    output/{client_id}/{website_id}/.seo_cache.json

    One JSON file per website.  The file is human-readable and can be inspected
    or pruned manually.

Cache key:
    topic_id                          — when no focus keyword is specified
    {topic_id}|{normalized_keyword}   — when a focus keyword is present

    The "|" separator is safe because topic_id tokens only contain [a-z0-9-].
    Normalized keyword: lower-case, non-alphanumeric characters collapsed to "-".

Thread safety:
    Reads are atomic (single json.loads call).  Writes use an atomic
    write-then-rename pattern via a temp sibling file so a crash mid-write
    never corrupts the cache.

No API calls are made in this module.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

logger = logging.getLogger(__name__)

_CACHE_FILENAME = ".seo_cache.json"
_CURRENT_VERSION = "1.0"


def _normalize_keyword(focus_keyword: str) -> str:
    """Lower-case + collapse non-alphanumeric runs to a single hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", focus_keyword.lower().strip()).strip("-")


class SEOCacheService:
    """
    Persistent per-website LRU-unlimited cache for SEO metadata.

    Usage::

        cache = SEOCacheService(output_dir, "client-a", "site-1")

        seo = cache.get("door-garage-repair-spring", "garage door spring repair")
        if seo is None:
            seo = agent._generate_seo(request, markdown)
            cache.put("door-garage-repair-spring", seo, "garage door spring repair")
    """

    def __init__(self, output_dir: Path, client_id: str, website_id: str) -> None:
        self._path: Path = output_dir / client_id / website_id / _CACHE_FILENAME

    # ── Public API ─────────────────────────────────────────────────────────

    def get(
        self,
        topic_id: str,
        focus_keyword: str | None = None,
    ):
        """
        Return cached SEOMetadata, or None on miss.

        Never raises — a corrupt or missing cache file is treated as a cold miss.
        """
        from models.article import SEOMetadata

        key = self._make_key(topic_id, focus_keyword)
        try:
            data = self._load()
            entry = data.get("entries", {}).get(key)
            if entry is None:
                return None
            seo = SEOMetadata(**entry["seo"])
            logger.info("SEO cache hit: %s", key)
            return seo
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
        except ValidationError as exc:
            logger.warning("SEO cache entry %s is malformed, evicting: %s", key, exc)
            self._evict(key)
            return None

    def put(
        self,
        topic_id: str,
        seo,
        focus_keyword: str | None = None,
    ) -> None:
        """
        Write SEOMetadata to the cache.

        Never raises — a write failure is logged but does not abort the pipeline.
        """
        key = self._make_key(topic_id, focus_keyword)
        try:
            try:
                data = self._load()
            except (FileNotFoundError, json.JSONDecodeError):
                data = {"version": _CURRENT_VERSION, "entries": {}}

            data.setdefault("version", _CURRENT_VERSION)
            data.setdefault("entries", {})[key] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "topic_id": topic_id,
                "focus_keyword": focus_keyword,
                "seo": seo.model_dump(),
            }

            self._atomic_write(data)
            logger.info("SEO cache stored: %s", key)
        except Exception as exc:
            logger.warning("SEO cache write failed (non-blocking): %s", exc)

    def stats(self) -> dict:
        """Return a dict with entry count and file size for display."""
        try:
            data = self._load()
            size = self._path.stat().st_size
            return {
                "entries": len(data.get("entries", {})),
                "size_bytes": size,
                "path": str(self._path),
            }
        except Exception:
            return {"entries": 0, "size_bytes": 0, "path": str(self._path)}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_key(self, topic_id: str, focus_keyword: str | None) -> str:
        if focus_keyword:
            return f"{topic_id}|{_normalize_keyword(focus_keyword)}"
        return topic_id

    def _load(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _atomic_write(self, data: dict) -> None:
        """Write JSON atomically: write to temp file, then rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".seo_cache_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _evict(self, key: str) -> None:
        """Remove a single malformed entry from the cache."""
        try:
            data = self._load()
            data.get("entries", {}).pop(key, None)
            self._atomic_write(data)
        except Exception:
            pass

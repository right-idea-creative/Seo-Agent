from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ArticleUsage:
    slug: str
    date: str
    purpose: str
    post_id: int | None = None


@dataclass
class ImageUsageRecord:
    drive_file_id: str
    filename: str
    times_used: int
    featured_count: int
    inline_count: int
    last_used_date: str
    last_article_slug: str
    last_post_id: int | None
    article_history: list[ArticleUsage] = field(default_factory=list)


class EditorialHistoryService:
    """Persistent store tracking every published Drive image for editorial diversity scoring."""

    _VERSION = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict = {"version": self._VERSION, "recent_articles": [], "images": {}}
        self._dirty: bool = False
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def get_record(self, file_id: str) -> ImageUsageRecord | None:
        raw = self._data["images"].get(file_id)
        if raw is None:
            return None
        history = [
            ArticleUsage(
                slug=e["slug"],
                date=e["date"],
                purpose=e["purpose"],
                post_id=e.get("post_id"),
            )
            for e in raw.get("article_history", [])
        ]
        return ImageUsageRecord(
            drive_file_id=raw["drive_file_id"],
            filename=raw.get("filename", ""),
            times_used=raw.get("times_used", 0),
            featured_count=raw.get("featured_count", 0),
            inline_count=raw.get("inline_count", 0),
            last_used_date=raw.get("last_used_date", ""),
            last_article_slug=raw.get("last_article_slug", ""),
            last_post_id=raw.get("last_post_id"),
            article_history=history,
        )

    def get_recent_articles(self, n: int = 20) -> list[str]:
        return self._data["recent_articles"][:n]

    def get_recent_featured_file_ids(self, n: int = 5) -> list[str]:
        # Search a wider window than n to find enough featured entries.
        recent_set = set(self._data["recent_articles"][: n * 4])
        featured: list[tuple[str, str]] = []
        for file_id, raw in self._data["images"].items():
            for entry in raw.get("article_history", []):
                if entry.get("purpose") == "featured" and entry.get("slug") in recent_set:
                    featured.append((entry["date"], file_id))
        featured.sort(key=lambda x: x[0], reverse=True)
        seen: list[str] = []
        seen_ids: set[str] = set()
        for _, fid in featured:
            if fid not in seen_ids:
                seen.append(fid)
                seen_ids.add(fid)
            if len(seen) >= n:
                break
        return seen

    def record_publication(
        self,
        *,
        file_id: str,
        filename: str,
        slug: str,
        post_id: int | None,
        purpose: str,
        date: str | None = None,
    ) -> None:
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        images = self._data["images"]
        if file_id not in images:
            images[file_id] = {
                "drive_file_id": file_id,
                "filename": filename,
                "times_used": 0,
                "featured_count": 0,
                "inline_count": 0,
                "last_used_date": date,
                "last_article_slug": slug,
                "last_post_id": post_id,
                "article_history": [],
            }
        rec = images[file_id]
        rec["times_used"] = rec.get("times_used", 0) + 1
        if purpose == "featured":
            rec["featured_count"] = rec.get("featured_count", 0) + 1
        else:
            rec["inline_count"] = rec.get("inline_count", 0) + 1
        rec["last_used_date"] = date
        rec["last_article_slug"] = slug
        rec["last_post_id"] = post_id
        rec["article_history"].append({
            "slug": slug,
            "date": date,
            "purpose": purpose,
            "post_id": post_id,
        })
        if filename:
            rec["filename"] = filename
        self._dirty = True

    def finalize_article(self, slug: str) -> None:
        recent = self._data["recent_articles"]
        if slug in recent:
            recent.remove(slug)
        recent.insert(0, slug)
        self._data["recent_articles"] = recent[:50]
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.rename(self._path)
            self._dirty = False
        except Exception as exc:
            logger.warning("EditorialHistoryService: could not save to %s: %s", self._path, exc)

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = {
                    "version": raw.get("version", self._VERSION),
                    "recent_articles": raw.get("recent_articles", []),
                    "images": raw.get("images", {}),
                }
        except Exception as exc:
            logger.warning("EditorialHistoryService: could not load %s: %s", self._path, exc)

    @staticmethod
    def _serialize_record(rec: ImageUsageRecord) -> dict:
        return {
            "drive_file_id": rec.drive_file_id,
            "filename": rec.filename,
            "times_used": rec.times_used,
            "featured_count": rec.featured_count,
            "inline_count": rec.inline_count,
            "last_used_date": rec.last_used_date,
            "last_article_slug": rec.last_article_slug,
            "last_post_id": rec.last_post_id,
            "article_history": [
                {
                    "slug": e.slug,
                    "date": e.date,
                    "purpose": e.purpose,
                    "post_id": e.post_id,
                }
                for e in rec.article_history
            ],
        }

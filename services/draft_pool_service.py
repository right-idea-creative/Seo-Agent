"""
DraftPoolService — global lightweight index of all generated articles.

Purpose
-------
Replace per-run filesystem glob scans with a single in-memory lookup over a
persistent JSON index.  The pool stores *only* the metadata needed for
topic-matching and safety-gate checks; the full Article object is loaded from
disk only when a match is confirmed and the content is actually needed.

Storage
-------
    output/.draft_pool.json

Layout::

    {
      "version": "1.0",
      "entries": [
        {
          "topic_id": "door-garage-repair-spring",
          "article_path": "client-a/site-1/garage-door-spring-repair/article.json",
          "client_id":    "client-a",
          "website_id":   "site-1",
          "reuse_group":  "garage-door-network",
          "city":         "Denver",
          "state":        "CO",
          "country":      "US",
          "focus_keyword": "garage door spring repair",
          "title":         "Garage Door Spring Repair in Denver",
          "category":      "Repair",
          "word_count":    863,
          "created_at":    "2026-07-14T12:00:00+00:00"
        },
        ...
      ]
    }

Lookup algorithm
----------------
1. Load pool index from disk (O(1) file read + JSON parse).
2. Build an in-memory dict keyed by topic_id for O(1) exact-match lookup.
3. Priority order for matches: same website → same client → same reuse_group.
4. Location gate: same city (or both locationless) before scoring.
5. Client safety gate: same client_id always allowed; cross-client only when
   both entries share the same non-empty reuse_group.
6. On topic_id miss, fall through to Jaccard similarity scan over pool entries
   (no filesystem I/O — everything is already in memory).

Fallback
--------
If the pool file is missing or corrupt, the caller should fall back to the
existing DraftReuseService filesystem scan.  This service does NOT raise —
build_or_load() returns an empty list on failure.

No API calls are made in this module.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.article import Article, ArticleRequest
    from models.tenant import TenantContext

logger = logging.getLogger(__name__)

_POOL_FILENAME = ".draft_pool.json"
_CURRENT_VERSION = "1.0"

# Statuses that make an article ineligible for the draft pool.
# Published and archived articles are final — they must never be offered as
# reuse candidates because the pool is a library of *internal drafts* only.
_EXCLUDED_STATUSES: frozenset[str] = frozenset({"published", "archived"})

# Reuse the same stop-words / Jaccard helpers from draft_reuse_service to keep
# the similarity scoring consistent.
import re as _re

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
    "from", "has", "he", "how", "in", "is", "it", "its", "of", "on",
    "or", "our", "the", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
    "guide", "tips", "best", "top", "get", "vs", "versus", "about",
    "also", "into", "that", "than", "then", "they", "them", "their",
})

SIMILARITY_THRESHOLD = 0.72


def _tokenize(text: str) -> frozenset[str]:
    words = _re.findall(r"[a-z]+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class PoolEntry:
    """Lightweight metadata for one generated article."""
    topic_id: str
    article_path: str          # relative to output_dir
    client_id: str
    website_id: str
    reuse_group: str | None
    city: str | None
    state: str | None
    country: str | None
    focus_keyword: str | None
    title: str
    category: str | None
    word_count: int
    created_at: str
    status: str = "review"     # ArticleStatus value; published/archived excluded from pool


@dataclass
class PoolMatch:
    """Result returned by DraftPoolService.find_match()."""
    entry: PoolEntry
    similarity: float
    matched_by_topic_id: bool
    same_website: bool


def _location_compatible(req_city: str | None, art_city: str | None) -> bool:
    if req_city is None and art_city is None:
        return True
    if req_city is None or art_city is None:
        return False
    return req_city.lower().strip() == art_city.lower().strip()


def _client_allowed(
    req_client: str,
    art_client: str,
    req_group: str | None,
    art_group: str | None,
) -> bool:
    if req_client == art_client:
        return True
    if req_group and art_group and req_group == art_group:
        return True
    return False


class DraftPoolService:
    """
    In-memory draft index backed by a persistent JSON file.

    Typical use::

        pool = DraftPoolService(output_dir)
        pool.build_or_load()                    # fast on warm runs

        match = pool.find_match(request, tenant, req_reuse_group)
        if match:
            article = pool.load_article(match, output_dir)

        # After a new article is saved:
        pool.add_entry(article, article_path)
        pool.save()
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._pool_path = output_dir / _POOL_FILENAME
        self._entries: list[PoolEntry] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def build_or_load(self) -> None:
        """
        Load pool from disk if it exists; otherwise scan the filesystem to build it.
        Silently succeeds with an empty pool on any error.
        """
        if self._pool_path.exists():
            try:
                self._load()
                logger.debug("Draft pool loaded: %d entries", len(self._entries))
                return
            except Exception as exc:
                logger.warning("Draft pool file is corrupt, rebuilding: %s", exc)

        try:
            self._build_from_filesystem()
            self.save()
            logger.info("Draft pool built from filesystem: %d entries", len(self._entries))
        except Exception as exc:
            logger.warning("Draft pool build failed (non-blocking): %s", exc)
            self._entries = []

    def find_match(
        self,
        request: "ArticleRequest",
        tenant: "TenantContext",
        req_reuse_group: str | None = None,
    ) -> PoolMatch | None:
        """
        Return the best PoolMatch or None.

        Priority order for equally-scored candidates:
          same_website > same_client > same_reuse_group

        Pass 1: exact topic_id match (O(1) dict lookup).
        Pass 2: Jaccard similarity over all pool entries (O(n), no filesystem I/O).
        """
        from services.topic_normalization import normalize_topic_id

        req_topic_id = normalize_topic_id(request.topic, request.location)
        req_city = request.location.city if request.location else None

        req_tokens = _tokenize(request.topic)
        if request.focus_keyword:
            req_tokens = req_tokens | _tokenize(request.focus_keyword)
        if request.service:
            req_tokens = req_tokens | _tokenize(request.service)

        # Separate buckets so we can apply priority ordering
        topic_id_matches: list[PoolMatch] = []
        jaccard_matches: list[PoolMatch] = []

        for entry in self._entries:
            # ── Eligibility: only unpublished drafts may be reused ────────
            if entry.status in _EXCLUDED_STATUSES:
                continue
            # ── Safety gates ──────────────────────────────────────────────
            if not _location_compatible(req_city, entry.city):
                continue
            if not _client_allowed(tenant.client_id, entry.client_id, req_reuse_group, entry.reuse_group):
                continue

            same_website = (entry.client_id == tenant.client_id and entry.website_id == tenant.website_id)

            # ── Pass 1: topic_id exact match ──────────────────────────────
            if req_topic_id and entry.topic_id and req_topic_id == entry.topic_id:
                topic_id_matches.append(PoolMatch(
                    entry=entry,
                    similarity=1.0,
                    matched_by_topic_id=True,
                    same_website=same_website,
                ))
                continue

            # ── Pass 2: Jaccard similarity ────────────────────────────────
            candidate_text = " ".join(filter(None, [
                entry.title,
                entry.focus_keyword,
                (entry.focus_keyword or "").replace("-", " "),
                entry.topic_id.replace("-", " ") if entry.topic_id else "",
            ]))
            score = _jaccard(req_tokens, _tokenize(candidate_text))
            if score >= SIMILARITY_THRESHOLD:
                jaccard_matches.append(PoolMatch(
                    entry=entry,
                    similarity=score,
                    matched_by_topic_id=False,
                    same_website=same_website,
                ))

        # ── Priority selection ────────────────────────────────────────────
        # topic_id matches take absolute precedence; within each group prefer
        # same_website, then same_client, then any reuse_group match.
        def _priority(m: PoolMatch) -> tuple:
            return (
                1 if m.same_website else (
                    2 if m.entry.client_id == tenant.client_id else 3
                ),
                -m.similarity,
            )

        candidates = topic_id_matches or jaccard_matches
        if not candidates:
            return None

        best = min(candidates, key=_priority)

        logger.info(
            "Draft pool match (%s, %.0f%% sim, %s): '%s'",
            "topic_id" if best.matched_by_topic_id else "jaccard",
            best.similarity * 100,
            "same-site" if best.same_website else "cross-site",
            best.entry.title[:60],
        )
        return best

    def load_article(self, match: PoolMatch, output_dir: Path) -> "Article | None":
        """
        Load the full Article object from disk using the path stored in the pool entry.
        Returns None if the file has been deleted or is unreadable.
        """
        from models.article import Article

        path = output_dir / match.entry.article_path
        try:
            return Article.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Pool entry points to unreadable file %s: %s", path, exc)
            return None

    def add_entry(self, article: "Article", article_path: Path) -> None:
        """
        Add a newly saved article to the pool.  Call this after every successful
        generation or reuse so the pool stays in sync with the filesystem.

        Published and archived articles are silently rejected — the pool is a
        library of internal drafts only.
        """
        art_status = article.status.value if hasattr(article.status, "value") else str(article.status)
        if art_status in _EXCLUDED_STATUSES:
            logger.debug("Draft pool: skipping %s article %s", art_status, article_path)
            return

        from services.topic_normalization import normalize_topic_id

        topic_id = article.topic_id or normalize_topic_id(
            article.request.topic, article.request.location if article.request else None
        )
        city = state = country = None
        if article.request and article.request.location:
            city = article.request.location.city
            state = article.request.location.state
            country = article.request.location.country

        reuse_group = getattr(article.tenant, "reuse_group", None)
        category = article.seo.suggested_category if article.seo else None
        created_at = article.created_at.isoformat() if article.created_at else ""

        # Use relative path so the pool is portable if output_dir moves
        try:
            rel_path = article_path.relative_to(self._output_dir)
        except ValueError:
            rel_path = article_path

        # Avoid duplicates: update existing entry if same path exists
        str_rel = str(rel_path)
        new_entry = PoolEntry(
            topic_id=topic_id,
            article_path=str_rel,
            client_id=article.tenant.client_id,
            website_id=article.tenant.website_id,
            reuse_group=reuse_group,
            city=city,
            state=state,
            country=country,
            focus_keyword=article.request.focus_keyword if article.request else None,
            title=article.title or "",
            category=category,
            word_count=article.word_count,
            created_at=created_at,
            status=art_status,
        )
        for i, e in enumerate(self._entries):
            if e.article_path == str_rel:
                self._entries[i] = new_entry
                return

        self._entries.append(new_entry)

    def save(self) -> None:
        """Atomically write the pool to disk."""
        data = {
            "version": _CURRENT_VERSION,
            "entries": [asdict(e) for e in self._entries],
        }
        self._atomic_write(data)

    def entry_count(self) -> int:
        return len(self._entries)

    # ── Internal ───────────────────────────────────────────────────────────

    def remove_entry(self, article_path: Path) -> bool:
        """
        Remove a pool entry by its article path.

        Call this immediately after an article is published so it can no longer
        be offered as a reuse candidate.  Returns True if an entry was removed.
        """
        try:
            rel = str(article_path.relative_to(self._output_dir))
        except ValueError:
            rel = str(article_path)

        before = len(self._entries)
        self._entries = [e for e in self._entries if e.article_path != rel]
        removed = len(self._entries) < before
        if removed:
            logger.info("Draft pool: evicted published article %s", rel)
        return removed

    def _load(self) -> None:
        raw = json.loads(self._pool_path.read_text(encoding="utf-8"))
        # Add backward-compatible default for the `status` field introduced later.
        self._entries = [
            PoolEntry(**{"status": "review", **e})
            for e in raw.get("entries", [])
        ]

    def _build_from_filesystem(self) -> None:
        """Scan all article.json files under output_dir and populate the pool."""
        from models.article import Article
        from services.topic_normalization import normalize_topic_id

        entries: list[PoolEntry] = []
        for path in self._output_dir.glob("**/article.json"):
            # Skip pool-adjacent temp files and checkpoints
            if ".checkpoints" in str(path) or path.name != "article.json":
                continue
            try:
                article = Article.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Skip published and archived articles — pool is for internal drafts only
            art_status = article.status.value if hasattr(article.status, "value") else str(article.status)
            if art_status in _EXCLUDED_STATUSES:
                continue

            topic_id = article.topic_id or normalize_topic_id(
                article.request.topic if article.request else "",
                article.request.location if article.request else None,
            )
            city = state = country = None
            if article.request and article.request.location:
                city = article.request.location.city
                state = article.request.location.state
                country = article.request.location.country

            try:
                rel_path = path.relative_to(self._output_dir)
            except ValueError:
                rel_path = path

            entries.append(PoolEntry(
                topic_id=topic_id,
                article_path=str(rel_path),
                client_id=article.tenant.client_id,
                website_id=article.tenant.website_id,
                reuse_group=getattr(article.tenant, "reuse_group", None),
                city=city,
                state=state,
                country=country,
                focus_keyword=article.request.focus_keyword if article.request else None,
                title=article.title or "",
                category=article.seo.suggested_category if article.seo else None,
                word_count=article.word_count,
                created_at=article.created_at.isoformat() if article.created_at else "",
                status=art_status,
            ))

        self._entries = entries

    def _atomic_write(self, data: dict) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._output_dir,
            prefix=".draft_pool_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._pool_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

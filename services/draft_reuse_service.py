"""
DraftReuseService — find and reuse existing draft articles to avoid redundant API calls.

Reuse strategy (in priority order):
  1. topic_id match — deterministic, zero-cost lookup when both the request-derived
     topic_id and the stored article.topic_id agree exactly.
  2. Jaccard keyword similarity — fallback when topic_id is absent on either side.
     Tokens drawn from topic + title + focus keyword + slug + service.

Safety gates (must ALL pass before a candidate is accepted):
  - Location gate: request city == article city, or both have no city.
  - Client gate: same client_id always allowed; different client_id allowed only
    when both sites share the same non-empty reuse_group.

No API calls are made at any point in this module.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.article import Article, ArticleRequest
    from models.tenant import TenantContext

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.72

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
    "from", "has", "he", "how", "in", "is", "it", "its", "of", "on",
    "or", "our", "the", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
    "guide", "tips", "best", "top", "get", "vs", "versus", "about",
    "also", "into", "that", "than", "then", "they", "them", "their",
})


def _tokenize(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _location_compatible(req_city: str | None, art_city: str | None) -> bool:
    if req_city is None and art_city is None:
        return True
    if req_city is None or art_city is None:
        return False
    return req_city.lower().strip() == art_city.lower().strip()


def _client_allowed(
    req_client_id: str,
    art_client_id: str,
    req_reuse_group: str | None,
    art_reuse_group: str | None,
) -> bool:
    """
    Return True when the requesting website may reuse content from the source website.

    Rules:
      - Same client_id: always allowed (same business, different sub-sites).
      - Different client_id: allowed only when both sites belong to the same
        non-empty reuse_group. A missing or empty reuse_group means no sharing.
    """
    if req_client_id == art_client_id:
        return True
    if req_reuse_group and art_reuse_group and req_reuse_group == art_reuse_group:
        return True
    return False


@dataclass
class DraftMatch:
    article: "Article"
    source_path: Path
    similarity: float
    same_website: bool
    matched_by_topic_id: bool = False


class DraftReuseService:
    """Scans the local article repository for topic-similar drafts without API calls."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def find_match(
        self,
        request: "ArticleRequest",
        tenant: "TenantContext",
        req_reuse_group: str | None = None,
    ) -> DraftMatch | None:
        """
        Return the best-matching existing article, or None.

        Pass 1 — topic_id: if the request has a topic_id and a candidate also has
        the same topic_id, the article is accepted immediately (score=1.0) once
        the location and client safety gates pass.

        Pass 2 — Jaccard similarity: used when topic_id is unavailable on either
        side. Tokens are drawn from topic + title + focus keyword + slug + service.
        Only candidates scoring above SIMILARITY_THRESHOLD (0.72) are considered.

        ``req_reuse_group`` should be the caller's site profile's ``reuse_group``
        field so that cross-client sharing can be evaluated.
        """
        from models.article import Article
        from services.topic_normalization import normalize_topic_id

        # Always compute req_topic_id from the request topic (ArticleRequest
        # has no topic_id field; topic_id lives on Article after generation).
        req_topic_id: str = normalize_topic_id(request.topic, request.location)

        req_tokens = _tokenize(request.topic)
        if request.focus_keyword:
            req_tokens = req_tokens | _tokenize(request.focus_keyword)
        if request.service:
            req_tokens = req_tokens | _tokenize(request.service)

        req_city: str | None = request.location.city if request.location else None

        best: DraftMatch | None = None
        best_score = SIMILARITY_THRESHOLD - 0.001

        for path in self._output_dir.glob("**/article.json"):
            try:
                article = Article.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # ── Eligibility: only unpublished drafts may be reused ────────────
            art_status = article.status.value if hasattr(article.status, "value") else str(article.status)
            if art_status in {"published", "archived"}:
                continue

            # ── Location gate ─────────────────────────────────────────────────
            art_city = (
                article.request.location.city
                if article.request and article.request.location
                else None
            )
            if not _location_compatible(req_city, art_city):
                continue

            # ── Client safety gate ────────────────────────────────────────────
            art_reuse_group: str | None = getattr(
                article.tenant, "reuse_group", None
            )
            if not _client_allowed(
                tenant.client_id,
                article.tenant.client_id,
                req_reuse_group,
                art_reuse_group,
            ):
                continue

            same_website = (
                article.tenant.client_id == tenant.client_id
                and article.tenant.website_id == tenant.website_id
            )

            # ── Pass 1: topic_id exact match ──────────────────────────────────
            # For articles generated before topic_id was introduced, derive it
            # in-memory from the stored request (backward-compatible, no file write).
            art_topic_id: str | None = getattr(article, "topic_id", None)
            if not art_topic_id and article.request:
                art_topic_id = normalize_topic_id(
                    article.request.topic, article.request.location
                )
            if req_topic_id and art_topic_id and req_topic_id == art_topic_id:
                logger.info(
                    "Draft reuse (topic_id match): '%s' — %s",
                    req_topic_id,
                    path,
                )
                return DraftMatch(
                    article=article,
                    source_path=path,
                    similarity=1.0,
                    same_website=same_website,
                    matched_by_topic_id=True,
                )

            # ── Pass 2: Jaccard similarity ────────────────────────────────────
            candidate_text = " ".join(filter(None, [
                article.request.topic if article.request else "",
                article.title or "",
                article.seo.focus_keyword if article.seo else "",
                (article.seo.slug or "").replace("-", " ") if article.seo else "",
                article.request.service if article.request else "",
            ]))
            score = _jaccard(req_tokens, _tokenize(candidate_text))

            if score > best_score:
                best_score = score
                best = DraftMatch(
                    article=article,
                    source_path=path,
                    similarity=score,
                    same_website=same_website,
                    matched_by_topic_id=False,
                )

        if best:
            logger.info(
                "Draft reuse candidate (Jaccard %.2f): '%s' — %s",
                best.similarity,
                best.article.title[:60],
                best.source_path,
            )

        return best

    def adapt(
        self,
        match: DraftMatch,
        request: "ArticleRequest",
        tenant: "TenantContext",
    ) -> "Article":
        """
        Return a copy of the matched article adapted for the target tenant.

        What changes:
          - tenant updated to caller's client_id / website_id
          - request replaced with the current request
          - WordPress-specific fields (post_id, post_url) cleared; publishing reset to defaults
          - Article lifecycle status reset to REVIEW (ready for publication pipeline)

        What is preserved verbatim:
          - title, content_markdown, image_plans, word_count, reading_time
          - seo metadata (caller must regenerate website-specific SEO fields after adapt)
        """
        from models.enums import ArticleStatus
        from models.publishing import PublishingOptions

        return match.article.model_copy(update={
            "tenant": tenant,
            "request": request,
            "publishing": PublishingOptions(),
            "status": ArticleStatus.REVIEW,
        })

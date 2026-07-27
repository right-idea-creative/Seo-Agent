"""
Editorial scoring layer for image candidate ranking.

Combines Claude Vision semantic scores with editorial diversity signals to
produce a final selection score. Semantic quality is always the dominant factor —
the maximum total editorial adjustment is +5 to −18 on a 0–100 Vision scale.
A 27-point semantic gap (97 vs 70) cannot be fully overridden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from services.editorial_history_service import ImageUsageRecord


def _usage_penalty(times_used: int) -> float:
    # 0 → 0.0, logarithmic, capped at 8.0
    return min(8.0, 2.5 * math.log1p(times_used))


def _recent_penalty(history_slugs: set[str], recent_articles: list[str]) -> float:
    # Scan recent_articles; return penalty based on earliest hit position.
    for position, slug in enumerate(recent_articles):
        if slug in history_slugs:
            if position < 3:
                return 10.0
            if position < 7:
                return 7.0
            if position < 15:
                return 4.0
            if position < 25:
                return 2.0
    return 0.0


def _diversity_bonus(times_used: int) -> float:
    if times_used == 0:
        return 5.0
    if times_used <= 2:
        return 3.0
    if times_used <= 5:
        return 1.0
    return 0.0


@dataclass
class EditorialSelectionResult:
    file_id: str
    filename: str
    folder_path: str
    vision_score: int
    editorial_score: float
    diversity_bonus: float
    usage_penalty: float
    recent_penalty: float
    folder_penalty: float
    times_used: int
    selection_reason: str


def score_candidates(
    candidates: list[tuple[str, str, str, int]],
    history_lookup: dict[str, ImageUsageRecord | None],
    recent_articles: list[str],
    is_featured: bool,
    used_folder_paths: set[str],
) -> list[EditorialSelectionResult]:
    results: list[EditorialSelectionResult] = []

    for file_id, filename, folder_path, vision_score in candidates:
        rec = history_lookup.get(file_id)
        times_used = rec.times_used if rec is not None else 0

        bonus = _diversity_bonus(times_used)
        u_pen = _usage_penalty(times_used)

        history_slugs: set[str] = set()
        if rec is not None:
            history_slugs = {e.slug for e in rec.article_history}

        r_pen = _recent_penalty(history_slugs, recent_articles)

        # Featured extra rule: any featured entry whose slug appears in the last 3 articles.
        if is_featured and rec is not None:
            featured_slugs = {e.slug for e in rec.article_history if e.purpose == "featured"}
            if featured_slugs & set(recent_articles[:3]):
                r_pen = max(r_pen, 12.0)

        f_pen = 3.0 if folder_path in used_folder_paths else 0.0

        editorial_score = vision_score + bonus - u_pen - r_pen - f_pen

        parts = [f"Vision={vision_score}"]
        if bonus > 0:
            label = "never-used" if times_used == 0 else f"{times_used}× used"
            parts.append(f"+{bonus:.0f} freshness ({label})")
        if u_pen > 0:
            parts.append(f"-{u_pen:.1f} usage ({times_used}×)")
        if r_pen > 0:
            parts.append(f"-{r_pen:.0f} recency")
        if f_pen > 0:
            parts.append(f"-{f_pen:.0f} folder-duplicate")
        selection_reason = "  ".join(parts)

        results.append(EditorialSelectionResult(
            file_id=file_id,
            filename=filename,
            folder_path=folder_path,
            vision_score=vision_score,
            editorial_score=editorial_score,
            diversity_bonus=bonus,
            usage_penalty=u_pen,
            recent_penalty=r_pen,
            folder_penalty=f_pen,
            times_used=times_used,
            selection_reason=selection_reason,
        ))

    results.sort(key=lambda r: r.editorial_score, reverse=True)
    return results

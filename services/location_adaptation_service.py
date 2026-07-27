"""
LocationAdaptationService — adapt a reused article to a different target city.

When a draft article was written for city A and is being reused for city B,
this service:

1. Scans every content field for remaining references to the original city.
2. Applies direct, free-form string replacement for all known location patterns
   (covers ~95 % of cases with zero API cost).
3. For any section where location strings still remain after direct replacement,
   makes a *targeted* LLM call to rewrite only that section — never the full article.
4. Returns a ScanReport describing what was found and what was changed.

Scanned fields
--------------
  SEO title, meta description, slug, H1 (article title), H2/H3 headings,
  article body sections, FAQ entries, internal links.

Direct replacement patterns (free, no API)
-------------------------------------------
  {orig_city}                  → {target_city}
  {orig_city}, {orig_state}    → {target_city}, {target_state}
  {orig_city} County           → {target_city} County
  {orig_city} area             → {target_city} area
  {orig_city} metro            → {target_city} metro
  in {orig_city}               → in {target_city}
  Serving {orig_city}          → Serving {target_city}
  Downtown {orig_city}         → Downtown {target_city}
  near {orig_city}             → near {target_city}
  {orig_city}-based            → {target_city}-based
  {orig_state} (abbreviation)  → {target_state}  (only replaced next to city refs)
  (and the case-insensitive equivalents of all of the above)

Targeted LLM rewrite (cheap, Haiku / seo_model)
------------------------------------------------
  Used only when direct replacement leaves residual location tokens in a section.
  The prompt is minimal: "replace all references to {orig} with {target}".
  max_tokens = 1500 per section.  Falls back gracefully if the call fails.

No API call is made unless residual location strings are detected after the
direct replacement pass.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.article import Article, SEOMetadata
    from models.location import Location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SectionResult:
    heading: str          # The H2/H3 heading text (empty string for preamble)
    original: str
    adapted: str
    had_location_refs: bool
    llm_rewritten: bool = False


@dataclass
class ScanReport:
    original_city: str
    target_city: str
    sections_scanned: int = 0
    sections_with_refs: int = 0
    sections_direct_replaced: int = 0
    sections_llm_rewritten: int = 0
    sections_llm_budget_skipped: int = 0   # LLM not called — budget exhausted
    seo_adapted: bool = False
    refs_found: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: location string inventory
# ---------------------------------------------------------------------------

def _location_strings(location: "Location") -> list[str]:
    """
    Return every string that could appear in content as a reference to `location`.

    Ordered longest-first so that more specific patterns are replaced before
    shorter ones (e.g. "San Diego County" before "San Diego").
    """
    city = location.city
    state = location.state
    country = getattr(location, "country", "")
    neighborhood = getattr(location, "neighborhood", None)
    zip_code = getattr(location, "zip_code", None)

    candidates = []
    if neighborhood:
        candidates.append(f"{neighborhood}, {city}")
        candidates.append(neighborhood)
    if zip_code:
        candidates.append(zip_code)
    candidates += [
        f"{city} County",
        f"{city}, {state}",
        f"{city} {state}",
        f"{city} area",
        f"{city} metro",
        f"{city} metropolitan",
        f"{city} region",
        f"{city} community",
        f"{city} residents",
        f"{city} homeowners",
        f"{city}-based",
        f"Downtown {city}",
        f"Serving {city}",
        f"near {city}",
        f"in {city}",
        city,
    ]
    return [s for s in candidates if s]


# ---------------------------------------------------------------------------
# Direct replacement (no API)
# ---------------------------------------------------------------------------

def _replace_location(text: str, original: "Location", target: "Location") -> str:
    """Case-preserving multi-pattern replacement of original location → target location."""
    replacements: list[tuple[str, str]] = []

    orig_strings = _location_strings(original)
    tgt_city = target.city
    tgt_state = target.state

    for orig_str in orig_strings:
        # Build the target string by replacing the city (and optionally state) part.
        # The target string mirrors the structure of the original string.
        if original.state in orig_str:
            tgt_str = orig_str.replace(original.city, tgt_city).replace(original.state, tgt_state)
        else:
            tgt_str = orig_str.replace(original.city, tgt_city)
        replacements.append((orig_str, tgt_str))

    for orig_str, tgt_str in replacements:
        if orig_str == tgt_str:
            continue
        # Replace case-insensitively, preserving surrounding context
        text = re.sub(re.escape(orig_str), tgt_str, text, flags=re.IGNORECASE)

    return text


def _has_location_ref(text: str, location: "Location") -> bool:
    """Return True if any location string appears in text."""
    city_lower = location.city.lower()
    return city_lower in text.lower()


# ---------------------------------------------------------------------------
# Markdown section splitter
# ---------------------------------------------------------------------------

def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown into (heading, body) pairs.

    The first element always has heading="" and contains content before the
    first H2/H3. Each subsequent element starts at an H2 or H3 boundary.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in markdown.splitlines(keepends=True):
        if line.startswith("## ") or line.startswith("### "):
            if current_lines:
                sections.append((current_heading, "".join(current_lines)))
            current_heading = line.rstrip("\n")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "".join(current_lines)))

    return sections


def _join_sections(sections: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for heading, body in sections:
        if heading:
            parts.append(heading + "\n" + body)
        else:
            parts.append(body)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class LocationAdaptationService:
    """
    Adapt a reused article's content from one city to another.

    Usage::

        svc = LocationAdaptationService(claude_service)
        adapted_article, report = svc.adapt(article, original_loc, target_loc)
    """

    def __init__(self, claude_service) -> None:
        self._claude = claude_service

    def adapt(
        self,
        article: "Article",
        original_location: "Location",
        target_location: "Location",
    ) -> tuple["Article", ScanReport]:
        """
        Return (adapted_article, report).

        If original and target cities are identical, returns the article unchanged
        with sections_with_refs=0.
        """
        report = ScanReport(
            original_city=original_location.city,
            target_city=target_location.city,
        )

        if original_location.city.lower() == target_location.city.lower():
            return article, report

        # ── Adapt body ────────────────────────────────────────────────────
        new_markdown, section_results = self._adapt_body(
            article.content_markdown, original_location, target_location, report
        )
        # ── Adapt title ───────────────────────────────────────────────────
        new_title = _replace_location(article.title or "", original_location, target_location)
        # ── Adapt SEO fields ──────────────────────────────────────────────
        new_seo = self._adapt_seo(article.seo, original_location, target_location)
        if new_seo is not article.seo:
            report.seo_adapted = True

        adapted = article.model_copy(update={
            "title": new_title,
            "content_markdown": new_markdown,
            "seo": new_seo,
        })
        return adapted, report

    # ── Body adaptation ────────────────────────────────────────────────────

    def _adapt_body(
        self,
        markdown: str,
        original: "Location",
        target: "Location",
        report: ScanReport,
    ) -> tuple[str, list[SectionResult]]:
        sections = _split_sections(markdown)
        report.sections_scanned = len(sections)
        results: list[SectionResult] = []

        for heading, body in sections:
            full_text = (heading + "\n" + body) if heading else body
            had_refs = _has_location_ref(full_text, original)

            if not had_refs:
                results.append(SectionResult(heading, full_text, full_text, False))
                continue

            report.sections_with_refs += 1
            report.refs_found.append(heading or "(preamble)")

            # Pass 1: direct replacement
            adapted_heading = _replace_location(heading, original, target) if heading else heading
            adapted_body = _replace_location(body, original, target)
            adapted_full = (adapted_heading + "\n" + adapted_body) if adapted_heading else adapted_body
            report.sections_direct_replaced += 1

            # Pass 2: targeted LLM rewrite for residual refs
            if _has_location_ref(adapted_full, original):
                rewritten, budget_blocked = self._llm_adapt_section(adapted_full, original, target)
                if rewritten:
                    adapted_full = rewritten
                    report.sections_llm_rewritten += 1
                    if adapted_heading:
                        results.append(SectionResult(adapted_heading, body, adapted_full, True, True))
                    else:
                        results.append(SectionResult("", body, adapted_full, True, True))
                    continue
                if budget_blocked:
                    report.sections_llm_budget_skipped += 1

            if adapted_heading:
                results.append(SectionResult(adapted_heading, body, adapted_full, True))
            else:
                results.append(SectionResult("", body, adapted_full, True))

        # Reassemble from adapted content
        assembled = _join_sections([
            (r.heading if r.heading else "", r.adapted if r.had_location_refs else
             ((sections[i][0] + "\n" + sections[i][1]) if sections[i][0] else sections[i][1]))
            for i, r in enumerate(results)
        ])
        return assembled, results

    def _llm_adapt_section(
        self,
        section_text: str,
        original: "Location",
        target: "Location",
    ) -> tuple[str | None, bool]:
        """
        Make a targeted, cheap LLM call to rewrite residual location references
        in a single section.

        Returns (result, budget_blocked):
          - (text, False)  — success
          - (None, True)   — skipped because monthly budget is exhausted
          - (None, False)  — skipped due to another error
        """
        from config import settings
        from services.budget_service import BudgetExceededError

        system = (
            "You are an expert editor updating location references in website content. "
            "Replace every reference to the original city with the target city. "
            "Preserve all other content exactly. Return only the updated text."
        )
        user_msg = (
            f"Original city: {original.city}, {original.state}\n"
            f"Target city: {target.city}, {target.state}\n\n"
            f"Text to update:\n\n{section_text}"
        )
        try:
            result = self._claude.generate(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=1500,
                thinking=False,
                model=settings.seo_model,
                label="location:adapt-section",
            )
            return (result.strip() if result else None), False
        except BudgetExceededError:
            logger.warning(
                "Location AI refinement skipped (monthly budget exceeded). "
                "Direct adaptation retained."
            )
            return None, True
        except Exception as exc:
            logger.warning("Location LLM rewrite failed (non-blocking): %s", exc)
            return None, False

    # ── SEO adaptation ─────────────────────────────────────────────────────

    def _adapt_seo(
        self,
        seo: "SEOMetadata",
        original: "Location",
        target: "Location",
    ) -> "SEOMetadata":
        """Direct string replacement on all SEO text fields."""
        new_title = _replace_location(seo.seo_title, original, target)
        new_desc = _replace_location(seo.meta_description, original, target)
        new_slug = _replace_location(seo.slug, original, target)
        new_kw = _replace_location(seo.focus_keyword, original, target)
        new_secondary = [_replace_location(kw, original, target) for kw in seo.secondary_keywords]
        new_tags = [_replace_location(t, original, target) for t in seo.suggested_tags]
        new_cat = _replace_location(seo.suggested_category or "", original, target) or seo.suggested_category
        new_og_title = _replace_location(seo.og_title or "", original, target) or seo.og_title
        new_og_desc = _replace_location(seo.og_description or "", original, target) or seo.og_description

        if (new_title == seo.seo_title and new_desc == seo.meta_description
                and new_slug == seo.slug and new_kw == seo.focus_keyword):
            return seo  # nothing changed

        return seo.model_copy(update={
            "seo_title": new_title,
            "meta_description": new_desc,
            "slug": new_slug,
            "focus_keyword": new_kw,
            "secondary_keywords": new_secondary,
            "suggested_tags": new_tags,
            "suggested_category": new_cat,
            "og_title": new_og_title,
            "og_description": new_og_desc,
        })

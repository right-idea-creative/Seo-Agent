"""
ReuseStatsService — persistent per-month statistics for draft reuse and API cost tracking.

Storage
-------
    output/.reuse_stats.json

Layout::

    {
      "2026-07": {
        "articles_generated":     5,
        "articles_reused":        8,
        "api_calls_avoided":      9,
        "dollars_saved":          4.40,
        "total_cost_usd":         2.75,
        "article_costs":          [0.55, 0.52, 0.00, 0.00, 0.48, ...],
        "reused_topics":          {"door-garage-repair-spring": 3, ...},
        "location_adapted":       2,
        "seo_cache_hits":         5,
        "pool_hits":              8,
        "pool_misses":            5
      }
    }

All monetary values are in USD.
Writes are atomic (temp file + rename).
No API calls are made in this module.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_STATS_FILENAME = ".reuse_stats.json"


def _current_month() -> str:
    """Return 'YYYY-MM' for the current UTC month."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m")


class ReuseStatsService:
    """
    Append-only per-month statistics tracker.

    Usage::

        stats = ReuseStatsService(output_dir)
        stats.record_generation(cost_usd=0.52)
        stats.record_reuse(topic_id="door-garage-repair-spring", savings_usd=0.55)
        stats.record_api_calls_avoided(2)
        stats.record_seo_cache_hit()
        stats.record_pool_hit()
        stats.record_location_adapted()
        stats.save()

        report = stats.monthly_report()
    """

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / _STATS_FILENAME
        self._data: dict = {}
        self._load()

    # ── Record events ──────────────────────────────────────────────────────

    def record_generation(self, cost_usd: float) -> None:
        m = self._month()
        m["articles_generated"] = m.get("articles_generated", 0) + 1
        m.setdefault("article_costs", []).append(round(cost_usd, 6))
        m["total_cost_usd"] = round(m.get("total_cost_usd", 0.0) + cost_usd, 6)
        m["pool_misses"] = m.get("pool_misses", 0) + 1

    def record_reuse(self, topic_id: str, savings_usd: float) -> None:
        m = self._month()
        m["articles_reused"] = m.get("articles_reused", 0) + 1
        m["dollars_saved"] = round(m.get("dollars_saved", 0.0) + savings_usd, 6)
        m.setdefault("article_costs", []).append(0.0)
        topics = m.setdefault("reused_topics", {})
        topics[topic_id] = topics.get(topic_id, 0) + 1
        m["pool_hits"] = m.get("pool_hits", 0) + 1

    def record_api_calls_avoided(self, count: int = 1) -> None:
        m = self._month()
        m["api_calls_avoided"] = m.get("api_calls_avoided", 0) + count

    def record_seo_cache_hit(self) -> None:
        m = self._month()
        m["seo_cache_hits"] = m.get("seo_cache_hits", 0) + 1
        m["api_calls_avoided"] = m.get("api_calls_avoided", 0) + 1

    def record_pool_hit(self) -> None:
        m = self._month()
        m["pool_hits"] = m.get("pool_hits", 0) + 1

    def record_pool_miss(self) -> None:
        m = self._month()
        m["pool_misses"] = m.get("pool_misses", 0) + 1

    def record_location_adapted(self) -> None:
        m = self._month()
        m["location_adapted"] = m.get("location_adapted", 0) + 1

    def record_seo_regen_skipped(self) -> None:
        """SEO regen was skipped because the monthly budget was exhausted."""
        m = self._month()
        m["seo_regens_skipped"] = m.get("seo_regens_skipped", 0) + 1
        m["budget_blocks_minor"] = m.get("budget_blocks_minor", 0) + 1

    def record_location_refinement_skipped(self, count: int = 1) -> None:
        """LLM location refinement was skipped because the monthly budget was exhausted."""
        m = self._month()
        m["location_refinements_skipped"] = m.get("location_refinements_skipped", 0) + count
        m["budget_blocks_minor"] = m.get("budget_blocks_minor", 0) + count

    def record_budget_block_generation(self) -> None:
        """A full article generation was blocked by the monthly budget."""
        m = self._month()
        m["budget_blocks_generation"] = m.get("budget_blocks_generation", 0) + 1

    # ── Reporting ──────────────────────────────────────────────────────────

    def monthly_report(self, month: str | None = None) -> dict:
        """Return the stats dict for `month` (defaults to current month)."""
        key = month or _current_month()
        m = self._data.get(key, {})

        generated = m.get("articles_generated", 0)
        reused = m.get("articles_reused", 0)
        total = generated + reused
        reuse_pct = round(100 * reused / total, 1) if total else 0.0

        costs = m.get("article_costs", [])
        avg_cost = round(sum(costs) / len(costs), 6) if costs else 0.0

        top_topics = sorted(
            m.get("reused_topics", {}).items(),
            key=lambda kv: -kv[1],
        )[:5]

        return {
            "month":                        key,
            "articles_generated":           generated,
            "articles_reused":              reused,
            "total_articles":               total,
            "reuse_percentage":             reuse_pct,
            "api_calls_avoided":            m.get("api_calls_avoided", 0),
            "dollars_saved":                round(m.get("dollars_saved", 0.0), 4),
            "total_cost_usd":               round(m.get("total_cost_usd", 0.0), 4),
            "average_article_cost":         avg_cost,
            "seo_cache_hits":               m.get("seo_cache_hits", 0),
            "seo_regens_skipped":           m.get("seo_regens_skipped", 0),
            "pool_hits":                    m.get("pool_hits", 0),
            "pool_misses":                  m.get("pool_misses", 0),
            "location_adapted":             m.get("location_adapted", 0),
            "location_refinements_skipped": m.get("location_refinements_skipped", 0),
            "budget_blocks_generation":     m.get("budget_blocks_generation", 0),
            "budget_blocks_minor":          m.get("budget_blocks_minor", 0),
            "most_reused_topics":           top_topics,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomically persist stats to disk. Silently no-ops on failure."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".stats_tmp_", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, ensure_ascii=False)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Stats save failed (non-blocking): %s", exc)

    # ── Internal ───────────────────────────────────────────────────────────

    def _month(self) -> dict:
        key = _current_month()
        if key not in self._data:
            self._data[key] = {}
        return self._data[key]

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

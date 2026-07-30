#!/usr/bin/env python3
"""
QA Score Benchmark — post-simplification pipeline.

Generates 25 articles and runs Dual QA without publishing to WordPress,
without image resolution, and without link enrichment.

Draft reuse is disabled so every article comes from the updated pipeline.

Saves one JSON record per article to benchmark_results.jsonl immediately after
each article completes, so a run can be interrupted and resumed without losing
data.  Run again to add more articles to the same file.

Usage:
    python3 benchmark.py [--resume]       # resume appends to existing results
    python3 benchmark.py --report-only    # print report from existing results
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import os
os.chdir(Path(__file__).parent)  # ensure relative paths (credentials/, profiles/) resolve

from config import settings

# ── Logging: INFO to stdout, suppress noisy sub-module debug ──────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ── Pipeline imports ───────────────────────────────────────────────────────────
from models.article import ArticleRequest
from models.location import Location
from models.tenant import TenantContext
from services.business_context_resolver import BusinessContextResolver
from services.claude_service import claude
from agents.article_agent import ArticleAgent
from agents.dual_qa_agent import DualQAAgent, DualQAFailedError

# ── Constants ─────────────────────────────────────────────────────────────────

CLIENT_ID   = "RIMC"
WEBSITE_ID  = "overheaddoornwi"
RESULTS_FILE = Path("benchmark_results.jsonl")

# Matches the site profile: Northwest Indiana, IN — Overhead Door Repair
_LOCATION = Location(city="Northwest Indiana", state="IN", country="USA")
_SERVICE  = "Overhead Door Repair"

# Disable draft reuse — benchmark must exercise the current generation pipeline,
# not reuse older articles that predate the simplification changes.
settings.enable_draft_reuse = False

# 25 diverse topics.  Chosen to span different article types (diagnostic,
# cost, procedure, comparison, seasonal) so the score distribution captures
# topic-driven variance, not just prompt variance.
TOPICS: list[str] = [
    "Garage door cable snapped: symptoms and what to do next",
    "Overhead door opener programming guide for Northwest Indiana homeowners",
    "How much does garage door panel replacement cost",
    "Garage door balance test: why it matters and how to do it",
    "Why overhead door springs break more in winter",
    "Replacing garage door weatherstripping: a practical guide",
    "Overhead door roller types and when to replace them",
    "Garage door keypad not responding: troubleshooting guide",
    "How to choose the right garage door opener horsepower",
    "Overhead door track alignment: causes, symptoms, and fixes",
    "Garage door insulation options for Midwest homeowners",
    "Why your garage door reverses before fully closing",
    "Commercial overhead door maintenance schedule",
    "Garage door torsion spring adjustment: what homeowners should know",
    "Overhead door bottom seal replacement guide",
    "How to extend garage door spring life in Northwest Indiana",
    "Garage door emergency release cord: how and when to use it",
    "Overhead door panel damage: repair or replace",
    "Garage door noise diagnosis: grinding, squeaking, rattling",
    "Smart garage door opener: is the upgrade worth it",
    "Overhead door cable drum failure: signs and repair options",
    "How a garage door counterbalance system works",
    "Why overhead door repairs cost what they do",
    "Garage door photo eye sensor: cleaning, alignment, and repair",
    "Overhead door section vs full door replacement in Northwest Indiana",
]


# ── Result record ─────────────────────────────────────────────────────────────

def _record(
    n: int,
    topic: str,
    *,
    seo: int,
    editorial: int,
    writing: int,
    authenticity: int,
    qa_passed: bool,
    iterations: int,
    gen_seconds: float,
    qa_seconds: float,
    error: str | None = None,
) -> dict:
    return {
        "n": n,
        "topic": topic,
        "seo": seo,
        "editorial": editorial,
        "writing": writing,
        "authenticity": authenticity,
        "qa_passed": qa_passed,
        "iterations": iterations,
        "gen_seconds": round(gen_seconds, 1),
        "qa_seconds": round(qa_seconds, 1),
        "error": error,
    }


def _load_existing() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    records = []
    for line in RESULTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _append(record: dict) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── QA agent ──────────────────────────────────────────────────────────────────

def _build_qa_agent() -> DualQAAgent:
    openai_reviewer = None
    if settings.openai_api_key:
        try:
            from services.openai_review_service import OpenAIReviewService
            openai_reviewer = OpenAIReviewService(
                api_key=settings.openai_api_key,
                text_model=settings.openai_text_review_model,
                vision_model=settings.openai_vision_review_model,
            )
        except Exception as exc:
            print(f"  ⚠ OpenAI setup failed — Claude-only QA: {exc}", file=sys.stderr)
    else:
        print("  ⚠ OPENAI_API_KEY not set — Claude-only QA.", file=sys.stderr)

    return DualQAAgent(
        claude=claude,
        openai_reviewer=openai_reviewer,
        min_seo=settings.qa_min_seo,
        min_editorial=settings.qa_min_editorial,
        min_writing=settings.qa_min_writing,
        min_authenticity=settings.qa_min_authenticity,
        max_cycles=settings.qa_max_cycles,
        enable_rescue=settings.qa_rescue_enabled,
    )


# ── Generation + QA for a single topic ───────────────────────────────────────

def _run_one(
    n: int,
    topic: str,
    tenant: TenantContext,
    agent: ArticleAgent,
    qa: DualQAAgent,
) -> dict:
    request = ArticleRequest(
        topic=topic,
        service=_SERVICE,
        location=_LOCATION,
        word_count=settings.default_word_count,
        language=settings.default_language,
        tone=settings.default_tone,
        website_url=None,
    )
    request = BusinessContextResolver(settings.profiles_dir).resolve(
        CLIENT_ID, WEBSITE_ID, request
    )

    # ── Generation ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        article = agent.generate(request, tenant)
    except Exception as exc:
        return _record(
            n, topic,
            seo=0, editorial=0, writing=0, authenticity=0,
            qa_passed=False, iterations=0,
            gen_seconds=time.perf_counter() - t0, qa_seconds=0.0,
            error=f"GENERATION_ERROR: {exc}",
        )
    gen_seconds = time.perf_counter() - t0

    # ── QA ────────────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        _article, _images, report = qa.run(article, resolved_images=[])
        qa_seconds = time.perf_counter() - t1

        r = report.final_article_review
        return _record(
            n, topic,
            seo=r.seo_score if r else 0,
            editorial=r.editorial_score if r else 0,
            writing=r.writing_score if r else 0,
            authenticity=r.authenticity_score if r else 0,
            qa_passed=True,
            iterations=report.iterations_used,
            gen_seconds=gen_seconds,
            qa_seconds=qa_seconds,
        )

    except DualQAFailedError as exc:
        qa_seconds = time.perf_counter() - t1
        r = exc.report.final_article_review
        return _record(
            n, topic,
            seo=r.seo_score if r else 0,
            editorial=r.editorial_score if r else 0,
            writing=r.writing_score if r else 0,
            authenticity=r.authenticity_score if r else 0,
            qa_passed=False,
            iterations=exc.report.iterations_used,
            gen_seconds=gen_seconds,
            qa_seconds=qa_seconds,
            error="QA_FAILED",
        )

    except Exception as exc:
        return _record(
            n, topic,
            seo=0, editorial=0, writing=0, authenticity=0,
            qa_passed=False, iterations=0,
            gen_seconds=gen_seconds, qa_seconds=time.perf_counter() - t1,
            error=f"QA_ERROR: {exc}",
        )


# ── Statistical report ────────────────────────────────────────────────────────

def _report(records: list[dict]) -> str:
    scored = [r for r in records if r["error"] not in ("GENERATION_ERROR", "QA_ERROR")]
    errors = [r for r in records if r.get("error") in ("GENERATION_ERROR", "QA_ERROR")]

    lines: list[str] = []
    sep = "─" * 62

    lines += [
        "",
        "═" * 62,
        "  QA SCORE DISTRIBUTION — POST-SIMPLIFICATION BENCHMARK",
        "═" * 62,
        f"  Articles attempted : {len(records)}",
        f"  Pipeline errors    : {len(errors)}",
        f"  Scored             : {len(scored)}",
        "",
    ]

    if not scored:
        lines.append("  No scored articles — nothing to report.")
        return "\n".join(lines)

    passed = [r for r in scored if r["qa_passed"]]
    failed = [r for r in scored if not r["qa_passed"]]
    lines += [
        f"  QA PASS            : {len(passed)} / {len(scored)}  "
        f"({100*len(passed)//len(scored)}%)",
        f"  QA FAIL            : {len(failed)} / {len(scored)}  "
        f"({100*len(failed)//len(scored)}%)",
        "",
        sep,
    ]

    def _stat_block(label: str, key: str, threshold: int = 90) -> list[str]:
        vals = [r[key] for r in scored]
        above = sum(1 for v in vals if v >= threshold)
        return [
            f"  {label}",
            f"    min={min(vals):3d}  max={max(vals):3d}  "
            f"mean={statistics.mean(vals):.1f}  "
            f"median={statistics.median(vals):.0f}  "
            f"stdev={statistics.stdev(vals) if len(vals) > 1 else 0:.1f}",
            f"    ≥{threshold}: {above}/{len(vals)} ({100*above//len(vals)}%)",
            "",
        ]

    lines += _stat_block("Claude SEO score",        "seo",           90)
    lines += _stat_block("Claude Editorial score",   "editorial",     90)
    lines += _stat_block("OpenAI Human Writing",     "writing",       90)
    lines += _stat_block("OpenAI Authenticity",      "authenticity",  90)

    # Distribution histogram for writing and authenticity (most informative)
    for label, key in (("Human Writing", "writing"), ("Authenticity", "authenticity")):
        vals = [r[key] for r in scored]
        buckets = {
            "95–100": sum(1 for v in vals if v >= 95),
            " 90–94": sum(1 for v in vals if 90 <= v < 95),
            " 85–89": sum(1 for v in vals if 85 <= v < 90),
            " 80–84": sum(1 for v in vals if 80 <= v < 85),
            "  <80":  sum(1 for v in vals if v < 80),
        }
        lines.append(f"  {label} distribution:")
        for bucket, count in buckets.items():
            bar = "█" * count + "░" * (len(scored) - count)
            lines.append(f"    {bucket}  {bar}  {count}")
        lines.append("")

    lines.append(sep)

    # Iteration breakdown
    cycles_1 = sum(1 for r in scored if r["iterations"] <= 1)
    cycles_2 = sum(1 for r in scored if r["iterations"] == 2)
    cycles_3 = sum(1 for r in scored if r["iterations"] >= 3)
    lines += [
        "  QA revision cycles:",
        f"    1 cycle (first pass)  : {cycles_1}",
        f"    2 cycles              : {cycles_2}",
        f"    3 cycles              : {cycles_3}",
        "",
    ]

    # Timing
    gen_times  = [r["gen_seconds"] for r in scored]
    qa_times   = [r["qa_seconds"]  for r in scored]
    lines += [
        "  Generation time (seconds):",
        f"    min={min(gen_times):.0f}  max={max(gen_times):.0f}  "
        f"mean={statistics.mean(gen_times):.0f}",
        "  QA time (seconds):",
        f"    min={min(qa_times):.0f}  max={max(qa_times):.0f}  "
        f"mean={statistics.mean(qa_times):.0f}",
        "",
    ]

    # Per-article table
    lines += [
        sep,
        "  Article-by-article scores:",
        f"  {'#':>2}  {'SEO':>4}  {'EDI':>4}  {'WRT':>4}  {'AUT':>4}  "
        f"{'CYC':>3}  {'PASS':>4}  {'TOPIC':<45}",
        f"  {'-'*2}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  "
        f"{'-'*3}  {'-'*4}  {'-'*45}",
    ]
    for r in records:
        err = r.get("error") or ""
        topic_short = r["topic"][:45]
        if err in ("GENERATION_ERROR", "QA_ERROR"):
            lines.append(
                f"  {r['n']:>2}  {'ERR':>4}  {'ERR':>4}  {'ERR':>4}  {'ERR':>4}  "
                f"{'—':>3}  {'FAIL':>4}  {topic_short}"
            )
        else:
            result = "PASS" if r["qa_passed"] else "FAIL"
            lines.append(
                f"  {r['n']:>2}  {r['seo']:>4}  {r['editorial']:>4}  "
                f"{r['writing']:>4}  {r['authenticity']:>4}  "
                f"{r['iterations']:>3}  {result:>4}  {topic_short}"
            )

    if errors:
        lines += ["", sep, "  Errors:"]
        for r in errors:
            lines.append(f"    #{r['n']} {r['topic'][:50]}: {r['error']}")

    lines += ["", "═" * 62, ""]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Print report from existing results, don't generate.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip topics already in results file.")
    args = parser.parse_args()

    existing = _load_existing()

    if args.report_only:
        if not existing:
            print("No results found in benchmark_results.jsonl")
            sys.exit(1)
        print(_report(existing))
        return

    # Determine which topics have already been run
    done_topics: set[str] = {r["topic"] for r in existing}

    remaining = [
        (i + 1, topic)
        for i, topic in enumerate(TOPICS)
        if topic not in done_topics
    ]

    if not remaining:
        print("All topics already completed. Use --report-only to see results.")
        print(_report(existing))
        return

    print(f"\nBenchmark: {len(remaining)} articles to generate "
          f"({len(existing)} already done)\n")

    tenant = TenantContext(client_id=CLIENT_ID, website_id=WEBSITE_ID)
    agent  = ArticleAgent(service=claude)
    qa     = _build_qa_agent()

    all_records = list(existing)

    for n, topic in remaining:
        print(f"[{n:>2}/25] {topic[:60]}", end="", flush=True)
        t_start = time.perf_counter()

        rec = _run_one(n, topic, tenant, agent, qa)

        elapsed = time.perf_counter() - t_start
        _append(rec)
        all_records.append(rec)

        if rec.get("error"):
            status = f"ERROR({rec['error'][:20]})"
        elif rec["qa_passed"]:
            status = (f"PASS  SEO={rec['seo']} EDI={rec['editorial']} "
                      f"WRT={rec['writing']} AUT={rec['authenticity']}  "
                      f"{rec['iterations']}cyc")
        else:
            status = (f"FAIL  SEO={rec['seo']} EDI={rec['editorial']} "
                      f"WRT={rec['writing']} AUT={rec['authenticity']}  "
                      f"{rec['iterations']}cyc")

        print(f"  →  {status}  ({elapsed:.0f}s)")

    report_text = _report(all_records)
    print(report_text)

    report_path = Path("benchmark_report.txt")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

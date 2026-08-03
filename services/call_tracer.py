"""
CallTracer — per-pipeline LLM call profiler.

Every LLM call records its stage label, model, token counts, and cost.
Reset once at the start of each pipeline run; print the summary table at the end.

Usage:
    import services.call_tracer as call_tracer

    # At pipeline start:
    tracer = call_tracer.start()

    # After any LLM call (done automatically by ClaudeService / OpenAIReviewService):
    call_tracer.record(stage="plan:article", model="claude-sonnet-4-6",
                       input_tokens=3200, output_tokens=1800, duration_s=4.2)

    # At pipeline end:
    print(tracer.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Per-million-token pricing [input_$/M, output_$/M]
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8":           (5.00, 25.00),
    "claude-opus-4-7":           (5.00, 25.00),
    "claude-opus-4-6":           (5.00, 25.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00,  5.00),
    "claude-haiku-4-5":          (1.00,  5.00),
    "gpt-4o":                    (2.50, 10.00),
    "gpt-4o-mini":               (0.15,  0.60),
}


@dataclass
class CallRecord:
    stage: str
    model: str
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    used: bool = True  # False when output was discarded (e.g. failed revision SEO)

    @property
    def cost_usd(self) -> float:
        in_p, out_p = _PRICING.get(self.model, (5.00, 25.00))
        return (
            self.input_tokens / 1_000_000 * in_p
            + self.cache_creation_tokens / 1_000_000 * in_p * 1.25
            + self.cache_read_tokens / 1_000_000 * in_p * 0.10
            + self.output_tokens / 1_000_000 * out_p
        )


@dataclass
class CallTracer:
    records: list[CallRecord] = field(default_factory=list)

    def record(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        used: bool = True,
    ) -> None:
        self.records.append(CallRecord(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            duration_s=duration_s,
            used=used,
        ))

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def summary(self) -> str:
        """Return a Rich-markup table string of all calls, sorted by cost descending."""
        from io import StringIO
        from rich import box as rich_box
        from rich.console import Console
        from rich.table import Table

        table = Table(
            "Stage", "Model", "In Tok", "Cache↑", "Cache↓", "Out Tok", "Cost", "Time",
            box=rich_box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            title="[bold]LLM Call Profile[/bold]",
        )

        sorted_records = sorted(self.records, key=lambda r: r.cost_usd, reverse=True)
        for r in sorted_records:
            warn = " [yellow]¬used[/yellow]" if not r.used else ""
            cache_create = f"[yellow]{r.cache_creation_tokens:,}[/yellow]" if r.cache_creation_tokens else "-"
            cache_read   = f"[green]{r.cache_read_tokens:,}[/green]"       if r.cache_read_tokens   else "-"
            table.add_row(
                r.stage + warn,
                r.model,
                f"{r.input_tokens:,}",
                cache_create,
                cache_read,
                f"{r.output_tokens:,}",
                f"${r.cost_usd:.4f}",
                f"{r.duration_s:.1f}s",
            )

        total_in     = sum(r.input_tokens          for r in self.records)
        total_create = sum(r.cache_creation_tokens for r in self.records)
        total_read   = sum(r.cache_read_tokens     for r in self.records)
        total_out    = sum(r.output_tokens         for r in self.records)
        total_dur    = sum(r.duration_s            for r in self.records)
        table.add_row(
            f"[bold]TOTAL ({len(self.records)} calls)[/bold]", "",
            f"[bold]{total_in:,}[/bold]",
            f"[bold yellow]{total_create:,}[/bold yellow]" if total_create else "[bold]-[/bold]",
            f"[bold green]{total_read:,}[/bold green]"     if total_read   else "[bold]-[/bold]",
            f"[bold]{total_out:,}[/bold]",
            f"[bold cyan]${self.total_cost():.4f}[/bold cyan]",
            f"[bold]{total_dur:.1f}s[/bold]",
        )

        buf = StringIO()
        Console(file=buf, highlight=False).print(table)
        return buf.getvalue()


# ── Module-level context ──────────────────────────────────────────────────────

_active: CallTracer | None = None


def start() -> CallTracer:
    """Reset the tracer for a new pipeline run. Returns the fresh tracer."""
    global _active
    _active = CallTracer()
    return _active


def get() -> CallTracer | None:
    """Return the active tracer, or None if no run is in progress."""
    return _active


def record(
    *,
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_s: float,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    used: bool = True,
) -> None:
    """Record a call to the active tracer. No-op if no tracer is active."""
    if _active is not None:
        _active.record(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            duration_s=duration_s,
            used=used,
        )

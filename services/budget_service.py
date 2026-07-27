import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Claude Opus 4.8 pricing (per million tokens)
_CLAUDE_INPUT_PRICE_PER_M = 5.00
_CLAUDE_OUTPUT_PRICE_PER_M = 10

# OpenAI gpt-image-1 pricing (1536×1024, quality="high").
# Token-based: ~6208 output image tokens × $40/1M + ~100 input tokens × $5/1M ≈ $0.25/image.
_OPENAI_IMAGE_PRICE = 0.25


class BudgetExceededError(Exception):
    """Raised when a monthly budget limit would be exceeded."""


class BudgetService:
    """
    File-based monthly budget tracker for Claude and OpenAI spend.

    One JSON file per calendar month under budget_dir (e.g. budget/2025-06.json).
    A new month automatically starts a fresh file.

    Inject as an optional dependency into ClaudeService and OpenAIImageGenerator.
    """

    def __init__(
        self,
        budget_dir: Path,
        claude_limit: float,
        openai_limit: float,
        year_month: str | None = None,
    ) -> None:
        self._dir = budget_dir
        self._claude_limit = claude_limit
        self._openai_limit = openai_limit
        self._year_month = year_month  # injected in tests; None = use current month

    # ── Public interface ──────────────────────────────────────────────────────

    def check_claude(self) -> None:
        """Raise BudgetExceededError if Claude limit is already reached."""
        data = self._load()
        spent = data["claude"]["usd"]
        if spent >= self._claude_limit:
            raise BudgetExceededError(
                f"Claude monthly budget exceeded: ${spent:.2f} / ${self._claude_limit:.2f}. "
                "Article generation is blocked until next month."
            )

    def check_openai(self) -> None:
        """Raise BudgetExceededError if OpenAI limit is already reached."""
        data = self._load()
        spent = data["openai"]["usd"]
        if spent >= self._openai_limit:
            raise BudgetExceededError(
                f"OpenAI monthly budget exceeded: ${spent:.2f} / ${self._openai_limit:.2f}. "
                "AI image generation will be skipped this month."
            )

    def check_monthly_total(self, max_usd: float) -> None:
        """Raise BudgetExceededError if combined Claude + OpenAI spend meets or exceeds max_usd."""
        data = self._load()
        total = data["claude"]["usd"] + data["openai"]["usd"]
        if total >= max_usd:
            raise BudgetExceededError(
                f"Monthly budget exceeded: ${total:.2f} spent / ${max_usd:.2f} limit. "
                "Article generation is blocked until the next billing month."
            )

    def total_spent(self) -> float:
        """Return combined Claude + OpenAI spend for the current month."""
        data = self._load()
        return round(data["claude"]["usd"] + data["openai"]["usd"], 6)

    def record_claude(self, input_tokens: int, output_tokens: int) -> None:
        """Record a Claude API call and persist the updated totals."""
        cost = (
            input_tokens / 1_000_000 * _CLAUDE_INPUT_PRICE_PER_M
            + output_tokens / 1_000_000 * _CLAUDE_OUTPUT_PRICE_PER_M
        )
        data = self._load()
        c = data["claude"]
        c["calls"] += 1
        c["input_tokens"] += input_tokens
        c["output_tokens"] += output_tokens
        c["usd"] = round(c["usd"] + cost, 6)
        self._save(data)
        logger.debug(
            "Claude recorded: +%d/%d tokens, +$%.4f (total $%.4f)",
            input_tokens, output_tokens, cost, c["usd"],
        )

    def record_openai(self, images: int = 1) -> None:
        """Record OpenAI image generation and persist the updated totals."""
        cost = images * _OPENAI_IMAGE_PRICE
        data = self._load()
        o = data["openai"]
        o["calls"] += 1
        o["images"] += images
        o["usd"] = round(o["usd"] + cost, 6)
        self._save(data)
        logger.debug(
            "OpenAI recorded: +%d image(s), +$%.4f (total $%.4f)",
            images, cost, o["usd"],
        )

    def status(self) -> dict:
        """Return current month's spend summary with limits."""
        data = self._load()
        return {
            "month": data["month"],
            "claude": {
                **data["claude"],
                "limit_usd": self._claude_limit,
                "remaining_usd": round(max(0.0, self._claude_limit - data["claude"]["usd"]), 6),
            },
            "openai": {
                **data["openai"],
                "limit_usd": self._openai_limit,
                "remaining_usd": round(max(0.0, self._openai_limit - data["openai"]["usd"]), 6),
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _current_month(self) -> str:
        if self._year_month:
            return self._year_month
        from datetime import datetime
        return datetime.now().strftime("%Y-%m")

    def _path(self) -> Path:
        return self._dir / f"{self._current_month()}.json"

    def _load(self) -> dict:
        path = self._path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt budget file %s — resetting.", path)
        return self._empty(self._current_month())

    def _save(self, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path().write_text(json.dumps(data, indent=2))

    @staticmethod
    def _empty(month: str) -> dict:
        return {
            "month": month,
            "claude": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            "openai": {"calls": 0, "images": 0, "usd": 0.0},
        }

"""
Regression test: BudgetService.record_claude() and record_openai() must use
an exclusive file lock to prevent race conditions from concurrent processes.

Root cause:
    record_claude() and record_openai() performed read-modify-write sequences
    on the budget JSON file without holding any lock. Two concurrent processes
    could both read the same starting state, each add their own spend, and the
    last writer would overwrite the first's update — silently dropping spend.

Fix: fcntl.flock exclusive lock on a .budget.lock file, held across the
read-modify-write sequence in both record methods.

Note: threading.Thread shares the same process and therefore the same fcntl
lock owner (per POSIX, flock is per open-file-description, not per thread).
True concurrent correctness requires separate processes. This test verifies
the sequential behavior and that the lock mechanism doesn't break anything.
We also verify that the lock file is created and that concurrent sequential
calls accumulate correctly.
"""
import tempfile
from pathlib import Path

import pytest
from services.budget_service import BudgetService


def _svc(tmp_dir: Path, ym: str = "2026-07") -> BudgetService:
    return BudgetService(
        budget_dir=tmp_dir,
        claude_limit=100.0,
        openai_limit=50.0,
        year_month=ym,
    )


class TestBudgetServiceLocking:
    def test_record_claude_accumulates_across_sequential_calls(self, tmp_path):
        """Sequential record_claude calls accumulate spend correctly."""
        svc = _svc(tmp_path)
        svc.record_claude(input_tokens=1_000_000, output_tokens=0, model="claude-sonnet-4")
        svc.record_claude(input_tokens=1_000_000, output_tokens=0, model="claude-sonnet-4")
        status = svc.status()
        # 2 × (1M input × $3.00/1M) = $6.00
        assert abs(status["claude"]["usd"] - 6.0) < 0.001, (
            f"Expected $6.00 spend after 2 record_claude calls, got ${status['claude']['usd']:.4f}. "
            "Concurrent overwrites would produce $3.00 (only the last write survives)."
        )
        assert status["claude"]["calls"] == 2

    def test_record_openai_accumulates_across_sequential_calls(self, tmp_path):
        """Sequential record_openai calls accumulate correctly."""
        svc = _svc(tmp_path)
        svc.record_openai(images=1)
        svc.record_openai(images=1)
        svc.record_openai(images=1)
        status = svc.status()
        # 3 × $0.25 = $0.75
        assert abs(status["openai"]["usd"] - 0.75) < 0.001
        assert status["openai"]["images"] == 3

    def test_lock_file_created_on_record(self, tmp_path):
        """The .budget.lock file must be created during record operations."""
        svc = _svc(tmp_path)
        lock_path = tmp_path / ".budget.lock"
        assert not lock_path.exists(), "Lock file should not exist before first record."
        svc.record_claude(input_tokens=1000, output_tokens=0, model="claude-haiku-4")
        assert lock_path.exists(), (
            ".budget.lock must be created by _exclusive_lock. "
            "Absent lock file means locking is not operational."
        )

    def test_mixed_claude_and_openai_accumulate_independently(self, tmp_path):
        """Claude and OpenAI spend accumulate independently without interference."""
        svc = _svc(tmp_path)
        svc.record_claude(input_tokens=1_000_000, output_tokens=0, model="claude-haiku-4")
        svc.record_openai(images=2)
        svc.record_claude(input_tokens=0, output_tokens=1_000_000, model="claude-haiku-4")
        status = svc.status()
        # Claude: 1M input × $1.00 + 1M output × $5.00 = $6.00
        assert abs(status["claude"]["usd"] - 6.0) < 0.001
        # OpenAI: 2 × $0.25 = $0.50
        assert abs(status["openai"]["usd"] - 0.50) < 0.001

    def test_record_does_not_corrupt_existing_data(self, tmp_path):
        """Recording on top of existing data preserves all prior fields."""
        svc = _svc(tmp_path)
        svc.record_claude(input_tokens=500_000, output_tokens=100_000, model="claude-sonnet-4")
        initial = svc.status()

        svc.record_claude(input_tokens=200_000, output_tokens=50_000, model="claude-sonnet-4")
        updated = svc.status()

        assert updated["claude"]["calls"] == 2
        assert updated["claude"]["input_tokens"] == initial["claude"]["input_tokens"] + 200_000
        assert updated["claude"]["output_tokens"] == initial["claude"]["output_tokens"] + 50_000
        assert updated["claude"]["usd"] > initial["claude"]["usd"]

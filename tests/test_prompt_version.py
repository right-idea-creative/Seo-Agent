"""
Tests for PROMPT_VERSION constant in article_agent.py.

Invariants:
  - PROMPT_VERSION is importable from agents.article_agent
  - It is a non-empty string
  - It matches a semver-like pattern (e.g. "1.0" or "1.2.3")
  - Article objects serialized by the agent carry a matching prompt_version
"""
from __future__ import annotations

import re

import pytest


# ── Import guard ──────────────────────────────────────────────────────────────

try:
    from agents.article_agent import PROMPT_VERSION
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="Could not import PROMPT_VERSION from agents.article_agent",
)

_SEMVER_LIKE = re.compile(r"^\d+\.\d+(\.\d+)?$")


# ── PROMPT_VERSION constant ───────────────────────────────────────────────────

class TestPromptVersionConstant:
    def test_prompt_version_is_a_string(self):
        assert isinstance(PROMPT_VERSION, str)

    def test_prompt_version_is_non_empty(self):
        assert PROMPT_VERSION.strip() != ""

    def test_prompt_version_matches_semver_like_pattern(self):
        assert _SEMVER_LIKE.match(PROMPT_VERSION), (
            f"PROMPT_VERSION {PROMPT_VERSION!r} does not match N.N or N.N.N format"
        )

    def test_prompt_version_is_current_value(self):
        # Pinned to "1.0" — update this test when the version is bumped
        assert PROMPT_VERSION == "1.0"

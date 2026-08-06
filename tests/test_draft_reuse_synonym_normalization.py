"""
Regression tests for synonym normalization consistency in the draft reuse pipeline.

Covers:
  - apply_synonyms() correctness
  - _tokenize() applies synonym normalization
  - Equivalent topic phrasings produce the same token set
  - Non-equivalent topics produce distinct token sets
  - Jaccard threshold boundary cases
  - The specific false-negative fixed in Request 7
"""
from __future__ import annotations

import re

import pytest

from services.draft_reuse_service import SIMILARITY_THRESHOLD, _jaccard, _tokenize
from services.topic_normalization import _SYNONYMS, _STOP, apply_synonyms, normalize_topic_id


# ---------------------------------------------------------------------------
# apply_synonyms
# ---------------------------------------------------------------------------

class TestApplySynonyms:
    def test_known_plural_noun(self):
        assert apply_synonyms("springs") == "spring"

    def test_known_problem_descriptor(self):
        assert apply_synonyms("broken") == "repair"

    def test_known_brand_to_category(self):
        assert apply_synonyms("liftmaster") == "opener"

    def test_known_overhead_to_door(self):
        assert apply_synonyms("overhead") == "door"

    def test_verb_inflection(self):
        assert apply_synonyms("repairing") == "repair"
        assert apply_synonyms("installed") == "install"
        assert apply_synonyms("replacement") == "replace"

    def test_unknown_word_returned_unchanged(self):
        assert apply_synonyms("garage") == "garage"
        assert apply_synonyms("spring") == "spring"

    def test_empty_string(self):
        assert apply_synonyms("") == ""


# ---------------------------------------------------------------------------
# _tokenize — synonym normalization applied
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_plural_normalized_to_singular(self):
        tokens = _tokenize("garage door springs")
        assert "spring" in tokens
        assert "springs" not in tokens

    def test_problem_descriptor_normalized_to_repair(self):
        tokens = _tokenize("broken garage door")
        assert "repair" in tokens
        assert "broken" not in tokens

    def test_overhead_normalized_to_door(self):
        tokens = _tokenize("overhead door spring")
        # both "overhead" and "door" collapse to "door"
        assert "door" in tokens
        assert "overhead" not in tokens

    def test_brand_normalized_to_category(self):
        tokens = _tokenize("liftmaster garage door opener")
        assert "opener" in tokens
        assert "liftmaster" not in tokens

    def test_stopwords_removed_after_synonym(self):
        # Ensure no stopword residue leaks through even after synonym application
        from services.draft_reuse_service import _STOPWORDS
        tokens = _tokenize("the broken garage door springs in your home")
        for t in tokens:
            assert t not in _STOPWORDS, f"stopword {t!r} survived"

    def test_canonical_tokens_deduplicated(self):
        # "overhead" and "door" both → "door"; frozenset deduplicates
        tokens = _tokenize("overhead door")
        assert tokens.count("door") == 1 if hasattr(tokens, "count") else "door" in tokens

    def test_short_tokens_filtered(self):
        tokens = _tokenize("do it")
        assert not tokens  # "do" in stopwords, "it" in stopwords


# ---------------------------------------------------------------------------
# Equivalent topic phrasings → identical token sets
# ---------------------------------------------------------------------------

class TestEquivalentTopics:
    def test_broken_springs_matches_spring_repair(self):
        """Problem descriptor + plural should normalize to same token set as singular + repair."""
        tokens_a = _tokenize("Broken Garage Door Springs")
        tokens_b = _tokenize("Garage Door Spring Repair")
        assert tokens_a == tokens_b

    def test_stuck_door_matches_door_repair(self):
        tokens_a = _tokenize("Stuck Garage Door")
        tokens_b = _tokenize("Garage Door Repair")
        assert tokens_a == tokens_b

    def test_damaged_cables_matches_cable_repair(self):
        tokens_a = _tokenize("Damaged Garage Door Cables")
        tokens_b = _tokenize("Garage Door Cable Repair")
        assert tokens_a == tokens_b

    def test_liftmaster_opener_matches_generic_opener(self):
        tokens_a = _tokenize("LiftMaster Garage Door Opener")
        tokens_b = _tokenize("Garage Door Opener")
        # brand maps to "opener"; duplicate deduplicates
        assert tokens_a == tokens_b

    def test_overhead_door_matches_garage_door(self):
        """overhead → door, so 'overhead door' and 'garage door' share the 'door' token."""
        tokens_a = _tokenize("overhead door spring")
        tokens_b = _tokenize("garage door spring")
        # Not identical (one has "garage" extra), but "door" and "spring" are shared
        assert "door" in tokens_a
        assert "spring" in tokens_a
        assert "door" in tokens_b
        assert "spring" in tokens_b


# ---------------------------------------------------------------------------
# Non-equivalent topics produce distinct token sets
# ---------------------------------------------------------------------------

class TestNonEquivalentTopics:
    def test_opener_vs_spring_remain_distinct(self):
        tokens_a = _tokenize("Garage Door Opener Repair")
        tokens_b = _tokenize("Garage Door Spring Repair")
        assert tokens_a != tokens_b

    def test_installation_vs_repair_remain_distinct(self):
        tokens_a = _tokenize("Garage Door Installation")
        tokens_b = _tokenize("Garage Door Repair")
        # "install" vs "repair" — different canonical forms
        assert "install" in tokens_a
        assert "repair" in tokens_b
        assert tokens_a != tokens_b

    def test_insulation_vs_repair_remain_distinct(self):
        tokens_a = _tokenize("Insulated Garage Door")
        tokens_b = _tokenize("Garage Door Repair")
        assert "insulate" in tokens_a
        assert tokens_a != tokens_b


# ---------------------------------------------------------------------------
# Jaccard threshold boundary cases
# ---------------------------------------------------------------------------

class TestJaccardBoundary:
    def test_identical_sets_score_one(self):
        tokens = _tokenize("garage door spring repair")
        assert _jaccard(tokens, tokens) == 1.0

    def test_disjoint_sets_score_zero(self):
        a = _tokenize("garage door spring repair")
        b = _tokenize("swimming pool cleaning")
        assert _jaccard(a, b) == 0.0

    def test_empty_sets_score_zero(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0
        assert _jaccard(_tokenize("the"), frozenset()) == 0.0

    def test_threshold_constant_unchanged(self):
        assert SIMILARITY_THRESHOLD == 0.72

    def test_score_just_above_threshold(self):
        # Crafted pair with known overlap
        # 4 shared out of 5 total → 0.80 > 0.72
        a = frozenset({"door", "garage", "opener", "repair"})
        b = frozenset({"door", "garage", "opener", "repair", "service"})
        assert _jaccard(a, b) > SIMILARITY_THRESHOLD

    def test_score_just_below_threshold(self):
        # 3 shared out of 6 total → 0.50 < 0.72
        a = frozenset({"broken", "garage", "door", "opener"})  # old (pre-fix) tokens
        b = frozenset({"garage", "door", "opener", "repair", "service"})
        assert _jaccard(a, b) < SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Request 7 specific regression — the confirmed false negative fixed
# ---------------------------------------------------------------------------

class TestRequest7FalseNegativeFixed:
    """
    Real pair from the production draft pool (both articles in overheaddoornwi):

      A: "How to Tell If Your Overhead Door Spring Is About to Break: Signs and
         Safety Tips for Northwest Indiana Homeowners"
      B: "How to Tell If Your Garage Door Spring Is Broken: Signs and Safety
         Tips for Northwest Indiana Homeowners"

    Before fix: Jaccard ≈ 0.692 < 0.72 → false negative (missed reuse).
    After fix:  Jaccard ≈ 0.727 > 0.72 → correct reuse hit.
    """

    TOPIC_A = (
        "How to Tell If Your Overhead Door Spring Is About to Break: "
        "Signs and Safety Tips for Northwest Indiana Homeowners"
    )
    TOPIC_B = (
        "How to Tell If Your Garage Door Spring Is Broken: "
        "Signs and Safety Tips for Northwest Indiana Homeowners"
    )

    def _tokenize_old(self, text: str) -> frozenset[str]:
        """Pre-fix tokenizer (no synonym application)."""
        from services.draft_reuse_service import _STOPWORDS
        words = re.findall(r"[a-z]+", text.lower())
        return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)

    def test_old_tokenizer_scores_below_threshold(self):
        a = self._tokenize_old(self.TOPIC_A)
        b = self._tokenize_old(self.TOPIC_B)
        score = _jaccard(a, b)
        assert score < SIMILARITY_THRESHOLD, (
            f"Expected pre-fix score below {SIMILARITY_THRESHOLD}, got {score:.3f}"
        )

    def test_new_tokenizer_scores_above_threshold(self):
        a = _tokenize(self.TOPIC_A)
        b = _tokenize(self.TOPIC_B)
        score = _jaccard(a, b)
        assert score >= SIMILARITY_THRESHOLD, (
            f"Expected post-fix score ≥ {SIMILARITY_THRESHOLD}, got {score:.3f}"
        )

    def test_fix_resolves_false_negative(self):
        """Confirm the specific improvement: score crossed the threshold."""
        old_score = _jaccard(self._tokenize_old(self.TOPIC_A), self._tokenize_old(self.TOPIC_B))
        new_score = _jaccard(_tokenize(self.TOPIC_A), _tokenize(self.TOPIC_B))
        assert old_score < SIMILARITY_THRESHOLD, f"old={old_score:.3f} should be below threshold"
        assert new_score >= SIMILARITY_THRESHOLD, f"new={new_score:.3f} should be at/above threshold"
        assert new_score > old_score, "Fix must strictly improve the score"


# ---------------------------------------------------------------------------
# normalize_topic_id — consistency check (unchanged behavior)
# ---------------------------------------------------------------------------

class TestNormalizeTopicIdUnchanged:
    """Verify normalize_topic_id still produces stable outputs (no regression)."""

    def test_broken_springs_equals_spring_repair(self):
        assert normalize_topic_id("Broken Garage Door Springs") == normalize_topic_id("Garage Door Spring Repair")

    def test_opener_repair_stable(self):
        assert normalize_topic_id("Garage Door Opener Repair") == "door-garage-opener-repair"

    def test_spring_repair_stable(self):
        assert normalize_topic_id("Garage Door Spring Repair") == "door-garage-repair-spring"

    def test_apply_synonyms_consistent_with_normalize(self):
        """apply_synonyms must map the same words that normalize_topic_id maps."""
        for word, canonical in _SYNONYMS.items():
            assert apply_synonyms(word) == canonical, f"Mismatch for {word!r}"

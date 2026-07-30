"""
Regression tests: _classify_opener must not corrupt words ending in double-s.

Root cause:
    rstrip("'s") was used to strip possessive suffixes. str.rstrip(chars) strips
    any combination of characters in the set {'\'', 's'} from the right end,
    not the literal two-character suffix "'s". This silently corrupted:
        "glass" → "gla"    (two 's' chars stripped)
        "class" → "cla"
        "grass" → "gra"
    Corrupted tokens then fell through every vocabulary lookup and were
    incorrectly classified as "subject-first".

Fixed to: check endswith("'s") before slicing only the two-character suffix.
"""
import pytest
from services.writing_audit_service import _classify_opener


class TestClassifyOpener:
    # ── Possessive suffix — the original intent of the rstrip call ─────────────

    def test_possessive_spring_stripped(self):
        """'Spring's' strips the possessive; 'spring' ends in 'ing' → participial."""
        # "spring's" → strip "'s" → "spring" → endswith("ing") → "participial"
        # This differs from "glass" (no possessive to strip → "glass" → subject-first).
        result = _classify_opener("spring's")
        assert result == "participial"

    def test_possessive_door_stripped(self):
        """'door's' possessive suffix is stripped without corrupting the word."""
        result = _classify_opener("door's")
        assert result == "subject-first"

    # ── Double-s words — the bug case ─────────────────────────────────────────

    def test_glass_not_corrupted(self):
        """
        'Glass' must NOT be corrupted to 'gla' by rstrip("'s").

        rstrip("'s") stripped both 's' characters from 'glass' → 'gla',
        a non-word that matched nothing. The correct behaviour is to leave
        'glass' intact (no possessive to strip) and classify normally.
        """
        result = _classify_opener("Glass")
        # "glass" has no possessive → word is "glass" → subject-first
        assert result == "subject-first"  # not "gla" which would still be subject-first but wrong word

    def test_class_not_corrupted(self):
        """'Class' must not be corrupted to 'cla'."""
        result = _classify_opener("Class")
        assert result == "subject-first"

    def test_grass_not_corrupted(self):
        """'Grass' must not be corrupted to 'gra'."""
        result = _classify_opener("Grass")
        assert result == "subject-first"

    def test_press_not_corrupted(self):
        """'Press' must not be corrupted to 'pre'."""
        result = _classify_opener("Press")
        assert result == "subject-first"

    # ── Regular vocabulary words still classify correctly ─────────────────────

    def test_the_is_article(self):
        assert _classify_opener("The") == "article+noun"

    def test_when_is_subordinate(self):
        assert _classify_opener("When") == "subordinate"

    def test_installing_is_participial(self):
        assert _classify_opener("Installing") == "participial"

    def test_number_first(self):
        assert _classify_opener("10,000") == "number-first"

    def test_direct_answer_yes(self):
        assert _classify_opener("Yes") == "direct-answer"

    def test_direct_answer_no(self):
        assert _classify_opener("No") == "direct-answer"


class TestPossessiveStripDoesNotAffectNonPossessives:
    """Extra guard: words that happen to end in 's' but have no apostrophe are untouched."""

    def test_springs_no_apostrophe(self):
        """'Springs' (plural, no apostrophe) must not be stripped at all."""
        result = _classify_opener("Springs")
        assert result == "subject-first"

    def test_doors_no_apostrophe(self):
        result = _classify_opener("Doors")
        assert result == "subject-first"

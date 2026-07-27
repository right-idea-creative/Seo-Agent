"""
topic_normalization — deterministic, LLM-free topic identifier generation.

Converts a free-text topic (e.g. "Broken Garage Door Springs in Denver") into a
stable, location-agnostic, semantically-normalized kebab slug that is the same
for all surface-form variants of the same underlying topic.

Algorithm:
  1. Lower-case + strip location words (city, state, country).
  2. Tokenize into word-only tokens (a-z).
  3. Apply synonym map: normalize plural nouns, verb inflections, and
     problem-descriptor words (broken/damaged → repair) to their canonical form.
  4. Remove stop words and single/double-char tokens.
  5. Deduplicate (preserves first occurrence before sorting).
  6. Sort alphabetically → canonical, order-independent form.
  7. Return the first 8 tokens joined with "-".

No API calls. Fully deterministic.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.location import Location

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------
_STOP: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
    "from", "has", "he", "how", "in", "is", "it", "its", "of", "on",
    "or", "our", "the", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
    "about", "also", "can", "did", "does", "get", "got", "had",
    "have", "him", "his", "her", "into", "its", "just", "let",
    "may", "more", "much", "my", "need", "now", "not", "off",
    "out", "own", "per", "put", "set", "she", "so", "some",
    "such", "than", "that", "the", "them", "then", "their",
    "they", "too", "us", "use", "via", "vs", "versus",
})

# ---------------------------------------------------------------------------
# Synonym map — maps any variant to its canonical form
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, str] = {
    # ── Plural nouns → singular ───────────────────────────────────────────
    "doors": "door",
    "springs": "spring",
    "openers": "opener",
    "cables": "cable",
    "tracks": "track",
    "panels": "panel",
    "sections": "section",
    "sensors": "sensor",
    "motors": "motor",
    "rollers": "roller",
    "hinges": "hinge",
    "remotes": "remote",
    "keypads": "keypad",
    "torsions": "torsion",
    "extensions": "extension",
    "repairs": "repair",
    "installs": "install",
    "services": "service",
    "garages": "garage",
    "locks": "lock",
    "struts": "strut",
    "drives": "drive",
    "belts": "belt",
    "chains": "chain",
    "screws": "screw",
    "coils": "coil",
    "drums": "drum",
    "bearings": "bearing",
    "brackets": "bracket",
    "seals": "seal",
    "weatherstrips": "weatherstrip",
    # ── Verb inflections → base form ──────────────────────────────────────
    "repairing": "repair",
    "repaired": "repair",
    "fix": "repair",
    "fixes": "repair",
    "fixing": "repair",
    "fixed": "repair",
    "installing": "install",
    "installed": "install",
    "installation": "install",
    "replacing": "replace",
    "replaced": "replace",
    "replacement": "replace",
    "servicing": "service",
    "maintaining": "service",
    "maintenance": "service",
    "adjusting": "adjust",
    "adjusted": "adjust",
    "adjustment": "adjust",
    "lubricating": "lubricate",
    "lubrication": "lubricate",
    "lubricant": "lubricate",
    "testing": "test",
    "tested": "test",
    "balancing": "balance",
    "balanced": "balance",
    "aligning": "align",
    "alignment": "align",
    "aligned": "align",
    "painting": "paint",
    "painted": "paint",
    "insulating": "insulate",
    "insulation": "insulate",
    "insulated": "insulate",
    "weatherstripping": "weatherstrip",
    # ── Problem descriptors → service action ──────────────────────────────
    # In the garage door niche, "broken spring" ≈ "spring repair"
    # and "damaged cable" ≈ "cable repair" — these are the same service topic.
    "broken": "repair",
    "broke": "repair",
    "breaking": "repair",
    "damaged": "repair",
    "damage": "repair",
    "faulty": "repair",
    "failed": "repair",
    "failing": "repair",
    "failure": "repair",
    "noisy": "repair",
    "stuck": "repair",
    "jammed": "repair",
    "worn": "repair",
    "squeaky": "repair",
    "bent": "repair",
    "snapped": "repair",
    "snapping": "repair",
    "snaps": "repair",
    # ── Orthographic variants ─────────────────────────────────────────────
    "liftmaster": "opener",      # brand → product category
    "chamberlain": "opener",
    "craftsman": "opener",
    "genie": "opener",
    "hormann": "opener",
    "clopay": "door",
    "amarr": "door",
    "wayne": "door",             # Wayne Dalton
    "dalton": "door",
    "overhead": "door",
    "rollup": "door",
    "sectional": "door",
    "tilt": "door",
    "swinging": "door",
    # ── Modifier synonyms ─────────────────────────────────────────────────
    "automatic": "auto",
    "automatically": "auto",
    "manual": "manual",
    "manually": "manual",
    "smart": "smart",
    "insulated": "insulate",
    "uninsulated": "insulate",   # same topic: insulated vs non-insulated
    "non": "non",
    "double": "double",
    "single": "single",
}


def normalize_topic_id(topic: str, location: "Location | None" = None) -> str:
    """
    Return a stable, location-agnostic topic identifier in kebab-case.

    Semantically equivalent topic phrasings produce the same identifier:

        "Garage Door Spring Repair"        → "door-garage-repair-spring"
        "Garage Door Spring Repairs"       → "door-garage-repair-spring"
        "Broken Garage Door Springs"       → "door-garage-repair-spring"
        "Repair Garage Door Spring"        → "door-garage-repair-spring"
        "Garage Door Broken Spring Repair" → "door-garage-repair-spring"

        "Garage Door Opener Repair"        → "door-garage-opener-repair"
        "Repair Garage Door Opener"        → "door-garage-opener-repair"
        "Broken Garage Door Opener"        → "door-garage-opener-repair"

    Parameters
    ----------
    topic:
        Free-text topic string, typically from ArticleRequest.topic.
    location:
        Optional Location object whose city/state/country values are stripped
        before tokenization so the ID is portable across cities.

    Returns
    -------
    str
        Kebab-case slug of ≤8 sorted canonical tokens, e.g. "door-garage-repair-spring".
        Returns "unknown" if no meaningful tokens survive normalization.
    """
    text = topic.lower()

    # ── Strip location words ───────────────────────────────────────────────
    if location:
        for part in filter(None, [location.city, location.state, location.country]):
            text = text.replace(part.lower(), " ")

    # ── Tokenize, apply synonyms, filter ──────────────────────────────────
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in re.findall(r"[a-z]+", text):
        canonical = _SYNONYMS.get(raw, raw)
        if canonical in _STOP or len(canonical) <= 2 or canonical in seen:
            continue
        seen.add(canonical)
        tokens.append(canonical)

    # ── Sort for canonical ordering (order-independent) ───────────────────
    tokens.sort()

    return "-".join(tokens[:8]) or "unknown"

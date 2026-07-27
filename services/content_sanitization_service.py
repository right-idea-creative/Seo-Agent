"""
ContentSanitizationService — removes website UI artifacts from article markdown.

Runs BEFORE the Publication Readiness Gate and BEFORE image resolution so
the entire pipeline sees clean editorial content.

Conservative by design: only removes content that exists as a STANDALONE
paragraph (text block between blank lines) and matches a known UI pattern.
Embedded links, contextual service mentions, and editorial CTAs inside prose
are never touched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SanitizationResult:
    markdown: str
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed)


class ContentSanitizationService:
    """
    Cleans article markdown before it enters the publication pipeline.

    Detection strategy: split markdown into paragraph blocks (separated by
    blank lines). A block is removed only when its stripped text matches a UI
    pattern exactly — words embedded within a sentence that also contain other
    content are preserved.
    """

    # (regex applied to stripped block text, human-readable label)
    # Each pattern matches when it accounts for the ENTIRE block content.
    _BLOCK_PATTERNS: list[tuple[re.Pattern, str]] = [
        # ── Navigation bars ────────────────────────────────────────────────
        # Three or more pipe-separated words/phrases on a single line
        (
            re.compile(
                r"^(?:[A-Za-z][A-Za-z\s]{0,25}\s*\|\s*){2,}[A-Za-z][A-Za-z\s]{0,25}$",
                re.IGNORECASE,
            ),
            "Navigation bar",
        ),
        # ── Call-to-action buttons ─────────────────────────────────────────
        (re.compile(r"^call\s+now[!.]?$", re.IGNORECASE), "Call Now button"),
        (re.compile(r"^call\s+us\s+(now|today)[!.]?$", re.IGNORECASE), "Call Us button"),
        (re.compile(r"^book\s+now[!.]?$", re.IGNORECASE), "Book Now button"),
        (re.compile(r"^book\s+online[!.]?$", re.IGNORECASE), "Book Online button"),
        (re.compile(r"^request\s+(a\s+)?service[!.]?$", re.IGNORECASE), "Request Service button"),
        (re.compile(r"^request\s+(a\s+)?(?:free\s+)?quote[!.]?$", re.IGNORECASE), "Request Quote button"),
        (re.compile(r"^get\s+(a\s+)?free\s+(?:quote|estimate)[!.]?$", re.IGNORECASE), "Free Quote button"),
        (
            re.compile(r"^schedule\s+(a\s+)?(?:service|appointment|consultation)[!.]?$", re.IGNORECASE),
            "Schedule Service button",
        ),
        (re.compile(r"^contact\s+us\s+(now|today)[!.]?$", re.IGNORECASE), "Contact Us button"),
        (re.compile(r"^get\s+(a\s+)?free\s+estimate[!.]?$", re.IGNORECASE), "Free Estimate button"),
        # ── Hero CTA blocks ────────────────────────────────────────────────
        (
            re.compile(r"^(?:hero|banner|cta)\s*(?:section|block|area)?$", re.IGNORECASE),
            "Hero CTA block",
        ),
        # ── Footer content ─────────────────────────────────────────────────
        (re.compile(r"^copyright\s*(?:©|©|\(c\)|\d{4})", re.IGNORECASE), "Copyright notice"),
        (re.compile(r"^©\s*\d{4}", re.IGNORECASE), "Copyright notice"),
        (re.compile(r"^all\s+rights\s+reserved[.]?$", re.IGNORECASE), "All Rights Reserved"),
        (re.compile(r"^privacy\s+policy(?:\s*\|.*)?$", re.IGNORECASE), "Privacy Policy footer"),
        (
            re.compile(r"^terms\s+(?:of\s+service|and\s+conditions)(?:\s*\|.*)?$", re.IGNORECASE),
            "Terms of Service footer",
        ),
        # ── Cookie banners ─────────────────────────────────────────────────
        (re.compile(r"^we\s+use\s+cookies", re.IGNORECASE), "Cookie banner"),
        (re.compile(r"^accept\s+all\s+cookies[!.]?$", re.IGNORECASE), "Accept Cookies button"),
        (re.compile(r"^this\s+(?:website|site)\s+uses\s+cookies", re.IGNORECASE), "Cookie notice"),
        # ── Floating / booking widgets ──────────────────────────────────────
        (re.compile(r"^(?:open|close)\s+chat[!.]?$", re.IGNORECASE), "Chat widget"),
        (re.compile(r"^chat\s+(?:with\s+us|now)[!.]?$", re.IGNORECASE), "Chat widget"),
    ]

    # Strip markdown formatting to get the bare text of a block
    _MD_FORMAT_RE = re.compile(r"[*_`#>\[\]()!|\\]")

    def sanitize(self, markdown: str) -> SanitizationResult:
        """
        Scan markdown for UI artifacts and return a cleaned copy.

        Splits on blank lines to find paragraph blocks. A block is removed only
        when its entire content (after stripping markdown syntax) matches a UI
        pattern. Multi-sentence blocks and contextual CTAs are preserved.
        """
        # Normalise line endings; split into blocks
        text = markdown.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n{2,}", text)

        removed_labels: list[str] = []
        kept_blocks: list[str] = []

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue

            # Get bare text (strip markdown syntax characters)
            bare = self._MD_FORMAT_RE.sub(" ", stripped).strip()
            # Collapse internal whitespace
            bare = re.sub(r"\s+", " ", bare)

            label = self._match_ui_pattern(bare)
            if label:
                if label not in removed_labels:
                    removed_labels.append(label)
                continue  # drop this block

            kept_blocks.append(block)

        result_md = "\n\n".join(kept_blocks).strip()
        return SanitizationResult(markdown=result_md, removed=removed_labels)

    def _match_ui_pattern(self, bare_text: str) -> str | None:
        """Return the label of the first matching UI pattern, or None."""
        for pattern, label in self._BLOCK_PATTERNS:
            if pattern.fullmatch(bare_text):
                return label
        return None

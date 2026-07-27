"""
ArticleStructureService — loads templates/article_structure.json, builds a structural
prompt scaffold for article generation, validates generated Markdown, and repairs
repairable violations before the article reaches the publisher.

Template updates:
    Edit templates/article_structure.json and restart. No code changes required.
    Increment the 'version' field so the change is visible in logs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "article_structure.json"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructureViolation:
    code: str
    description: str
    line: int | None = None
    repairable: bool = False


@dataclass
class StructureValidationResult:
    violations: list[StructureViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.violations

    @property
    def repairable_violations(self) -> list[StructureViolation]:
        return [v for v in self.violations if v.repairable]

    @property
    def fatal_violations(self) -> list[StructureViolation]:
        return [v for v in self.violations if not v.repairable]


# ── Heading parser ─────────────────────────────────────────────────────────────

# Matches ATX headings with or without trailing text (`##` alone is an empty heading).
_HEADING_RE = re.compile(r'^(#{1,6})(?:\s+(.*?))?\s*$')
_FAQ_BOLD_Q_RE = re.compile(r'^\*\*.+\?\*\*\s*$')
_IMG_PLACEHOLDER_BARE = re.compile(r'SEO_AGENT_IMAGE')


def _parse_headings(markdown: str) -> list[tuple[int, int, str]]:
    """Return [(level, 1-based line number, text), ...] for every heading in markdown."""
    result = []
    for i, line in enumerate(markdown.splitlines(), start=1):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text = (m.group(2) or "").strip()
            result.append((level, i, text))
    return result


# ── Service ───────────────────────────────────────────────────────────────────

class ArticleStructureService:
    """
    Singleton service backed by templates/article_structure.json.

    Call validate_and_repair(markdown) after article generation to enforce
    structural consistency before the content reaches the publisher.

    Call build_structure_prompt() to get the structural scaffold block
    for injection into the generation prompt.
    """

    _template: dict[str, Any] | None = None
    _template_version: str = "unloaded"

    @classmethod
    def _load(cls) -> dict[str, Any]:
        if cls._template is None:
            if not _TEMPLATE_PATH.exists():
                logger.warning(
                    "Article structure template not found at %s — structural enforcement disabled.",
                    _TEMPLATE_PATH,
                )
                cls._template = {}
            else:
                cls._template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
                cls._template_version = cls._template.get("version", "unknown")
                logger.info(
                    "Article structure template loaded (version=%s, source=%s)",
                    cls._template_version,
                    cls._template.get("canonical_source", "?"),
                )
        return cls._template

    @classmethod
    def _rules(cls) -> dict[str, Any]:
        return cls._load().get("validation_rules", {})

    @classmethod
    def _formatting(cls) -> dict[str, Any]:
        return cls._load().get("formatting", {})

    @classmethod
    def _skeleton(cls) -> dict[str, Any]:
        return cls._load().get("document_skeleton", {})

    @classmethod
    def _repair_cfg(cls) -> dict[str, Any]:
        return cls._load().get("repair_rules", {})

    # ── Prompt building ───────────────────────────────────────────────────────

    @classmethod
    def build_structure_prompt(cls) -> str:
        """
        Build the REQUIRED DOCUMENT STRUCTURE block for injection into the
        article generation prompt. Returns an empty string if the template
        is not loaded (structural enforcement is disabled).
        """
        t = cls._load()
        if not t:
            return ""

        skel = cls._skeleton()
        fmt = cls._formatting()
        rules = cls._rules()

        intro = skel.get("intro", {})
        h2_cfg = skel.get("h2_sections", {})
        faq = skel.get("faq", {})
        closing = skel.get("closing", {})

        faq_heading = rules.get("faq_heading_exact_match", "Frequently Asked Questions")
        callout = fmt.get("callout_prefix", "> ⚠️ **Important:**")
        img_ph = fmt.get("image_placeholder", "<!-- SEO_AGENT_IMAGE: {id} -->")
        max_depth = fmt.get("max_heading_depth", 2)
        h2_min = rules.get("h2_min", 4)
        h2_max = rules.get("h2_max", 7)
        faq_count = faq.get("count", "4-6")
        faq_q_fmt = faq.get("question_format", "**Question ending with a question mark?**")
        faq_a_fmt = faq.get("answer_format", "Answer paragraph.")
        img_positions = h2_cfg.get("image_after_positions", [1, 2, 4])
        img_position_note = h2_cfg.get("image_position_note", "")

        sep = "──────────────────────────────────────────"
        lines: list[str] = [
            sep,
            "REQUIRED DOCUMENT STRUCTURE",
            sep,
            f"Every article MUST follow this exact skeleton.",
            f"Topic and content change every article. Structure never changes.",
            f"Do NOT use heading levels deeper than H{max_depth}.",
            f"Do NOT omit the FAQ section.",
            f"Do NOT add unnamed or extra sections.",
            "",
            f"── SKELETON ({h2_min}–{h2_max} H2 body sections, then FAQ) ──",
            "",
            "# [H1 article title — keyword-rich, ≤70 chars]",
            "",
        ]

        # Intro
        p_count = intro.get("paragraph_count", "1-2")
        intro_note = intro.get("note", "Hook sentence. No heading before this block.")
        lines += [
            f"[INTRO — {p_count} paragraph(s), no heading]",
            f"  {intro_note}",
            "",
        ]

        # Body sections
        elems = h2_cfg.get("element_sequence", [])
        lines += [
            f"## [H2 section heading — repeat {h2_min} to {h2_max} times]",
            "",
        ]
        for elem in elems:
            if ":" in elem:
                kind, spec = elem.split(":", 1)
                spec = spec.strip()
            else:
                kind, spec = elem, ""
            if kind == "paragraph":
                lines.append(f"  [Body paragraphs — {spec}]")
            elif kind == "list":
                lines.append(f"  [List — {spec}]")
            elif kind == "callout":
                lines.append(f"  [{callout} ... — {spec}]")
            elif kind == "thesis_closer":
                lines.append(f"  [{spec}]")

        lines.append("")

        # Image positions
        for pos in sorted(img_positions):
            ex = img_ph.replace("{id}", f"img_00{pos + 1}")
            lines.append(f"  {ex}  ← after section {pos}")
        if img_position_note:
            lines.append(f"  ({img_position_note})")
        lines.append("")

        # FAQ
        lines += [
            f"## {faq_heading}",
            "",
            f"  {faq_q_fmt}",
            "",
            f"  {faq_a_fmt}",
            "",
            f"  [Repeat for {faq_count} questions. FAQ note: {faq.get('note', '')}]",
            "",
        ]

        # Closing
        closing_note = closing.get("note", "One closing paragraph, no heading.")
        lines += [
            "[CLOSING — no heading]",
            f"  {closing_note}",
            "",
        ]

        # Formatting rules
        lines += [
            "── FORMATTING RULES (enforced by validator) ──",
            f"  • Callouts must start with exactly: {callout}",
            f"  • FAQ questions must be bold with double asterisks: **Question text?**",
            f"  • Image placeholders must be exactly: {img_ph}",
            "  • Bold key terms in body paragraphs using **double asterisks**",
            "  • Never use bold text as a substitute for a heading",
            sep,
        ]

        return "\n".join(lines)

    # ── Validation ────────────────────────────────────────────────────────────

    @classmethod
    def validate(cls, markdown: str) -> StructureValidationResult:
        """
        Validate markdown against the structural template rules.
        Returns a StructureValidationResult listing all violations and warnings.
        """
        result = StructureValidationResult()
        t = cls._load()
        if not t:
            return result

        rules = cls._rules()
        lines_raw = markdown.splitlines()
        headings = _parse_headings(markdown)

        h1s = [(lv, ln, tx) for lv, ln, tx in headings if lv == 1]
        h2s = [(lv, ln, tx) for lv, ln, tx in headings if lv == 2]
        h3s = [(lv, ln, tx) for lv, ln, tx in headings if lv == 3]

        # ── H1 count ──────────────────────────────────────────────────────────
        h1_max = rules.get("h1_max", 1)
        if len(h1s) > h1_max:
            result.violations.append(StructureViolation(
                code="H1_COUNT",
                description=f"Found {len(h1s)} H1 headings; expected at most {h1_max}.",
                line=h1s[h1_max][1],
                repairable=False,
            ))

        # ── H2 count ──────────────────────────────────────────────────────────
        h2_min = rules.get("h2_min", 4)
        h2_max_cnt = rules.get("h2_max", 8)
        if len(h2s) < h2_min:
            result.violations.append(StructureViolation(
                code="H2_TOO_FEW",
                description=f"Found {len(h2s)} H2 sections; template requires at least {h2_min}.",
                repairable=False,
            ))
        elif len(h2s) > h2_max_cnt:
            result.warnings.append(
                f"Found {len(h2s)} H2 sections; template recommends at most {h2_max_cnt}."
            )

        # ── H3 count ──────────────────────────────────────────────────────────
        h3_max = rules.get("h3_max", 4)
        if len(h3s) > h3_max:
            result.warnings.append(
                f"Found {len(h3s)} H3 headings; template recommends at most {h3_max}."
            )

        # ── Empty headings ─────────────────────────────────────────────────────
        if rules.get("no_empty_headings", True):
            for lv, ln, tx in headings:
                if not tx:
                    result.violations.append(StructureViolation(
                        code="EMPTY_HEADING",
                        description=f"Empty H{lv} at line {ln}.",
                        line=ln,
                        repairable=True,
                    ))

        # ── Duplicate headings ─────────────────────────────────────────────────
        if rules.get("no_duplicate_headings", True):
            seen_headings: dict[str, int] = {}
            for lv, ln, tx in headings:
                key = tx.lower().strip()
                if key in seen_headings:
                    result.violations.append(StructureViolation(
                        code="DUPLICATE_HEADING",
                        description=(
                            f"Duplicate heading '{tx}' at line {ln} "
                            f"(first seen at line {seen_headings[key]})."
                        ),
                        line=ln,
                        repairable=True,
                    ))
                else:
                    seen_headings[key] = ln

        # ── No skipped heading levels ──────────────────────────────────────────
        if rules.get("no_skipped_heading_levels", True):
            for i in range(1, len(headings)):
                prev_lv, _, _ = headings[i - 1]
                curr_lv, curr_ln, _ = headings[i]
                if curr_lv > prev_lv + 1:
                    result.violations.append(StructureViolation(
                        code="HEADING_LEVEL_SKIP",
                        description=(
                            f"Heading jumps from H{prev_lv} to H{curr_lv} at line {curr_ln} "
                            f"(skips H{prev_lv + 1})."
                        ),
                        line=curr_ln,
                        repairable=False,
                    ))

        # ── FAQ section ────────────────────────────────────────────────────────
        faq_heading = rules.get("faq_heading_exact_match", "Frequently Asked Questions")
        faq_h2_matches = [(lv, ln, tx) for lv, ln, tx in h2s if tx.strip() == faq_heading]

        if not faq_h2_matches:
            result.violations.append(StructureViolation(
                code="FAQ_MISSING",
                description=f"No H2 section heading matching '{faq_heading}' found.",
                repairable=False,
            ))
        else:
            faq_start_line = faq_h2_matches[0][1]  # 1-based

            faq_min = rules.get("faq_min_questions", 4)
            faq_max_q = rules.get("faq_max_questions", 7)

            # Find bold question lines inside the FAQ block.
            # A bold line in the FAQ block: starts with ** and ends with **
            faq_lines = lines_raw[faq_start_line:]  # 0-based after the heading
            bold_lines: list[tuple[int, str]] = []  # (absolute 1-based line, text)
            for offset, line in enumerate(faq_lines):
                stripped = line.strip()
                if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                    absolute_ln = faq_start_line + offset + 1
                    bold_lines.append((absolute_ln, stripped))

            q_count = len(bold_lines)

            if q_count < faq_min:
                result.violations.append(StructureViolation(
                    code="FAQ_TOO_FEW_QUESTIONS",
                    description=(
                        f"FAQ section has {q_count} question(s); "
                        f"template requires at least {faq_min}."
                    ),
                    line=faq_start_line,
                    repairable=False,
                ))
            elif q_count > faq_max_q:
                result.warnings.append(
                    f"FAQ has {q_count} questions; template recommends at most {faq_max_q}."
                )

            # Each bold line must end with ?** (question mark before closing **)
            for abs_ln, text in bold_lines:
                if not _FAQ_BOLD_Q_RE.match(text):
                    result.violations.append(StructureViolation(
                        code="FAQ_QUESTION_FORMAT",
                        description=(
                            f"FAQ question at line {abs_ln} does not end with '?': {text!r}"
                        ),
                        line=abs_ln,
                        repairable=True,
                    ))

        # ── Image placeholder format ───────────────────────────────────────────
        img_re_str = rules.get("image_placeholder_regex", "")
        if img_re_str:
            img_re = re.compile(img_re_str)
            for i, line in enumerate(lines_raw, start=1):
                if _IMG_PLACEHOLDER_BARE.search(line) and not img_re.search(line):
                    result.violations.append(StructureViolation(
                        code="MALFORMED_IMAGE_PLACEHOLDER",
                        description=f"Malformed image placeholder at line {i}: {line.strip()!r}",
                        line=i,
                        repairable=False,
                    ))

        return result

    # ── Repair ────────────────────────────────────────────────────────────────

    @classmethod
    def repair(cls, markdown: str, result: StructureValidationResult) -> tuple[str, int]:
        """
        Apply repairs for all repairable violations in *result*.

        Returns (repaired_markdown, repairs_applied_count).
        Does not attempt to fix fatal violations.
        """
        cfg = cls._repair_cfg()
        if not cfg or not result.repairable_violations:
            return markdown, 0

        lines = markdown.splitlines()
        repairs = 0

        # Collect line indices (0-based) to remove
        lines_to_remove: set[int] = set()

        # ── Remove empty headings ─────────────────────────────────────────────
        if cfg.get("remove_empty_headings", True):
            targets = {
                v.line - 1
                for v in result.repairable_violations
                if v.code == "EMPTY_HEADING" and v.line is not None
            }
            if targets:
                lines_to_remove |= targets
                repairs += len(targets)
                logger.info("Structure repair: removing %d empty heading(s).", len(targets))

        # ── Deduplicate headings ──────────────────────────────────────────────
        if cfg.get("deduplicate_headings") == "keep_first":
            targets = {
                v.line - 1
                for v in result.repairable_violations
                if v.code == "DUPLICATE_HEADING" and v.line is not None
            }
            if targets:
                lines_to_remove |= targets
                repairs += len(targets)
                logger.info("Structure repair: removing %d duplicate heading(s).", len(targets))

        # Apply line removals
        if lines_to_remove:
            lines = [ln for i, ln in enumerate(lines) if i not in lines_to_remove]

        # ── Normalize FAQ question bold format ────────────────────────────────
        if cfg.get("normalize_faq_question_bold", True):
            fmt_violations = {
                v.line - 1
                for v in result.repairable_violations
                if v.code == "FAQ_QUESTION_FORMAT" and v.line is not None
            }
            # Adjust indices after line removals
            adjusted = set()
            for idx in fmt_violations:
                removed_before = sum(1 for r in lines_to_remove if r < idx)
                adjusted.add(idx - removed_before)

            for idx in adjusted:
                if 0 <= idx < len(lines):
                    text = lines[idx].strip()
                    # Ensure it ends with ?**
                    inner = text[2:-2]  # strip leading ** and trailing **
                    if not inner.rstrip().endswith("?"):
                        inner = inner.rstrip().rstrip("?") + "?"
                    lines[idx] = f"**{inner}**"
                    repairs += 1
            if adjusted:
                logger.info("Structure repair: normalized %d FAQ question format(s).", len(adjusted))

        # ── Strip trailing whitespace ─────────────────────────────────────────
        if cfg.get("strip_trailing_whitespace", True):
            original = list(lines)
            lines = [ln.rstrip() for ln in lines]
            if lines != original:
                repairs += 1

        return "\n".join(lines), repairs

    # ── Public entry point ────────────────────────────────────────────────────

    @classmethod
    def validate_and_repair(cls, markdown: str) -> str:
        """
        Validate *markdown* against the structural template.

        Logs every violation and warning.
        Attempts repair for repairable violations.
        Returns the (possibly repaired) markdown.

        Never raises — structural issues are logged but do not block the pipeline.
        """
        t = cls._load()
        if not t:
            return markdown

        result = cls.validate(markdown)

        if result.is_valid and not result.warnings:
            logger.debug(
                "Article structure valid (template v%s). No violations.",
                cls._template_version,
            )
            return markdown

        # Log warnings
        for w in result.warnings:
            logger.warning("Structure: %s", w)

        # Log violations
        if result.violations:
            logger.warning(
                "Article structure: %d violation(s) found (template v%s).",
                len(result.violations),
                cls._template_version,
            )
            for v in result.violations:
                loc = f" (line {v.line})" if v.line else ""
                tag = "[repairable]" if v.repairable else "[FATAL — cannot auto-repair]"
                logger.warning("  %s [%s]%s %s", tag, v.code, loc, v.description)

        # Attempt repair
        if result.repairable_violations:
            repaired, count = cls.repair(markdown, result)
            if count:
                # Re-validate to confirm repairs worked
                post = cls.validate(repaired)
                remaining = len(post.fatal_violations)
                if remaining:
                    logger.warning(
                        "Structure repair applied %d fix(es); %d fatal violation(s) remain.",
                        count, remaining,
                    )
                else:
                    logger.info(
                        "Structure repair applied %d fix(es). All violations resolved.",
                        count,
                    )
                return repaired

        return markdown

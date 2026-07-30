"""
WritingAuditService — diagnostic analysis of article prose quality.

Runs before QA to surface writing patterns correlated with low Human Writing scores:
  • Paragraph opener diversity (grammatical pattern variety)
  • Sentence length standard deviation (rhythm uniformity)
  • Paragraph template uniformity (structural repetition)
  • Transition word frequency and diversity
  • Voice balance (observational / explanatory / instructional / definitional)

Diagnostic only — never modifies the article. Call log_audit() from the pipeline.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Opener classifier ─────────────────────────────────────────────────────────

_ARTICLE_WORDS = frozenset(["the", "a", "an", "this", "these", "that", "those"])
_QUANTIFIER_WORDS = frozenset([
    "most", "many", "some", "few", "all", "each", "every", "no",
    "several", "both", "any", "either", "neither",
])
_SUBORDINATE_WORDS = frozenset([
    "when", "if", "although", "while", "since", "because", "unless",
    "until", "whether", "though", "even", "once", "wherever", "whenever",
])
_PREPOSITIONAL_WORDS = frozenset([
    "in", "on", "at", "for", "by", "from", "with", "under", "over",
    "during", "between", "without", "within", "across", "after", "before",
    "beyond", "among", "around", "despite", "through", "toward",
])
_DEICTIC_WORDS = frozenset(["here", "there", "now", "then"])
_NUMBER_RE = re.compile(r"^\d")


def _classify_opener(first_word: str) -> str:
    w = re.sub(r"[^a-zA-Z']", "", first_word).lower()
    if w.endswith("'s"):
        w = w[:-2]
    if w in ("yes", "no"):
        return "direct-answer"
    if _NUMBER_RE.match(first_word):
        return "number-first"
    if w in _ARTICLE_WORDS:
        return "article+noun"
    if w in _QUANTIFIER_WORDS:
        return "quantifier"
    if w in _SUBORDINATE_WORDS:
        return "subordinate"
    if w in _PREPOSITIONAL_WORDS:
        return "prepositional"
    if w in _DEICTIC_WORDS:
        return "deictic"
    if w.endswith("ing"):
        return "participial"
    if w.endswith("ed") and len(w) > 4:
        return "past-participle"
    return "subject-first"


# ── Transition list ───────────────────────────────────────────────────────────

_KNOWN_TRANSITIONS = [
    "however", "additionally", "furthermore", "moreover", "that said",
    "in addition", "on the other hand", "nevertheless", "nonetheless",
    "consequently", "therefore", "thus", "hence", "accordingly",
    "meanwhile", "subsequently", "conversely", "alternatively",
    "similarly", "likewise", "in contrast", "in other words",
    "for example", "for instance", "specifically", "notably",
    "importantly", "significantly", "ultimately", "essentially",
    "generally speaking", "broadly speaking", "on balance",
]

# ── Voice classifiers ─────────────────────────────────────────────────────────

_INSTRUCTIONAL_RE = re.compile(
    r"^(?:check|test|watch|listen|measure|inspect|look|disconnect|pull|push|"
    r"avoid|never|don't|do not|always|make sure|ensure|confirm|verify|"
    r"replace|remove|tighten|adjust|call|contact|hire|schedule|run|try)\b",
    re.IGNORECASE,
)
_EXPLANATORY_RE = re.compile(
    r"\b(?:because|since|due to|as a result|this means|which means|"
    r"explains?\s+why|the reason|causes?|results?\s+in|leads?\s+to|"
    r"stems?\s+from|follows?\s+from)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_RE = re.compile(
    r"\b(?:is a\b|are a\b|refers?\s+to|defined as|known as|called |"
    r"consists?\s+of|made of|designed to|intended to|refers?\s+to)\b",
    re.IGNORECASE,
)
_OBSERVATIONAL_RE = re.compile(
    r"\b(?:fails?|breaks?|snaps?|wears?|corrodes?|rusts?|loosens?|"
    r"tightens?|binds?|creaks?|wobbles?|hesitates?|strains?|"
    r"sticks?|jerks?|appears?|shows?|sounds?|moves?|drops?|"
    r"develops?|accumulates?|builds?\s+up|begins?\s+to|starts?\s+to)\b",
    re.IGNORECASE,
)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class OpenerDiversity:
    pattern_counts: dict[str, int]
    diversity_score: float      # unique patterns / total paragraphs (0–1)
    dominant_pattern: str | None
    dominant_pct: float
    flag: bool                  # True when dominant_pct ≥ 0.30


@dataclass
class SentenceRhythm:
    avg_length: float
    std_dev: float
    shortest: int
    longest: int
    flag: bool                  # True when std_dev < 10.0


@dataclass
class ParagraphRhythm:
    sentence_count_dist: dict[int, int]
    avg_sentences: float
    uniformity_score: float     # most-common-count / total paragraphs (0–1)
    flag: bool                  # True when uniformity_score ≥ 0.65


@dataclass
class TransitionAnalysis:
    frequency: dict[str, int]
    total_transitions: int
    diversity_score: float      # unique / total (0–1; 1 = all different)
    flag: bool                  # True when total > 3 and diversity_score < 0.40


@dataclass
class VoiceBalance:
    observational_pct: float
    explanatory_pct: float
    instructional_pct: float
    definitional_pct: float
    dominant_voice: str
    flag: bool                  # True when dominant voice > 60%


@dataclass
class WritingAuditReport:
    paragraph_count: int
    sentence_count: int
    openers: OpenerDiversity
    rhythm: SentenceRhythm
    para_rhythm: ParagraphRhythm
    transitions: TransitionAnalysis
    voice: VoiceBalance
    overall_risk: str           # "low" | "medium" | "high"
    flags: list[str] = field(default_factory=list)


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_paragraphs(markdown: str) -> list[str]:
    """Return prose paragraphs — skips headings, images, tables, blockquotes."""
    result = []
    for block in re.split(r"\n{2,}", markdown):
        block = block.strip()
        if not block:
            continue
        if block.startswith(("#", "<!--", ">", "|", "```", "---", "===")):
            continue
        # Markdown list items (- / * / numbered) — skip
        if re.match(r"^[-*•]\s", block) or re.match(r"^\d+\.\s", block):
            continue
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", block)
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
        clean = re.sub(r"<!--.*?-->", "", clean, flags=re.DOTALL)
        clean = clean.strip()
        if len(clean.split()) >= 5:
            result.append(clean)
    return result


def _extract_sentences(paragraphs: list[str]) -> list[str]:
    sentences = []
    for p in paragraphs:
        raw = re.split(r"(?<=[.!?])\s+", p)
        sentences.extend(s.strip() for s in raw if len(s.split()) >= 3)
    return sentences


# ── Analysis functions ────────────────────────────────────────────────────────

def _analyze_openers(paragraphs: list[str]) -> OpenerDiversity:
    patterns: list[str] = []
    for p in paragraphs:
        words = p.split()
        if words:
            patterns.append(_classify_opener(words[0]))
    if not patterns:
        return OpenerDiversity({}, 0.0, None, 0.0, False)
    cnt = Counter(patterns)
    total = len(patterns)
    diversity = len(cnt) / total
    dominant, dom_count = cnt.most_common(1)[0]
    dom_pct = dom_count / total
    return OpenerDiversity(
        pattern_counts=dict(cnt),
        diversity_score=round(diversity, 3),
        dominant_pattern=dominant,
        dominant_pct=round(dom_pct, 3),
        flag=dom_pct >= 0.30,
    )


def _analyze_rhythm(sentences: list[str]) -> SentenceRhythm:
    if not sentences:
        return SentenceRhythm(0.0, 0.0, 0, 0, False)
    lengths = [len(s.split()) for s in sentences]
    avg = statistics.mean(lengths)
    std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    return SentenceRhythm(
        avg_length=round(avg, 1),
        std_dev=round(std, 1),
        shortest=min(lengths),
        longest=max(lengths),
        flag=std < 10.0,
    )


def _analyze_para_rhythm(paragraphs: list[str]) -> ParagraphRhythm:
    if not paragraphs:
        return ParagraphRhythm({}, 0.0, 0.0, False)
    sent_counts: list[int] = []
    for p in paragraphs:
        sents = re.split(r"(?<=[.!?])\s+", p)
        count = len([s for s in sents if len(s.split()) >= 3])
        sent_counts.append(max(1, count))
    avg = statistics.mean(sent_counts)
    cnt = Counter(sent_counts)
    uniformity = cnt.most_common(1)[0][1] / len(sent_counts)
    return ParagraphRhythm(
        sentence_count_dist=dict(sorted(cnt.items())),
        avg_sentences=round(avg, 1),
        uniformity_score=round(uniformity, 3),
        flag=uniformity >= 0.65,
    )


def _analyze_transitions(markdown: str) -> TransitionAnalysis:
    text_lower = markdown.lower()
    freq: dict[str, int] = {}
    for t in _KNOWN_TRANSITIONS:
        pattern = r"\b" + re.escape(t) + r"\b"
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            freq[t] = count
    total = sum(freq.values())
    diversity = len(freq) / total if total > 0 else 1.0
    return TransitionAnalysis(
        frequency=freq,
        total_transitions=total,
        diversity_score=round(diversity, 3),
        flag=total > 3 and diversity < 0.40,
    )


def _analyze_voice(sentences: list[str]) -> VoiceBalance:
    if not sentences:
        return VoiceBalance(0.0, 0.0, 0.0, 0.0, "unknown", False)
    obs = exp = ins = dfn = 0
    for s in sentences:
        if _INSTRUCTIONAL_RE.match(s):
            ins += 1
        elif _EXPLANATORY_RE.search(s):
            exp += 1
        elif _DEFINITIONAL_RE.search(s):
            dfn += 1
        elif _OBSERVATIONAL_RE.search(s):
            obs += 1
    n = len(sentences)
    obs_p = obs / n
    exp_p = exp / n
    ins_p = ins / n
    dfn_p = dfn / n
    voices = {
        "observational": obs_p,
        "explanatory": exp_p,
        "instructional": ins_p,
        "definitional": dfn_p,
    }
    dominant = max(voices, key=voices.__getitem__)
    return VoiceBalance(
        observational_pct=round(obs_p, 3),
        explanatory_pct=round(exp_p, 3),
        instructional_pct=round(ins_p, 3),
        definitional_pct=round(dfn_p, 3),
        dominant_voice=dominant,
        flag=voices[dominant] > 0.60,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def audit(markdown: str) -> WritingAuditReport:
    """Analyse markdown prose and return a structured WritingAuditReport."""
    paragraphs = _extract_paragraphs(markdown)
    sentences = _extract_sentences(paragraphs)

    openers = _analyze_openers(paragraphs)
    rhythm = _analyze_rhythm(sentences)
    para_rhythm = _analyze_para_rhythm(paragraphs)
    transitions = _analyze_transitions(markdown)
    voice = _analyze_voice(sentences)

    flags: list[str] = []
    if openers.flag:
        flags.append(
            f"opener '{openers.dominant_pattern}' opens "
            f"{openers.dominant_pct*100:.0f}% of paragraphs — "
            f"threshold 30%"
        )
    if rhythm.flag:
        flags.append(
            f"sentence rhythm: std-dev {rhythm.std_dev} (avg {rhythm.avg_length}w) — "
            f"uniform cadence; target std-dev ≥10"
        )
    if para_rhythm.flag:
        flags.append(
            f"paragraph structure: {para_rhythm.uniformity_score*100:.0f}% share "
            f"the same sentence count — template visible"
        )
    if transitions.flag:
        flags.append(
            f"transitions: diversity score {transitions.diversity_score:.2f} — "
            f"same transitions repeated; target ≥0.40"
        )
    if voice.flag:
        dom_pct = getattr(voice, voice.dominant_voice + "_pct")
        flags.append(
            f"voice: '{voice.dominant_voice}' at {dom_pct*100:.0f}% dominates — "
            f"threshold 60%"
        )

    risk = "high" if len(flags) >= 3 else ("medium" if flags else "low")

    return WritingAuditReport(
        paragraph_count=len(paragraphs),
        sentence_count=len(sentences),
        openers=openers,
        rhythm=rhythm,
        para_rhythm=para_rhythm,
        transitions=transitions,
        voice=voice,
        overall_risk=risk,
        flags=flags,
    )


def format_report(report: WritingAuditReport) -> str:
    """Return a human-readable audit report string."""
    sep = "─" * 52
    lines = [
        "WRITING DIVERSITY AUDIT",
        "═" * 52,
        f"  paragraphs: {report.paragraph_count}   sentences: {report.sentence_count}",
        "",
        f"  PARAGRAPH OPENER DIVERSITY  "
        + ("⚠ FLAG" if report.openers.flag else "✓"),
        sep,
        f"  diversity score: {report.openers.diversity_score:.2f}  "
        f"(unique pattern types / paragraph count)",
    ]
    total_p = max(report.paragraph_count, 1)
    for pat, n in sorted(report.openers.pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * n // total_p
        bar = "█" * n + "░" * max(0, 8 - n)
        lines.append(f"    {pat:<22} {n:>2}×  {pct:>3}%  {bar}")
    if report.openers.flag:
        lines.append(
            f"  ⚠ '{report.openers.dominant_pattern}' exceeds 30% — "
            f"{report.openers.dominant_pct*100:.0f}% of paragraph openers"
        )
    lines.append("")

    lines += [
        f"  SENTENCE LENGTH RHYTHM  " + ("⚠ FLAG" if report.rhythm.flag else "✓"),
        sep,
        f"  avg: {report.rhythm.avg_length}w   std-dev: {report.rhythm.std_dev}   "
        f"range: {report.rhythm.shortest}–{report.rhythm.longest}w",
    ]
    if report.rhythm.flag:
        lines.append(
            f"  ⚠ std-dev {report.rhythm.std_dev} below 10 — sentences are rhythmically uniform"
        )
    lines.append("")

    lines += [
        f"  PARAGRAPH RHYTHM  " + ("⚠ FLAG" if report.para_rhythm.flag else "✓"),
        sep,
        f"  avg sentences/paragraph: {report.para_rhythm.avg_sentences}   "
        f"uniformity: {report.para_rhythm.uniformity_score:.0%}",
        "  distribution: "
        + "  ".join(
            f"{k}sent×{v}"
            for k, v in sorted(report.para_rhythm.sentence_count_dist.items())
        ),
    ]
    if report.para_rhythm.flag:
        lines.append("  ⚠ paragraph lengths cluster — structural template visible")
    lines.append("")

    lines += [
        f"  TRANSITION ANALYSIS  " + ("⚠ FLAG" if report.transitions.flag else "✓"),
        sep,
    ]
    if report.transitions.frequency:
        top = sorted(report.transitions.frequency.items(), key=lambda x: -x[1])[:6]
        lines.append("  detected: " + "   ".join(f'"{t}" {n}×' for t, n in top))
        lines.append(f"  diversity score: {report.transitions.diversity_score:.2f}")
    else:
        lines.append("  no tracked transitions detected ✓")
    if report.transitions.flag:
        lines.append("  ⚠ same transitions repeated — diversity score below 0.40")
    lines.append("")

    def _bar(pct: float) -> str:
        filled = round(pct * 10)
        return "█" * filled + "░" * (10 - filled)

    lines += [
        f"  VOICE BALANCE  " + ("⚠ FLAG" if report.voice.flag else "✓"),
        sep,
        f"  observational:  {report.voice.observational_pct*100:>4.0f}%  "
        f"{_bar(report.voice.observational_pct)}",
        f"  explanatory:    {report.voice.explanatory_pct*100:>4.0f}%  "
        f"{_bar(report.voice.explanatory_pct)}",
        f"  instructional:  {report.voice.instructional_pct*100:>4.0f}%  "
        f"{_bar(report.voice.instructional_pct)}",
        f"  definitional:   {report.voice.definitional_pct*100:>4.0f}%  "
        f"{_bar(report.voice.definitional_pct)}",
    ]
    if report.voice.flag:
        dom_pct = getattr(report.voice, report.voice.dominant_voice + "_pct")
        lines.append(
            f"  ⚠ '{report.voice.dominant_voice}' at {dom_pct*100:.0f}% dominates — "
            "aim for more balance across voice types"
        )
    lines.append("")

    risk_sym = {"low": "✓", "medium": "⚠", "high": "✗"}[report.overall_risk]
    lines += [
        "═" * 52,
        f"  OVERALL RISK: {report.overall_risk.upper()}  {risk_sym}",
    ]
    if report.flags:
        lines.append("  Active flags:")
        for f in report.flags:
            lines.append(f"    → {f}")
    lines.append("═" * 52)
    return "\n".join(lines)


def log_audit(markdown: str, label: str = "") -> WritingAuditReport:
    """Run audit, emit formatted report to the logger, and return the result."""
    report = audit(markdown)
    header = "PRE-QA WRITING AUDIT" + (f" — {label}" if label else "")
    logger.info("%s\n%s", header, format_report(report))
    if report.flags:
        logger.warning(
            "Writing audit: %d flag(s)  risk=%s  (diagnostic only — article proceeds to QA)",
            len(report.flags),
            report.overall_risk,
        )
    return report

"""
Editorial placement engine for inline images in Markdown articles.

Computes optimal inline image positions according to professional editorial rules:
- Images are distributed naturally throughout the article.
- No consecutive images; minimum prose spacing enforced.
- Images never appear in FAQ, CTA, tables, lists, blockquotes, or code blocks.
- First image appears in the introduction (after ≥ 3 intro paragraphs) or in the
  first eligible H2 section.
- Subsequent images placed at an adaptive depth (short section → near top;
  long section → ~37% in) inside their assigned sections.
- Global layout optimization selects positions that minimize a global cost function.
- Editorial layout score computed after every placement.

Entry point:
    result = EditorialPlacementEngine(embed_fn=...).place(markdown, images)
    updated_markdown = result.markdown
    logger.info(result.score.format_report())

embed_fn is an optional callable [[str, ...] → [embedding, ...]] supplied by the
caller (e.g. from make_openai_embed_fn()).  When None, falls back to stemmed
lexical similarity.

The Publisher calls this engine once, before markdown → HTML conversion.
_inject_images() then replaces the markers with <figure> HTML as usual.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product as _iproduct
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from models.image_request import ImageRequest
    from models.media import ImageMetadata

logger = logging.getLogger(__name__)


# ── Block model ───────────────────────────────────────────────────────────────

class BlockType(Enum):
    H1         = auto()
    H2         = auto()
    H3         = auto()
    PARAGRAPH  = auto()
    TABLE      = auto()
    UL         = auto()
    OL         = auto()
    BLOCKQUOTE = auto()
    CODE       = auto()
    HR         = auto()
    BLANK      = auto()
    MARKER     = auto()   # <!-- SEO_AGENT_IMAGE: id -->


# Strict: never place an image immediately before or after a TABLE (Rule 9).
_STRICT_DANGER = frozenset({BlockType.TABLE})

# Soft: prefer not to place images adjacent to these; allowed as fallback when no
# strictly-safe position exists in the section (UL/OL/BLOCKQUOTE/CODE are common
# in technical articles and would otherwise make sections ineligible entirely).
_SOFT_DANGER = frozenset({
    BlockType.TABLE,
    BlockType.UL,
    BlockType.OL,
    BlockType.BLOCKQUOTE,
    BlockType.CODE,
})

# ── Semantic similarity helpers ────────────────────────────────────────────────

# Common English function words that carry no content signal in short headings.
_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'your', 'you', 'it', 'its',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'any', 'some', 'more',
    'most', 'other', 'than', 'then', 'so', 'if', 'as', 'up', 'out',
    'into', 'through', 'during', 'before', 'after', 'about', 'same',
    'much', 'many', 'such', 'own', 'not', 'no', 'nor', 'yet', 'both',
    'either', 'neither', 'very', 'just', 'also', 'only', 'even',
})

# Single-pass suffix list (longest to shortest) for lightweight stemming.
# Applied once per token; trailing 'e' stripped separately after suffix removal.
_SUFFIXES = (
    'ment', 'ness', 'tion', 'sion', 'ance', 'ence', 'ical',
    'ing', 'ies', 'ize', 'ful', 'ive', 'ous', 'ist', 'ism',
    'er', 'ed', 'ly', 'al', 'ic', 'le', 'en',
    'es', 's',
)


# ── Public output types ───────────────────────────────────────────────────────

@dataclass
class EditorialScore:
    """
    Multi-dimensional layout quality score produced by the placement engine.

    Saved alongside the article so editors can review placement decisions at a
    glance without re-running the engine.
    """
    overall:          int    # 0–100 weighted average
    distribution:     int    # 0–100 evenness of image spread across the document
    spacing:          int    # 0–100 prose gap between consecutive images
    section_matching: int    # 0–100 semantic confidence of image→section assignments
    visual_rhythm:    int    # 0–100 global layout cost (lower cost → higher score)
    layout_cost:      float  # raw optimizer cost (diagnostic; lower is better)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> EditorialScore:
        """Returned when there are no images to place."""
        return cls(
            overall=100, distribution=100, spacing=100,
            section_matching=100, visual_rhythm=100,
            layout_cost=0.0, warnings=[],
        )

    def format_report(self) -> str:
        lines = [
            "Editorial Layout",
            "",
            f"  Overall:          {self.overall} / 100",
            "",
            f"  Distribution:     {self.distribution}",
            f"  Spacing:          {self.spacing}",
            f"  Section Matching: {self.section_matching}",
            f"  Visual Rhythm:    {self.visual_rhythm}",
        ]
        if self.warnings:
            lines += ["", "  Warnings:"]
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        else:
            lines += ["", "  Warnings: None"]
        return "\n".join(lines)


@dataclass
class PlacementResult:
    """Return value from EditorialPlacementEngine.place()."""
    markdown: str
    score:    EditorialScore


# ── Module-level helpers ──────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors, clamped to [−1, 1]."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def make_openai_embed_fn() -> "Callable[[list[str]], list[list[float]]] | None":
    """
    Return an OpenAI text-embedding-3-small function, or None if unavailable.

    The returned callable accepts a list of strings and returns a list of
    embedding vectors (one per input string), suitable for cosine similarity.
    Falls back to None when the openai package is not installed or the API key
    is missing, so the engine degrades gracefully to lexical matching.
    """
    try:
        import openai  # local import — module loads fine without openai installed
        client = openai.OpenAI()

        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [e.embedding for e in resp.data]

        return embed
    except Exception:
        return None


# ── Block and section data models ─────────────────────────────────────────────

@dataclass
class Block:
    type: BlockType
    text: str   # raw markdown (original lines joined with \n)


@dataclass
class _Section:
    """A contiguous block of article content under one H2 heading (or the preamble)."""
    heading_text:  str          # raw heading text, stripped of ## prefix
    heading_block: int | None   # flat-list index of the H2 block; None for preamble
    start:         int          # flat-list index of first block inside this section
    end:           int          # flat-list index of the first block in the NEXT section
    is_faq: bool = False
    is_cta: bool = False

    @property
    def is_preamble(self) -> bool:
        return self.heading_block is None


# ── Engine ────────────────────────────────────────────────────────────────────

class EditorialPlacementEngine:
    """
    Computes editorially optimal inline image positions in a Markdown article.

    Behaves like a magazine layout editor, not a markdown formatter.  Every
    image placement decision is made to maximize readability, semantic relevance,
    and visual balance across the entire article.

    Improvements over a naive formatter:
      1. True semantic section matching — OpenAI embeddings when available (blend
         with lexical as tiebreaker); lexical-stemmed fallback otherwise.
      2. Global layout optimization — exhaustive cost-function minimization over
         all candidate position combinations; greedy fallback for large articles.
      3. Adaptive insertion depth — rule-based per section length, not fixed 35%.
      4. Global rhythm validation — belt-and-suspenders spacing guarantee.
      5. Editorial layout score — returned with every placement.

    Usage:
        embed_fn = make_openai_embed_fn()           # None → lexical mode
        result   = EditorialPlacementEngine(embed_fn=embed_fn).place(md, images)
        updated_markdown = result.markdown
        logger.info(result.score.format_report())
    """

    # ── Editorial parameters ──────────────────────────────────────────────────
    _MIN_INTRO_PARAS    = 3        # preamble must have ≥ this many prose blocks
    _MIN_SPACING        = 2        # minimum prose blocks between any two images
    _SECTION_MIN_PROSE  = 2        # section must have ≥ this many prose blocks
    _SEMANTIC_THRESHOLD = 0.12     # minimum lexical score for semantic assignment
    _EMBED_THRESHOLD    = 0.25     # minimum combined score when using embeddings
    _MAX_CANDIDATES     = 8        # max candidate positions per section in optimizer
    _MAX_SEARCH_SPACE   = 500_000  # fall back to greedy if product exceeds this

    # ── Layout cost weights ───────────────────────────────────────────────────
    _COST_CONSECUTIVE  = 100   # images with < MIN_SPACING prose gap
    _COST_CLOSE        = 50    # images with exactly MIN_SPACING prose gap
    _COST_TABLE_ADJ    = 40    # image immediately adjacent to a TABLE
    _COST_SHORT_SECT   = 30    # image in section with ≤ SECTION_MIN_PROSE prose blocks
    _COST_DISTRIBUTION = 20    # all images clustered in one half of the document
    _COST_NEAR_END     = 10    # image in the last 10% of document
    _COST_DEPTH_DEV    = 3     # penalty per position of deviation from adaptive target

    # ── Patterns ──────────────────────────────────────────────────────────────
    _MARKER_RE = re.compile(r'\n*<!-- SEO_AGENT_IMAGE: \S+ -->\n*')

    _FAQ_RE = re.compile(
        r'^#{1,4}\s+(?:FAQ|Frequently Asked Questions|Preguntas [Ff]recuentes)',
        re.IGNORECASE,
    )
    _CTA_RE = re.compile(
        r"^#{1,4}\s+(?:Don'?t Wait|Contact Us|Call Us|Get a Quote|Reach Out|"
        r"Schedule|Request|Book|Protect|Ready to|Let'?s Get|Start Today|"
        r"Free Estimate|About Us|Learn More|Get Started|Call Now)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        embed_fn: "Callable[[list[str]], list[list[float]]] | None" = None,
    ) -> None:
        self._embed_fn = embed_fn

    # ── Public interface ──────────────────────────────────────────────────────

    def place(
        self,
        markdown: str,
        images: "list[tuple[ImageRequest, ImageMetadata]]",
    ) -> PlacementResult:
        """
        Return a PlacementResult with updated markdown and an editorial score.

        ``images`` must contain only INLINE images; FEATURED images must be
        filtered out by the caller before invoking this method.
        """
        if not images:
            return PlacementResult(markdown=markdown, score=EditorialScore.empty())

        # 1. Remove all existing markers — positions recomputed from scratch.
        clean = self._MARKER_RE.sub('\n\n', markdown).strip()
        clean = re.sub(r'\n{3,}', '\n\n', clean)

        # 2. Parse markdown into a flat list of typed blocks.
        blocks = self._parse(clean)

        # 3. Build H2-delimited sections (section 0 = preamble).
        sections = self._build_sections(blocks)

        # 4. Assign each image to its best-matching section (semantic or lexical).
        assignments, semantic_scores = self._assign(sections, images, blocks)

        # 5. Global layout optimization: exhaustive cost-function minimization.
        insertions, layout_cost = self._optimize_layout(blocks, sections, assignments)

        # 6. Belt-and-suspenders spacing validation.
        insertions = self._validate_rhythm(blocks, insertions, sections)

        # 7. Compute editorial layout score.
        score = self._compute_score(
            blocks, sections, insertions, semantic_scores, layout_cost,
        )

        # 8. Rebuild markdown with markers at optimized positions.
        result_md = self._rebuild(blocks, insertions)

        return PlacementResult(markdown=result_md, score=score)

    # ── Step 1: Parse ─────────────────────────────────────────────────────────

    def _parse(self, md: str) -> list[Block]:
        """
        Parse markdown into a flat list of Blocks.

        Handles: headings, paragraphs, tables, unordered/ordered lists,
        blockquotes, fenced code blocks, HRs, blank lines, and SEO markers.
        """
        lines = md.splitlines()
        blocks: list[Block] = []
        i = 0

        while i < len(lines):
            raw = lines[i]
            s = raw.strip()

            # ── Fenced code block ─────────────────────────────────────────────
            if s.startswith('```') or s.startswith('~~~'):
                fence_char = s[:3]
                collected = [raw]
                i += 1
                while i < len(lines):
                    collected.append(lines[i])
                    if lines[i].strip().startswith(fence_char) and len(lines[i].strip()) >= 3:
                        i += 1
                        break
                    i += 1
                blocks.append(Block(BlockType.CODE, '\n'.join(collected)))
                continue

            # ── Blank line ────────────────────────────────────────────────────
            if not s:
                blocks.append(Block(BlockType.BLANK, ''))
                i += 1
                continue

            # ── SEO image marker (must check before generic HTML comment) ─────
            if re.match(r'^<!-- SEO_AGENT_IMAGE: \S+ -->', s):
                blocks.append(Block(BlockType.MARKER, raw))
                i += 1
                continue

            # ── HTML comments / inline HTML — treat as blank ──────────────────
            if s.startswith('<!--') or (s.startswith('<') and not re.match(
                r'^<(?:img|figure|p|div|h[1-6]|strong|em|a)\b', s, re.I
            )):
                blocks.append(Block(BlockType.BLANK, ''))
                i += 1
                continue

            # ── Headings ──────────────────────────────────────────────────────
            if re.match(r'^#{3,}\s', s):
                blocks.append(Block(BlockType.H3, raw))
                i += 1
                continue
            if s.startswith('## '):
                blocks.append(Block(BlockType.H2, raw))
                i += 1
                continue
            if s.startswith('# ') or re.match(r'^#\s', s):
                blocks.append(Block(BlockType.H1, raw))
                i += 1
                continue

            # ── Horizontal rule ───────────────────────────────────────────────
            if re.match(r'^[-*_]{3,}\s*$', s):
                blocks.append(Block(BlockType.HR, raw))
                i += 1
                continue

            # ── Blockquote ────────────────────────────────────────────────────
            if s.startswith('>'):
                collected = [raw]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('>'):
                    collected.append(lines[i])
                    i += 1
                blocks.append(Block(BlockType.BLOCKQUOTE, '\n'.join(collected)))
                continue

            # ── Table ─────────────────────────────────────────────────────────
            if s.startswith('|'):
                collected = [raw]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    collected.append(lines[i])
                    i += 1
                blocks.append(Block(BlockType.TABLE, '\n'.join(collected)))
                continue

            # ── Unordered list ────────────────────────────────────────────────
            if re.match(r'^[-*+] ', s):
                collected = [raw]
                i += 1
                while i < len(lines):
                    ns = lines[i]
                    nss = ns.strip()
                    if re.match(r'^[-*+] ', nss):
                        collected.append(ns)
                        i += 1
                    elif ns.startswith(('  ', '\t')) and nss:
                        collected.append(ns)
                        i += 1
                    else:
                        break
                blocks.append(Block(BlockType.UL, '\n'.join(collected)))
                continue

            # ── Ordered list ──────────────────────────────────────────────────
            if re.match(r'^\d+[.)]\s', s):
                collected = [raw]
                i += 1
                while i < len(lines):
                    ns = lines[i]
                    nss = ns.strip()
                    if re.match(r'^\d+[.)]\s', nss):
                        collected.append(ns)
                        i += 1
                    elif ns.startswith(('  ', '\t')) and nss:
                        collected.append(ns)
                        i += 1
                    else:
                        break
                blocks.append(Block(BlockType.OL, '\n'.join(collected)))
                continue

            # ── Paragraph (default) ───────────────────────────────────────────
            collected = [raw]
            i += 1
            while i < len(lines):
                ns = lines[i]
                if not ns.strip():
                    break
                if self._is_block_start(ns.strip()):
                    break
                collected.append(ns)
                i += 1
            blocks.append(Block(BlockType.PARAGRAPH, '\n'.join(collected)))

        return blocks

    @staticmethod
    def _is_block_start(s: str) -> bool:
        """Return True if this line begins a new block type (not paragraph continuation)."""
        return bool(
            re.match(r'^#{1,6}\s', s) or
            re.match(r'^[-*_]{3,}\s*$', s) or
            re.match(r'^[-*+] ', s) or
            re.match(r'^\d+[.)]\s', s) or
            s.startswith('>') or
            s.startswith('|') or
            s.startswith('```') or
            s.startswith('~~~') or
            s.startswith('<!-- ')
        )

    # ── Step 2: Build sections ────────────────────────────────────────────────

    def _build_sections(self, blocks: list[Block]) -> list[_Section]:
        """
        Group blocks into sections delimited by H2 headings.

        Section 0 is always the preamble (everything before the first H2).
        Each subsequent section spans from just after its H2 heading to just
        before the next H2 heading (or end of document).
        """
        sections: list[_Section] = []

        first_h2 = next(
            (i for i, b in enumerate(blocks) if b.type == BlockType.H2),
            len(blocks),
        )

        sections.append(_Section(
            heading_text='',
            heading_block=None,
            start=0,
            end=first_h2,
        ))

        i = first_h2
        while i < len(blocks):
            if blocks[i].type != BlockType.H2:
                i += 1
                continue

            heading_text = re.sub(r'^#{1,3}\s+', '', blocks[i].text.strip())

            j = i + 1
            while j < len(blocks) and blocks[j].type != BlockType.H2:
                j += 1

            sections.append(_Section(
                heading_text=heading_text,
                heading_block=i,
                start=i + 1,
                end=j,
                is_faq=bool(self._FAQ_RE.search(blocks[i].text)),
                is_cta=bool(self._CTA_RE.search(blocks[i].text)),
            ))
            i = j

        return sections

    # ── Step 3: Assign images to sections ────────────────────────────────────

    def _assign(
        self,
        sections: list[_Section],
        images: "list[tuple[ImageRequest, ImageMetadata]]",
        blocks: list[Block],
    ) -> "tuple[list[tuple[ImageRequest, ImageMetadata, _Section]], list[float]]":
        """
        Assign each image to the most suitable section.

        Returns (assignments, semantic_scores) where semantic_scores[i] is the
        matching confidence for assignments[i] in [0, 1] (0.0 for sequential
        fallbacks with no semantic signal).

        Matching strategy (priority order):
          1. OpenAI embeddings (cosine similarity, 85% weight) blended with
             lexical stemming (15% weight) — when embed_fn is available.
          2. Stemmed token overlap only — when embed_fn is None.
          3. Sequential assignment to next unused eligible section — fallback for
             images with no confident semantic match.

        Section eligibility rules:
          - Never assign to FAQ or CTA sections.
          - Each section accepts at most one image.
          - Preamble is eligible only if it has ≥ _MIN_INTRO_PARAS prose blocks.
        """
        preamble = sections[0]
        preamble_eligible = (
            sum(1 for i in range(preamble.start, preamble.end)
                if blocks[i].type == BlockType.PARAGRAPH)
            >= self._MIN_INTRO_PARAS
        )
        eligible: list[_Section] = []
        if preamble_eligible:
            eligible.append(preamble)
        eligible += [
            s for s in sections[1:]
            if not s.is_faq and not s.is_cta
        ]

        used:       set[str] = set()
        matched:    list[tuple] = []
        sem_scores: list[float] = []
        unmatched:  list[tuple] = []

        # ── Build embedding map (one batch API call for all texts) ─────────────
        embed_map: dict[str, list[float]] = {}
        use_embeddings = self._embed_fn is not None

        if use_embeddings:
            texts = (
                [req.section_title or '' for req, _ in images] +
                [s.heading_text for s in eligible]
            )
            try:
                all_embs = self._embed_fn(texts)
                for k in range(len(images)):
                    embed_map[f'img_{k}'] = all_embs[k]
                for j in range(len(eligible)):
                    embed_map[f'sect_{j}'] = all_embs[len(images) + j]
            except Exception as exc:
                logger.warning(
                    "Embedding API failed (%s); using lexical fallback for assignment.",
                    exc,
                )
                use_embeddings = False

        # ── Phase 1: semantic matching ─────────────────────────────────────────
        threshold = self._EMBED_THRESHOLD if use_embeddings else self._SEMANTIC_THRESHOLD

        for i, (req, meta) in enumerate(images):
            best_sect:  _Section | None = None
            best_score: float = 0.0

            for j, sect in enumerate(eligible):
                if sect.heading_text in used:
                    continue

                lex = self._semantic_score(req.section_title or '', sect.heading_text)

                if use_embeddings:
                    cos   = _cosine(
                        embed_map.get(f'img_{i}', []),
                        embed_map.get(f'sect_{j}', []),
                    )
                    score = 0.85 * cos + 0.15 * lex
                else:
                    score = lex

                if score > best_score:
                    best_score = score
                    best_sect  = sect

            if best_sect is not None and best_score >= threshold:
                used.add(best_sect.heading_text)
                matched.append((req, meta, best_sect))
                sem_scores.append(best_score)
            else:
                unmatched.append((req, meta))

        # ── Phase 2: sequential fallback for unmatched images ─────────────────
        remaining = [s for s in eligible if s.heading_text not in used]
        for (req, meta), sect in zip(unmatched, remaining):
            used.add(sect.heading_text)
            matched.append((req, meta, sect))
            sem_scores.append(0.0)   # no semantic signal; score omitted from average

        if len(unmatched) > len(remaining):
            logger.warning(
                "%d image(s) could not be assigned to any eligible section — "
                "they will be omitted from the published article.",
                len(unmatched) - len(remaining),
            )

        matched.sort(key=lambda x: x[2].start)
        return matched, sem_scores

    # ── Step 4: Global layout optimization ───────────────────────────────────

    def _optimize_layout(
        self,
        blocks: list[Block],
        sections: list[_Section],
        assignments: "list[tuple[ImageRequest, ImageMetadata, _Section]]",
    ) -> "tuple[dict[int, list[str]], float]":
        """
        Select image positions by minimizing a global layout cost function.

        For each assigned (image, section) pair, all eligible candidate positions
        inside the section are collected.  The optimizer then evaluates every
        combination (exhaustive when ≤ _MAX_SEARCH_SPACE, greedy otherwise) and
        returns the combination with the lowest total cost.

        Returns (insertions, best_cost).
        """
        # Collect eligible candidates per assignment (soft danger → strict fallback)
        search_items: list[tuple] = []
        for req, meta, sect in assignments:
            cands = self._collect_eligible(blocks, sect, _SOFT_DANGER)
            if not cands:
                cands = self._collect_eligible(blocks, sect, _STRICT_DANGER)
            if cands:
                search_items.append((req, meta, sect, cands[:self._MAX_CANDIDATES]))
            else:
                logger.warning(
                    "No valid position for %s in section %r — image omitted.",
                    req.id, sect.heading_text or "preamble",
                )

        if not search_items:
            return {}, 0.0

        candidate_lists = [item[3] for item in search_items]

        # Compute search space (early-exit once we exceed the cap)
        space = 1
        for cl in candidate_lists:
            space *= len(cl)
            if space > self._MAX_SEARCH_SPACE:
                break

        if space > self._MAX_SEARCH_SPACE:
            logger.info(
                "Search space %d > %d; using greedy layout optimizer.",
                space, self._MAX_SEARCH_SPACE,
            )
            return self._greedy_layout(blocks, sections, search_items)

        # Exhaustive search over all position combinations
        best_cost: float = float('inf')
        best_combo: list[int] | None = None

        for combo in _iproduct(*candidate_lists):
            cost = self._layout_cost(blocks, sections, search_items, list(combo))
            if cost < best_cost:
                best_cost = cost
                best_combo = list(combo)

        insertions: dict[int, list[str]] = {}
        for (req, meta, sect, _), pos in zip(search_items, best_combo or []):
            insertions.setdefault(pos, []).append(req.placement_marker)

        return insertions, best_cost

    def _greedy_layout(
        self,
        blocks: list[Block],
        sections: list[_Section],
        search_items: list[tuple],
    ) -> "tuple[dict[int, list[str]], float]":
        """
        Greedy per-image cost minimization — fallback for large search spaces.

        For each image in document order, picks the candidate position that
        minimizes the cumulative layout cost given all previously placed images.
        Less optimal than exhaustive search but much faster.
        """
        insertions: dict[int, list[str]] = {}
        placed:     list[int] = []
        total_cost: float = 0.0

        for k, (req, meta, sect, cands) in enumerate(search_items):
            best_pos:  int | None = None
            best_cost: float = float('inf')

            for pos in cands:
                cost = self._layout_cost(
                    blocks, sections,
                    search_items[:k + 1],
                    placed + [pos],
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pos  = pos

            if best_pos is not None:
                insertions.setdefault(best_pos, []).append(req.placement_marker)
                placed.append(best_pos)
                total_cost = best_cost

        return insertions, total_cost

    def _layout_cost(
        self,
        blocks: list[Block],
        sections: list[_Section],
        search_items: list[tuple],
        positions: list[int],
    ) -> float:
        """
        Global layout cost for a candidate set of image positions.

        Lower cost = better editorial layout.  The optimizer selects the
        position combination that minimizes this value.

        Penalties applied:
          +100  consecutive images (< MIN_SPACING prose gap)
          + 50  close images (exactly MIN_SPACING prose gap)
          + 40  image immediately adjacent to a TABLE (before or after)
          + 30  image in a very short section (≤ SECTION_MIN_PROSE prose blocks)
          + 20  all images clustered in the first or second half of document
          + 10  image in the last 10% of document
          +  3  × deviation from adaptive target depth (within eligible list)
        """
        sorted_pos = sorted(positions)
        n     = len(sorted_pos)
        total = len(blocks)
        cost  = 0.0

        # ── Spacing penalties ──────────────────────────────────────────────────
        for i in range(n - 1):
            gap = self._count_prose_between(blocks, sorted_pos[i], sorted_pos[i + 1])
            if gap < self._MIN_SPACING:
                cost += self._COST_CONSECUTIVE
            elif gap == self._MIN_SPACING:
                cost += self._COST_CLOSE

        # ── Per-position penalties ────────────────────────────────────────────
        for k, pos in enumerate(positions):
            prev = self._prev_significant(blocks, pos)
            nxt  = self._next_significant(blocks, pos)

            # Table adjacency
            if prev is not None and blocks[prev].type == BlockType.TABLE:
                cost += self._COST_TABLE_ADJ
            if nxt  is not None and blocks[nxt ].type == BlockType.TABLE:
                cost += self._COST_TABLE_ADJ

            # Short section
            sect = self._section_for(sections, pos)
            if sect is not None:
                sect_prose = sum(
                    1 for idx in range(sect.start, sect.end)
                    if blocks[idx].type == BlockType.PARAGRAPH
                )
                if sect_prose <= self._SECTION_MIN_PROSE:
                    cost += self._COST_SHORT_SECT

                # Adaptive depth deviation
                if k < len(search_items):
                    _, _, _, cands = search_items[k]
                    if pos in cands:
                        idx_in_cands = cands.index(pos)
                        target = min(self._target_index(len(cands)), len(cands) - 1)
                        cost += abs(idx_in_cands - target) * self._COST_DEPTH_DEV

            # Near-end penalty
            if total > 0 and pos > total * 0.90:
                cost += self._COST_NEAR_END

        # ── Distribution penalties ────────────────────────────────────────────
        if n > 1 and total > 0:
            if max(sorted_pos) / total < 0.50:
                cost += self._COST_DISTRIBUTION   # all images in first half
            if min(sorted_pos) / total > 0.50:
                cost += self._COST_DISTRIBUTION   # all images in second half

        return cost

    # ── Step 5: Collect eligible candidate positions ──────────────────────────

    def _collect_eligible(
        self,
        blocks: list[Block],
        sect: "_Section",
        danger_set: "frozenset[BlockType]",
    ) -> "list[int]":
        """
        Return PARAGRAPH block indices within ``sect`` that are not immediately
        adjacent (prev or next significant block) to any block in ``danger_set``.
        """
        eligible: list[int] = []
        for i in range(sect.start, sect.end):
            if blocks[i].type != BlockType.PARAGRAPH:
                continue
            prev = self._prev_significant(blocks, i)
            if prev is not None and blocks[prev].type in danger_set:
                continue
            nxt = self._next_significant(blocks, i)
            if nxt is not None and blocks[nxt].type in danger_set:
                continue
            eligible.append(i)
        return eligible

    # ── Step 6: Belt-and-suspenders rhythm validation ─────────────────────────

    def _validate_rhythm(
        self,
        blocks: list[Block],
        insertions: "dict[int, list[str]]",
        sections: list[_Section],
    ) -> "dict[int, list[str]]":
        """
        Final spacing pass over all insertions.

        The cost-function optimizer enforces spacing via heavy penalties, so
        violations here are rare.  This pass catches any remaining edge cases,
        relocating violators forward and omitting them only if no valid position
        exists.
        """
        pairs: list[tuple[int, str]] = [
            (idx, marker)
            for idx in sorted(insertions)
            for marker in insertions[idx]
        ]

        result:   dict[int, list[str]] = {}
        last_img: int = -1000   # sentinel: no previous image

        for idx, marker in pairs:
            gap = self._count_prose_between(blocks, last_img, idx)

            if last_img < 0 or gap >= self._MIN_SPACING:
                result.setdefault(idx, []).append(marker)
                last_img = idx
                continue

            # Spacing violation — scan forward for next safe paragraph.
            relocated = False
            for candidate in range(idx + 1, len(blocks)):
                if blocks[candidate].type != BlockType.PARAGRAPH:
                    continue
                sect = self._section_for(sections, candidate)
                if sect is not None and (sect.is_faq or sect.is_cta):
                    continue
                prev = self._prev_significant(blocks, candidate)
                nxt  = self._next_significant(blocks, candidate)
                if (prev is not None and blocks[prev].type in _STRICT_DANGER) or \
                   (nxt  is not None and blocks[nxt ].type in _STRICT_DANGER):
                    continue
                if self._count_prose_between(blocks, last_img, candidate) >= self._MIN_SPACING:
                    result.setdefault(candidate, []).append(marker)
                    last_img = candidate
                    logger.warning(
                        "Rhythm: relocated %s from block %d → %d (spacing violation).",
                        marker, idx, candidate,
                    )
                    relocated = True
                    break

            if not relocated:
                logger.warning(
                    "Rhythm: omitted %s — no position with ≥%d prose gap after block %d.",
                    marker, self._MIN_SPACING, last_img,
                )

        return result

    # ── Step 7: Editorial scoring ─────────────────────────────────────────────

    def _compute_score(
        self,
        blocks: list[Block],
        sections: list[_Section],
        insertions: "dict[int, list[str]]",
        semantic_scores: list[float],
        layout_cost: float,
    ) -> EditorialScore:
        """
        Compute a multi-dimensional editorial layout score from the final layout.

        Dimensions:
          distribution     — evenness of image spread across the document
          spacing          — prose gap between consecutive images
          section_matching — semantic confidence of image→section assignments
          visual_rhythm    — derived from the optimizer's layout cost
          overall          — weighted average (30% distribution, 25% each others)
        """
        positions = sorted(insertions.keys())
        n     = len(positions)
        total = len(blocks)
        warnings: list[str] = []

        # ── Spacing (0–100) ───────────────────────────────────────────────────
        if n <= 1:
            spacing_score = 95
        else:
            gaps = [
                self._count_prose_between(blocks, positions[i], positions[i + 1])
                for i in range(n - 1)
            ]
            min_gap = min(gaps)
            avg_gap = sum(gaps) / len(gaps)

            if min_gap < self._MIN_SPACING:
                spacing_score = 35
                warnings.append(
                    f"Consecutive images detected: minimum gap is {min_gap} prose "
                    f"paragraph(s) (required: ≥{self._MIN_SPACING})."
                )
            elif min_gap == self._MIN_SPACING:
                spacing_score = 72
            elif avg_gap >= 4.0:
                spacing_score = 97
            elif avg_gap >= 3.0:
                spacing_score = 88
            else:
                spacing_score = 80

        # ── Distribution (0–100) ──────────────────────────────────────────────
        if n == 0:
            dist_score = 0
        elif n == 1:
            frac = positions[0] / max(total, 1)
            dist_score = 90 if 0.05 <= frac <= 0.65 else 65
        else:
            span_frac = (max(positions) - min(positions)) / max(total, 1)
            last_frac = max(positions) / max(total, 1)
            if span_frac >= 0.55 and last_frac >= 0.55:
                dist_score = 97
            elif span_frac >= 0.40:
                dist_score = 88
            elif span_frac >= 0.25:
                dist_score = 74
            else:
                dist_score = 55
                warnings.append(
                    "Images are clustered: they span less than 25% of the article. "
                    "Consider assigning images to sections that are more spread out."
                )

        # ── Section matching (0–100) ──────────────────────────────────────────
        # Scale raw semantic scores (lexical ~0–0.8; cosine blend ~0.25–1.0) to 50–100.
        matched_scores = [s for s in semantic_scores if s > 0.0]
        if matched_scores:
            avg_sem   = sum(matched_scores) / len(matched_scores)
            sem_score = min(100, int(50 + avg_sem * 55))
        else:
            # All images fell back to sequential assignment; no semantic signal.
            sem_score = 72

        # ── Visual rhythm (0–100) ─────────────────────────────────────────────
        if layout_cost == 0.0:
            rhythm_score = 100
        elif layout_cost < 10:
            rhythm_score = 97
        elif layout_cost < 30:
            rhythm_score = 88
        elif layout_cost < 60:
            rhythm_score = 72
        elif layout_cost < 100:
            rhythm_score = 55
        else:
            rhythm_score = 35
            warnings.append(
                f"High layout cost ({layout_cost:.0f}): images may be adjacent to "
                "tables or placed in very short sections."
            )

        # ── Overall (weighted average) ────────────────────────────────────────
        overall = int(
            0.30 * dist_score    +
            0.25 * spacing_score +
            0.25 * sem_score     +
            0.20 * rhythm_score
        )

        return EditorialScore(
            overall=overall,
            distribution=dist_score,
            spacing=spacing_score,
            section_matching=sem_score,
            visual_rhythm=rhythm_score,
            layout_cost=layout_cost,
            warnings=warnings,
        )

    # ── Step 8: Rebuild ───────────────────────────────────────────────────────

    def _rebuild(
        self,
        blocks: list[Block],
        insertions: "dict[int, list[str]]",
    ) -> str:
        """
        Rebuild the markdown string from the flat block list, inserting image
        markers at the computed positions.

        Old MARKER blocks are dropped; new markers come from ``insertions``.
        """
        parts: list[str] = []

        for i, block in enumerate(blocks):
            if block.type == BlockType.BLANK:
                parts.append('')
            elif block.type == BlockType.MARKER:
                pass   # drop old markers
            else:
                parts.append(block.text)

            for marker in insertions.get(i, []):
                parts.append('')
                parts.append(marker)

        result = '\n'.join(parts)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _target_index(eligible_count: int) -> int:
        """
        Adaptive insertion depth based on the number of eligible paragraphs.

        ≤ 2  → after paragraph 1 (index 0)  — short section, top placement
        3–5  → after paragraph 2 (index 1)  — medium section
        6–8  → after paragraph 3 (index 2)  — long section
        9+   → ~37% deep (minimum index 3)  — very long section
        """
        if eligible_count <= 2:
            return 0
        elif eligible_count <= 5:
            return 1
        elif eligible_count <= 8:
            return 2
        else:
            return max(3, int(eligible_count * 0.37))

    @staticmethod
    def _stem(word: str) -> str:
        """Single-pass suffix stripping followed by trailing-e removal."""
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                word = word[:-len(suffix)]
                break
        if word.endswith('e') and len(word) > 3:
            word = word[:-1]
        return word

    @classmethod
    def _tokenize(cls, text: str) -> frozenset[str]:
        """Normalize text to a frozenset of stemmed content tokens."""
        tokens = re.findall(r'[a-z]+', text.lower())
        return frozenset(
            cls._stem(t)
            for t in tokens
            if t not in _STOP_WORDS and len(t) > 2
        )

    @classmethod
    def _semantic_score(cls, text_a: str, text_b: str) -> float:
        """
        Stemmed-token similarity between two short text strings in [0, 1].

        Blends overlap coefficient (normalized by the smaller set; better for
        short queries) with Jaccard (penalises very different set sizes).
        Used as the primary signal in lexical mode and as a tiebreaker when
        embeddings are active.
        """
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        if intersection == 0:
            return 0.0
        overlap = intersection / min(len(tokens_a), len(tokens_b))
        jaccard  = intersection / len(tokens_a | tokens_b)
        return 0.7 * overlap + 0.3 * jaccard

    @staticmethod
    def _count_prose_between(blocks: list[Block], start: int, end: int) -> int:
        """
        Count PARAGRAPH blocks strictly between flat-list indices ``start`` and ``end``.

        ``start`` may be a negative sentinel (−1000) meaning "no previous image".
        The lower bound is clamped to 0 so no negative indexing occurs.
        """
        lo = max(0, min(start, end) + 1)
        hi = max(start, end)
        if lo >= hi:
            return 0
        return sum(1 for i in range(lo, hi) if blocks[i].type == BlockType.PARAGRAPH)

    @staticmethod
    def _prev_significant(blocks: list[Block], i: int) -> "int | None":
        """Index of the nearest non-blank, non-marker block BEFORE i."""
        for j in range(i - 1, -1, -1):
            if blocks[j].type not in (BlockType.BLANK, BlockType.MARKER):
                return j
        return None

    @staticmethod
    def _next_significant(blocks: list[Block], i: int) -> "int | None":
        """Index of the nearest non-blank, non-marker block AFTER i."""
        for j in range(i + 1, len(blocks)):
            if blocks[j].type not in (BlockType.BLANK, BlockType.MARKER):
                return j
        return None

    @staticmethod
    def _section_for(
        sections: list[_Section],
        block_idx: int,
    ) -> "_Section | None":
        """Return the section whose [start, end) range contains block_idx."""
        for s in sections:
            if s.start <= block_idx < s.end:
                return s
        return None

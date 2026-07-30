import re
from dataclasses import dataclass

from models.article import Article
from models.seo_report import IssueSeverity, SEOIssue, SEOReport, SEOSummary


# ── Rule definition ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Rule:
    """
    Internal rule descriptor. Each rule defines its own penalty explicitly
    rather than inheriting a flat amount from its severity tier.
    """
    code: str
    severity: IssueSeverity
    penalty: int
    message: str


# ── Rule table ────────────────────────────────────────────────────────────────
# Ordered by severity then impact. Add new rules here — no other file changes.
# CRITICAL issues block publishing unconditionally (independent of min_score).
# INFO issues are reported but carry zero penalty.

_RULE_LIST: list[_Rule] = [
    # CRITICAL — must-fix conditions
    _Rule("content_empty",                  IssueSeverity.CRITICAL, 40, "Article content is empty."),
    _Rule("seo_title_missing",              IssueSeverity.CRITICAL, 35, "SEO title is empty."),

    # ERROR — significant problems
    _Rule("content_too_short",              IssueSeverity.ERROR,    25, "Article is shorter than the minimum word count."),
    _Rule("no_h2_in_body",                  IssueSeverity.ERROR,    20, "No subheadings (H2+) found in article body."),
    _Rule("keyword_not_in_title",           IssueSeverity.ERROR,    15, "Focus keyword words are missing from the article title."),
    _Rule("keyword_not_in_first_100_words", IssueSeverity.ERROR,    15, "Focus keyword words do not appear in the first 100 words."),
    _Rule("slug_has_uppercase",             IssueSeverity.ERROR,    12, "Slug contains uppercase letters (breaks URL consistency)."),

    # WARNING — notable concerns
    _Rule("h1_in_body",                     IssueSeverity.WARNING,   8, "H1 heading found in body content (duplicates the WordPress page title)."),
    _Rule("broken_heading_hierarchy",       IssueSeverity.WARNING,  10, "Heading hierarchy skips a level (e.g. H2 → H4)."),
    _Rule("no_internal_links",              IssueSeverity.WARNING,   8, "No internal links found in the content."),
    _Rule("title_too_long",                 IssueSeverity.WARNING,   8, "SEO title exceeds the recommended 60-character limit."),
    _Rule("meta_desc_too_long",             IssueSeverity.WARNING,   8, "Meta description exceeds the recommended 160-character limit."),
    _Rule("meta_desc_too_short",            IssueSeverity.WARNING,   6, "Meta description is shorter than the recommended minimum."),
    _Rule("slug_too_long",                  IssueSeverity.WARNING,   5, "Slug exceeds the recommended 75-character limit."),

    # INFO — observations, no score impact
    _Rule("title_too_short",               IssueSeverity.INFO,      0, "SEO title is shorter than the recommended minimum."),
    _Rule("keyword_not_in_meta_desc",      IssueSeverity.INFO,      0, "Focus keyword words not present in the meta description."),
]

_RULES: dict[str, _Rule] = {r.code: r for r in _RULE_LIST}


# ── Keyword matching helpers ──────────────────────────────────────────────────

# Words that carry no SEO signal on their own. Removed from both the keyword
# and the target text before matching so they don't create false negatives.
# Example: keyword "garage door repair in Denver" → significant words:
#          {garage, door, repair, denver}  ("in" is excluded).
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "as", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "its", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "your", "what", "which", "who", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "not",
})


def _strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting and return plain readable text.

    Converts [anchor text](url) → anchor text, strips heading markers,
    bold/italic syntax, HTML comments, and other markup so that word
    counts and keyword checks operate on actual content words.
    """
    # [anchor](url) → anchor
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # Heading markers: ## Title → Title
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bold/italic/strikethrough
    text = re.sub(r'[*_~]{1,3}', '', text)
    # Inline code
    text = re.sub(r'`[^`]+`', ' ', text)
    # HTML comments (e.g. <!-- SEO_AGENT_IMAGE: img_001 -->)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    # Remaining Markdown special characters
    text = re.sub(r'[#>|\\]', ' ', text)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


def _significant_words(text: str) -> frozenset[str]:
    """
    Normalize text to its set of meaningful words.

    Steps: strip Markdown → lowercase → remove punctuation → split →
    drop stop words → drop tokens shorter than 2 characters.

    Returns a frozenset so callers can do fast membership tests.
    """
    plain = _strip_markdown(text).lower()
    plain = re.sub(r"[^\w\s]", " ", plain)   # punctuation → space
    return frozenset(
        w for w in plain.split()
        if w not in _STOP_WORDS and len(w) >= 2
    )


def _missing_kw_words(keyword: str, text: str) -> frozenset[str]:
    """
    Return the significant keyword words NOT found in text.

    An empty result means all keyword words are present — the keyword matches.
    A non-empty result lists only the specific words that are absent.

    Singular/plural variants handled:
    - "repair" matches "repairs"   (keyword singular, text plural: word + 's')
    - "service" matches "services" (keyword singular, text plural: word + 'es')
    - "repairs" matches "repair"   (keyword plural, text singular: word[:-1])
    - "services" matches "service" (keyword plural, text singular: word[:-1])

    This covers the common English cases without a full stemmer, which would
    add a dependency and risk false positives on irregular forms.
    """
    kw_words = _significant_words(keyword)
    if not kw_words:
        return frozenset()

    text_words = _significant_words(text)
    missing: set[str] = set()

    for word in kw_words:
        found = (
            word in text_words
            or (word + "s") in text_words                             # repair → repairs
            or (word + "es") in text_words                            # service → services
            or (word.endswith("s") and len(word) > 3                  # repairs → repair
                and word[:-1] in text_words)
        )
        if not found:
            missing.add(word)

    return frozenset(missing)


# ── Service ───────────────────────────────────────────────────────────────────

class SEOQAService:
    """
    Analyzes an Article and returns a scored SEOReport.

    Completely stateless — no I/O, no side effects. Can be instantiated
    and called anywhere without setup. The same instance can be reused
    safely across multiple analyze() calls.

    Score calculation:
        Starts at 100. Each issue deducts its individual penalty.
        Score is clamped to [0, 100].

    CRITICAL issues are reported and penalized, but their primary
    significance is that they always block publishing — regardless of
    whether the accumulated score would still pass the threshold.
    That decision is made by PublisherAgent, not here.

    Keyword matching (title, intro, meta description):
        Uses word-level coverage rather than exact substring matching.
        Stop words are excluded from both the keyword and the target text.
        Simple singular/plural variants are considered equivalent.
        This reflects how search engines interpret keyword presence and
        avoids false negatives from prepositions, articles, or word order
        variations that appear in natural editorial titles.

    Adding a new check:
        1. Add a _Rule entry to _RULE_LIST.
        2. Add a branch in the appropriate _check_*() method.
        No other files need to change.
    """

    def analyze(self, article: Article) -> SEOReport:
        """Run all checks and return a fully populated SEOReport."""
        issues: list[SEOIssue] = []

        self._check_content(article, issues)
        self._check_headings(article.content_markdown, issues)
        self._check_seo_fields(article, issues)
        self._check_keyword(article, issues)

        score = max(0, 100 - sum(i.penalty for i in issues))

        summary = SEOSummary(
            critical=sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            errors=sum(1 for i in issues if i.severity == IssueSeverity.ERROR),
            warnings=sum(1 for i in issues if i.severity == IssueSeverity.WARNING),
            info=sum(1 for i in issues if i.severity == IssueSeverity.INFO),
        )

        return SEOReport(score=score, issues=issues, summary=summary)

    # ── Check groups ──────────────────────────────────────────────────────────

    def _check_content(self, article: Article, issues: list[SEOIssue]) -> None:
        if not article.content_markdown.strip():
            issues.append(self._issue("content_empty"))
            return  # all other content checks are meaningless on empty content

        if article.word_count < 300:
            issues.append(self._issue(
                "content_too_short",
                f"{article.word_count} words (minimum: 300)",
            ))

        # Internal-link check requires knowing the site domain to distinguish
        # internal links from external ones. Skip when domain is unavailable.
        _site_url = article.request.website_url if article.request else None
        if _site_url:
            from urllib.parse import urlparse
            _domain = urlparse(_site_url).netloc
            if _domain and not re.search(
                re.escape(_domain), article.content_markdown
            ):
                issues.append(self._issue("no_internal_links"))

    def _check_headings(self, content: str, issues: list[SEOIssue]) -> None:
        # Architecture note: article.content_markdown is the body WITHOUT the H1.
        # The H1 lives in article.title and WordPress renders it as the page title.
        # An H1 in the body would create a duplicate H1 on the rendered page.
        headings = re.findall(r'^(#{1,6})\s+.+', content, re.MULTILINE)
        levels = [len(h) for h in headings]

        h1_count = levels.count(1)
        if h1_count > 0:
            issues.append(self._issue(
                "h1_in_body",
                f"{h1_count} H1 heading(s) in body — conflicts with WordPress page title",
            ))

        # Body should have at least one H2 to create content structure.
        body_levels = [lv for lv in levels if lv >= 2]
        if not body_levels:
            issues.append(self._issue("no_h2_in_body"))
            return

        # Hierarchy check on H2+ levels only (H1 is the title, not part of the body outline).
        for i in range(1, len(body_levels)):
            if body_levels[i] > body_levels[i - 1] + 1:
                issues.append(self._issue(
                    "broken_heading_hierarchy",
                    f"H{body_levels[i - 1]} followed by H{body_levels[i]}",
                ))
                break  # report only the first occurrence

    def _check_seo_fields(self, article: Article, issues: list[SEOIssue]) -> None:
        # SEO title
        if not article.seo.seo_title.strip():
            issues.append(self._issue("seo_title_missing"))
        else:
            t_len = len(article.seo.seo_title)
            if t_len > 60:
                issues.append(self._issue("title_too_long", f"{t_len} chars (max: 60)"))
            elif t_len < 30:
                issues.append(self._issue("title_too_short", f"{t_len} chars (recommended: 30–60)"))

        # Meta description
        m_len = len(article.seo.meta_description)
        if m_len > 160:
            issues.append(self._issue("meta_desc_too_long", f"{m_len} chars (max: 160)"))
        elif m_len < 120:
            issues.append(self._issue("meta_desc_too_short", f"{m_len} chars (recommended: 120–160)"))

        # Slug
        slug = article.seo.slug
        if slug != slug.lower():
            issues.append(self._issue("slug_has_uppercase", f"'{slug}'"))
        elif len(slug) > 75:
            issues.append(self._issue("slug_too_long", f"{len(slug)} chars (max: 75)"))

    def _check_keyword(self, article: Article, issues: list[SEOIssue]) -> None:
        """
        Keyword presence checks using word-level coverage.

        Instead of exact substring matching, each check extracts the
        meaningful words from the keyword (stop words removed) and verifies
        that all of them appear in the target field. Simple plural/singular
        variants are treated as equivalent.

        This mirrors how search engines assess keyword relevance — presence
        of the significant terms matters more than their exact contiguous
        arrangement.
        """
        kw = article.seo.focus_keyword.strip()
        if not kw:
            return  # no keyword defined — skip all keyword checks

        # ── Title ─────────────────────────────────────────────────────────────
        missing_title = _missing_kw_words(kw, article.title)
        if missing_title:
            issues.append(self._issue(
                "keyword_not_in_title",
                f"Missing: {', '.join(sorted(missing_title))} "
                f"— keyword: '{kw}'",
            ))

        # ── First 100 words ───────────────────────────────────────────────────
        # Strip Markdown before splitting so heading markers (##), bold (**),
        # and link syntax ([text](url)) don't inflate the word count.
        plain_content = _strip_markdown(article.content_markdown)
        first_100_text = " ".join(plain_content.split()[:100])
        missing_intro = _missing_kw_words(kw, first_100_text)
        if missing_intro:
            issues.append(self._issue(
                "keyword_not_in_first_100_words",
                f"Missing: {', '.join(sorted(missing_intro))} "
                f"— keyword: '{kw}'",
            ))

        # ── Meta description (INFO — no score penalty) ────────────────────────
        missing_meta = _missing_kw_words(kw, article.seo.meta_description)
        if missing_meta:
            issues.append(self._issue(
                "keyword_not_in_meta_desc",
                f"Missing: {', '.join(sorted(missing_meta))} — keyword: '{kw}'",
            ))

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def _issue(code: str, detail: str | None = None) -> SEOIssue:
        rule = _RULES[code]
        return SEOIssue(
            severity=rule.severity,
            code=code,
            message=rule.message,
            detail=detail,
            penalty=rule.penalty,
        )

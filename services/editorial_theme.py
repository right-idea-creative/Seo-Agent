"""
Editorial Theme — the single source of truth for article presentation values.

The theme defines appearance. EditorialHTMLRenderer defines rendering logic.
Separating these concerns means any future website brand can supply its own
EditorialTheme without touching a single line of rendering code.

Usage::

    # Production default (matches the original hardcoded renderer exactly)
    renderer = EditorialHTMLRenderer()                             # uses DefaultEditorialTheme
    renderer = EditorialHTMLRenderer(theme=DefaultEditorialTheme())  # explicit

    # Future brand overrides
    renderer = EditorialHTMLRenderer(theme=OverheadDoorTheme())
    renderer = EditorialHTMLRenderer(theme=BlueTheme())

Creating a custom theme::

    theme = EditorialTheme(
        body_font="'Merriweather', Georgia, serif",
        heading_font="'Montserrat', 'Helvetica Neue', sans-serif",
        code_font="'JetBrains Mono', monospace",
        primary="#111827",
        accent="#b91c1c",          # red brand accent
        ...
    )

All fields are required; supply every value explicitly so the theme is
self-contained and portable across sites.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Callout variant ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalloutVariant:
    """
    Visual definition for one callout type (background fill + accent border).

    background:   CSS color value for the callout box fill.
    border_color: CSS color value for the left accent bar.

    Example::

        CalloutVariant(background="#eff6ff", border_color="#3b82f6")  # blue info
    """
    background: str
    border_color: str


# ── Theme ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EditorialTheme:
    """
    Complete visual identity specification for an editorial article.

    Owns every presentation value that varies between brands:
    - Font stacks (body, heading, code)
    - Type scale (sizes for all heading levels, body, captions)
    - Spacing rhythm (paragraph gaps, section margins, image margins)
    - Color palette (semantic names — primary, accent, muted, border, link…)
    - Component appearance (image corners, code blocks, callout shapes, tables)
    - Feature flags (striped tables, responsive wrapper, external-link behavior)

    The theme is frozen (immutable after construction). Create a new instance
    to represent a different brand; never mutate an existing theme.

    Note: ``callout_variants`` is a plain dict, which means the dataclass is
    not hashable. This is intentional — themes are configuration objects, not
    dict keys.
    """

    # ── Fonts ─────────────────────────────────────────────────────────────────
    body_font: str
    """CSS font-family stack for body text and list items."""

    heading_font: str
    """CSS font-family stack for headings, tables, captions, and callouts."""

    code_font: str
    """CSS font-family stack for <pre> blocks and inline <code>."""

    # ── Type scale ────────────────────────────────────────────────────────────
    h1_size: str
    """CSS font-size for H1 (article title)."""

    h2_size: str
    """CSS font-size for H2 (section headings)."""

    h3_size: str
    """CSS font-size for H3 (sub-section headings)."""

    h4_size: str
    """CSS font-size for H4."""

    body_size: str
    """CSS font-size for body paragraphs and list items."""

    lead_size: str
    """CSS font-size for the first paragraph (editorial lead / intro paragraph)."""

    caption_size: str
    """CSS font-size for <figcaption> image captions."""

    code_block_size: str
    """CSS font-size inside <pre> code blocks."""

    table_size: str
    """CSS font-size for table cells."""

    # ── Spacing ───────────────────────────────────────────────────────────────
    paragraph_spacing: str
    """Bottom margin applied to every <p> tag."""

    heading_spacing: str
    """Top margin on H2 — the primary section rhythm control."""

    h3_spacing: str
    """Top margin on H3."""

    h4_spacing: str
    """Top margin on H4."""

    image_spacing: str
    """Top margin on <figure> blocks."""

    table_spacing: str
    """Top and bottom margin on the table scroll wrapper."""

    list_indent: str
    """padding-left on <ul> and <ol> (controls bullet/number indentation)."""

    list_item_spacing: str
    """Bottom margin on each <li>."""

    # ── Colors ────────────────────────────────────────────────────────────────
    primary: str
    """Heading text color — darkest, highest contrast."""

    secondary: str
    """Body text color — slightly lighter than primary for long-form readability."""

    accent: str
    """Brand accent — H2 left border bar, internal link underline decoration."""

    muted_text: str
    """Secondary / caption text — figcaption, metadata, de-emphasized labels."""

    border: str
    """Table borders and dividers."""

    table_header_bg: str
    """<th> background color."""

    table_header_text: str
    """<th> text color."""

    table_stripe: str
    """Alternating (even) row background in striped tables."""

    external_link: str
    """Color of external hyperlinks."""

    internal_link: str
    """Color of internal hyperlinks."""

    lead_text_color: str
    """Color of the lead (first) paragraph — slightly softer than body text."""

    code_bg: str
    """<pre> block background color."""

    code_text: str
    """<pre> block text color."""

    code_inline_bg: str
    """Inline <code> background color."""

    code_inline_text: str
    """Inline <code> text color."""

    # ── Images ────────────────────────────────────────────────────────────────
    image_border_radius: str
    """CSS border-radius on <img> elements."""

    image_shadow: str
    """CSS box-shadow on <img> elements."""

    # ── Component borders ─────────────────────────────────────────────────────
    callout_border_radius: str
    """CSS border-radius on callout boxes (right corners only, e.g. "0 8px 8px 0")."""

    callout_font_size: str
    """CSS font-size for callout body text."""

    code_border_radius: str
    """CSS border-radius on <pre> fenced code blocks."""

    code_inline_border_radius: str
    """CSS border-radius on inline <code> elements."""

    table_wrapper_border_radius: str
    """CSS border-radius on the table scroll wrapper div."""

    # ── Feature flags ─────────────────────────────────────────────────────────
    table_striped: bool
    """Apply alternating row backgrounds to data tables."""

    table_responsive: bool
    """Wrap tables in an overflow-x:auto div for mobile scrolling."""

    external_links_new_tab: bool
    """Add target="_blank" rel="noopener noreferrer" to external links."""

    # ── Link decoration ───────────────────────────────────────────────────────
    external_link_decoration: str
    """CSS text-decoration for external links (e.g. "underline", "none")."""

    internal_link_decoration: str
    """CSS text-decoration for internal links."""

    internal_link_weight: str
    """CSS font-weight for internal links (e.g. "600" to make them slightly bold)."""

    # ── Callouts ─────────────────────────────────────────────────────────────
    callout_variants: dict[str, CalloutVariant]
    """
    Emoji → callout variant map.

    Keys are leading emoji characters. When the first character of a blockquote
    paragraph matches a key, that variant's colors are used. Example::

        {
            "💡": CalloutVariant("#eff6ff", "#3b82f6"),  # blue tip
            "⚠️": CalloutVariant("#fffbeb", "#f59e0b"),  # amber warning
        }
    """

    callout_default: CalloutVariant
    """Fallback variant for blockquotes whose first character is not in callout_variants."""

    # ── Reserved / forward-compatibility ──────────────────────────────────────
    content_max_width: str
    """
    Maximum content column width (e.g. "720px").

    Currently reserved for future use. WordPress themes control the content
    column width through their own CSS — this value is not yet applied to
    output HTML. It is included so theme objects carry the brand's intended
    reading measure when that feature is added.
    """


# ── Default theme ─────────────────────────────────────────────────────────────

def DefaultEditorialTheme() -> EditorialTheme:
    """
    Return the standard production editorial theme.

    Visual output is **identical** to the original hardcoded renderer — this
    is a refactor, not a redesign. Every value here was previously a literal
    constant inside EditorialHTMLRenderer or its _Tokens helper class.

    Use this as the base for custom themes: inspect the values you want to
    change and supply them in a new EditorialTheme() call.
    """
    return EditorialTheme(
        # ── Fonts ─────────────────────────────────────────────────────────────
        body_font    = "Georgia,'Times New Roman',Times,serif",
        heading_font = (
            "-apple-system,BlinkMacSystemFont,'Segoe UI',"
            "'Helvetica Neue',Arial,sans-serif"
        ),
        code_font = (
            "'SFMono-Regular','JetBrains Mono','Fira Code',"
            "Consolas,'Liberation Mono',monospace"
        ),

        # ── Type scale ────────────────────────────────────────────────────────
        h1_size         = "40px",
        h2_size         = "30px",
        h3_size         = "22px",
        h4_size         = "18px",
        body_size       = "18px",
        lead_size       = "20px",
        caption_size    = "14px",
        code_block_size = "14px",
        table_size      = "16px",

        # ── Spacing ───────────────────────────────────────────────────────────
        paragraph_spacing = "22px",
        heading_spacing   = "56px",
        h3_spacing        = "36px",
        h4_spacing        = "28px",
        image_spacing     = "40px",
        table_spacing     = "36px",
        list_indent       = "28px",
        list_item_spacing = "8px",

        # ── Colors ────────────────────────────────────────────────────────────
        primary           = "#0f172a",
        secondary         = "#1e293b",
        accent            = "#1e40af",
        muted_text        = "#64748b",
        border            = "#e2e8f0",
        table_header_bg   = "#0f172a",
        table_header_text = "#f8fafc",
        table_stripe      = "#f8fafc",
        external_link     = "#1d4ed8",
        internal_link     = "#0f172a",
        lead_text_color   = "#374151",
        code_bg           = "#0f172a",
        code_text         = "#e2e8f0",
        code_inline_bg    = "#f1f5f9",
        code_inline_text  = "#0f172a",

        # ── Images ────────────────────────────────────────────────────────────
        image_border_radius = "6px",
        image_shadow        = "0 1px 3px rgba(0,0,0,0.10)",

        # ── Component borders ─────────────────────────────────────────────────
        callout_border_radius      = "0 8px 8px 0",
        callout_font_size          = "17px",
        code_border_radius         = "8px",
        code_inline_border_radius  = "4px",
        table_wrapper_border_radius = "6px",

        # ── Feature flags ─────────────────────────────────────────────────────
        table_striped          = True,
        table_responsive       = True,
        external_links_new_tab = True,

        # ── Link decoration ───────────────────────────────────────────────────
        external_link_decoration = "underline",
        internal_link_decoration = "underline",
        internal_link_weight     = "600",

        # ── Callout variants ──────────────────────────────────────────────────
        callout_variants = {
            "💡": CalloutVariant("#eff6ff", "#3b82f6"),   # tip — blue
            "⚠️": CalloutVariant("#fffbeb", "#f59e0b"),   # warning — amber
            "🚨": CalloutVariant("#fef2f2", "#ef4444"),   # alert — red
            "❌": CalloutVariant("#fef2f2", "#ef4444"),   # error — red
            "✅": CalloutVariant("#f0fdf4", "#22c55e"),   # success — green
            "📝": CalloutVariant("#f8fafc", "#94a3b8"),   # note — slate
            "🔑": CalloutVariant("#fdf4ff", "#a855f7"),   # key point — purple
            "💰": CalloutVariant("#fefce8", "#ca8a04"),   # cost note — yellow
            "🛠": CalloutVariant("#fff7ed", "#f97316"),   # pro tip — orange
            "📌": CalloutVariant("#f0f9ff", "#0284c7"),   # pinned — sky
        },
        callout_default = CalloutVariant("#fffbeb", "#f59e0b"),

        # ── Reserved ──────────────────────────────────────────────────────────
        content_max_width = "none",
    )

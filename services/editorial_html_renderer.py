"""
EditorialHTMLRenderer — the single owner of all HTML rendering logic.

The renderer knows HOW to render. The EditorialTheme knows WHAT it should look like.

To change visual appearance, supply a different theme — do not modify this file.
To change rendering behavior (new passes, new HTML structure), modify this file.

Public interface::

    renderer = EditorialHTMLRenderer()                             # DefaultEditorialTheme
    renderer = EditorialHTMLRenderer(theme=OverheadDoorTheme())   # custom theme

    html = renderer.render(markdown, site_url="https://example.com")
    figure_html = EditorialHTMLRenderer.render_figure(req, meta)  # classmethod

render_figure() is a classmethod so PublisherAgent._inject_images() can call it
without constructing a renderer instance. It accepts an optional theme kwarg.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.image_request import ImageRequest
    from models.media import ImageMetadata
    from services.editorial_theme import EditorialTheme


# ── Theme-derived CSS cache ───────────────────────────────────────────────────

class _ThemeStyles:
    """
    Pre-computed inline CSS strings derived from one EditorialTheme instance.

    Instantiated once per render() call so the CSS string-building cost is paid
    once, not per regex match. Each _pass_* method receives this object and
    reads from it rather than constructing strings on each callback invocation.

    self.theme is the full theme object, available for callout variant lookups
    and feature-flag checks that can't be reduced to a single CSS string.
    """

    __slots__ = (
        "theme",
        "h1", "h2", "h3", "h4",
        "p", "p_lead", "li", "ul", "ol", "strong",
        "code_inline", "pre", "pre_code",
        "table", "table_wrap", "th", "td", "tr_odd", "tr_even",
        "figure", "img", "figcaption",
        "link_ext", "link_int",
    )

    def __init__(self, theme: "EditorialTheme") -> None:
        self.theme = theme
        t = theme

        def _css(**kw: str) -> str:
            return ";".join(
                f"{k.replace('_', '-')}:{v}" for k, v in kw.items() if v
            )

        self.h1 = _css(
            font_family=t.heading_font, font_size=t.h1_size,
            font_weight="800", line_height="1.15", letter_spacing="-0.5px",
            color=t.primary, margin_top="0", margin_bottom="28px",
        )
        self.h2 = _css(
            font_family=t.heading_font, font_size=t.h2_size,
            font_weight="700", line_height="1.25", letter_spacing="-0.25px",
            color=t.primary, margin_top=t.heading_spacing, margin_bottom="20px",
            padding_left="16px", border_left=f"3px solid {t.accent}",
        )
        self.h3 = _css(
            font_family=t.heading_font, font_size=t.h3_size,
            font_weight="700", line_height="1.35", color=t.primary,
            margin_top=t.h3_spacing, margin_bottom="14px",
        )
        self.h4 = _css(
            font_family=t.heading_font, font_size=t.h4_size,
            font_weight="700", line_height="1.4", color=t.primary,
            margin_top=t.h4_spacing, margin_bottom="10px",
        )
        self.p = _css(
            font_family=t.body_font, font_size=t.body_size,
            line_height="1.85", color=t.secondary,
            margin_top="0", margin_bottom=t.paragraph_spacing,
        )
        self.p_lead = _css(
            font_family=t.body_font, font_size=t.lead_size,
            line_height="1.8", color=t.lead_text_color,
            font_weight="400", margin_top="0", margin_bottom="28px",
        )
        self.li = _css(
            font_family=t.body_font, font_size=t.body_size,
            line_height="1.8", color=t.secondary,
            margin_bottom=t.list_item_spacing,
        )
        self.ul = _css(
            padding_left=t.list_indent, margin_top="4px", margin_bottom="24px",
        )
        self.ol = _css(
            padding_left=t.list_indent, margin_top="4px", margin_bottom="24px",
        )
        self.strong = _css(font_weight="700", color=t.primary)
        self.code_inline = _css(
            font_family=t.code_font, font_size="0.875em",
            background=t.code_inline_bg, color=t.code_inline_text,
            padding="2px 6px", border_radius=t.code_inline_border_radius,
        )
        # Vendor-prefixed properties can't use keyword args — build manually
        self.pre = (
            f"background:{t.code_bg};color:{t.code_text};"
            f"overflow-x:auto;padding:20px 24px;"
            f"border-radius:{t.code_border_radius};"
            f"margin:32px 0;font-family:{t.code_font};"
            f"font-size:{t.code_block_size};line-height:1.7;white-space:pre;"
        )
        self.pre_code = (
            "background:none;padding:0;border-radius:0;"
            "font-size:inherit;color:inherit;"
        )
        self.table = _css(
            width="100%", border_collapse="collapse",
            font_family=t.heading_font, font_size=t.table_size, line_height="1.6",
        )
        if t.table_responsive:
            self.table_wrap = (
                f"overflow-x:auto;margin:{t.table_spacing} 0 {t.table_spacing};"
                f"-webkit-overflow-scrolling:touch;"
                f"border-radius:{t.table_wrapper_border_radius};"
                f"border:1px solid {t.border};"
            )
        else:
            self.table_wrap = f"margin:{t.table_spacing} 0;"
        self.th = _css(
            padding="13px 16px", text_align="left", font_weight="700",
            background=t.table_header_bg, color=t.table_header_text,
            border_bottom=f"2px solid {t.accent}", white_space="nowrap",
        )
        self.td = _css(
            padding="12px 16px", border_bottom=f"1px solid {t.border}",
            vertical_align="top", color=t.secondary,
        )
        self.tr_odd  = "background:#ffffff;"
        self.tr_even = f"background:{t.table_stripe};"
        self.figure  = f"margin:{t.image_spacing} 0 36px;display:block;"
        self.img = (
            f"width:100%;height:auto;"
            f"border-radius:{t.image_border_radius};"
            f"display:block;box-shadow:{t.image_shadow};"
        )
        self.figcaption = _css(
            font_family=t.heading_font, font_size=t.caption_size,
            line_height="1.6", color=t.muted_text,
            margin_top="10px", font_style="italic", text_align="center",
        )
        self.link_ext = _css(
            color=t.external_link,
            text_decoration=t.external_link_decoration,
            text_underline_offset="3px",
        )
        self.link_int = _css(
            color=t.internal_link,
            text_decoration=t.internal_link_decoration,
            text_underline_offset="3px",
            font_weight=t.internal_link_weight,
            text_decoration_color=t.accent,
        )


# ── Renderer ──────────────────────────────────────────────────────────────────

class EditorialHTMLRenderer:
    """
    Transforms Markdown into production-quality editorial HTML.

    Rendering logic only — no presentation values. All visual decisions come
    from the EditorialTheme supplied at construction time. The renderer is
    stateless across render() calls (theme is fixed; no mutable state).

    Multi-pass pipeline — each pass owns one structural concern:
      1. callouts   — blockquote → icon-led callout box (emoji-aware variant)
      2. tables     — scroll wrapper + striped rows + column styling
      3. code       — <pre> blocks and inline <code>
      4. typography — h1–h4, p (lead detection), li, ul, ol, strong
      5. links      — external (target+rel) / internal (weighted underline)
      6. img        — lazy loading on any <img> from Markdown source

    render_figure() is a classmethod called by PublisherAgent._inject_images()
    to build figure HTML for uploaded images.
    """

    def __init__(self, theme: "EditorialTheme | None" = None) -> None:
        if theme is None:
            from services.editorial_theme import DefaultEditorialTheme
            theme = DefaultEditorialTheme()
        self._theme = theme

    def render(self, markdown: str, *, site_url: str = "") -> str:
        """
        Convert Markdown to production editorial HTML using the instance theme.

        Args:
            markdown:  Article body in Markdown (H1 title already stripped).
            site_url:  WordPress site base URL for internal vs external link
                       detection. Empty string treats all links as external.

        Returns:
            Self-contained inline-styled HTML ready for the WP REST API
            ``content`` field.
        """
        import markdown as _md

        styles = _ThemeStyles(self._theme)
        html = _md.markdown(markdown, extensions=["extra"])
        html = self._pass_callouts(html, styles)
        html = self._pass_tables(html, styles)
        html = self._pass_code(html, styles)
        html = self._pass_typography(html, styles)
        html = self._pass_links(html, site_url, styles)
        html = self._pass_inline_images(html)
        return html

    @classmethod
    def render_figure(
        cls,
        req: "ImageRequest",
        meta: "ImageMetadata",
        *,
        theme: "EditorialTheme | None" = None,
    ) -> str:
        """
        Build the <figure> HTML block for an uploaded inline image.

        Called by PublisherAgent._inject_images() after WordPress media upload.
        Accepts an optional theme; uses DefaultEditorialTheme when omitted so
        existing callers require no changes.

        Args:
            req:   ImageRequest describing alt text, caption, purpose, etc.
            meta:  ImageMetadata with the uploaded URL, dimensions, etc.
            theme: Optional theme override. Defaults to DefaultEditorialTheme.
        """
        if theme is None:
            from services.editorial_theme import DefaultEditorialTheme
            theme = DefaultEditorialTheme()
        styles = _ThemeStyles(theme)

        attrs = (
            f'src="{meta.url}" alt="{req.alt_text}" '
            f'loading="lazy" decoding="async" style="{styles.img}"'
        )
        if meta.width:
            attrs += f' width="{meta.width}"'
        if meta.height:
            attrs += f' height="{meta.height}"'
        img_tag = f'<img {attrs} />'

        if req.caption:
            return (
                f'<figure class="wp-block-image size-full" style="{styles.figure}">'
                f'{img_tag}'
                f'<figcaption class="wp-element-caption" style="{styles.figcaption}">'
                f'{req.caption}'
                f'</figcaption>'
                f'</figure>'
            )
        return (
            f'<figure class="wp-block-image size-full" style="{styles.figure}">'
            f'{img_tag}'
            f'</figure>'
        )

    # ── Pass 1: Callouts ──────────────────────────────────────────────────────

    def _pass_callouts(self, html: str, styles: _ThemeStyles) -> str:
        """
        Convert <blockquote> blocks to styled editorial callout boxes.

        The Python markdown library merges consecutive blockquotes (separated by
        blank lines) into one <blockquote> with multiple <p> children. This pass
        splits each child <p> into its own callout div so that:

            > 💡 tip text        → blue callout box
            > ⚠️ warning text    → amber callout box  (two separate boxes)

        The leading emoji of each paragraph selects the callout variant from
        theme.callout_variants. Paragraphs with no recognized emoji receive the
        theme.callout_default variant.

        The class "wp-block-callout" is preserved for analyze_html() compatibility.
        """
        t = styles.theme

        def _callout_for_paragraph(para_html: str) -> str:
            text_m = re.search(r'<p[^>]*>(.*?)(?:</p>|$)', para_html, re.DOTALL)
            inner_text = text_m.group(1).strip() if text_m else para_html

            variant = t.callout_default
            detected_icon = "💡"
            for emoji, v in t.callout_variants.items():
                if inner_text.startswith(emoji):
                    variant = v
                    detected_icon = emoji
                    break

            callout_style = (
                f"display:flex;align-items:flex-start;gap:14px;"
                f"border-left:4px solid {variant.border_color};"
                f"padding:20px 24px;background:{variant.background};"
                f"margin:40px 0 32px;border-radius:{t.callout_border_radius};"
                f"font-family:{t.heading_font};"
                f"font-size:{t.callout_font_size};line-height:1.7;"
            )
            icon_style = "font-size:22px;flex-shrink:0;line-height:1;margin-top:2px;"
            body_style = f"flex:1;color:{t.secondary};"

            return (
                f'<div class="wp-block-callout" style="{callout_style}">'
                f'<div style="{icon_style}">{detected_icon}</div>'
                f'<div style="{body_style}">{para_html}</div>'
                f'</div>'
            )

        def _build_callouts(m: re.Match) -> str:
            inner = m.group(1).strip()
            paragraphs = re.findall(r'<p[^>]*>.*?</p>', inner, re.DOTALL)
            if not paragraphs:
                return _callout_for_paragraph(inner)
            return "\n".join(_callout_for_paragraph(p) for p in paragraphs)

        return re.sub(
            r'<blockquote>(.*?)</blockquote>',
            _build_callouts,
            html,
            flags=re.DOTALL,
        )

    # ── Pass 2: Tables ────────────────────────────────────────────────────────

    def _pass_tables(self, html: str, styles: _ThemeStyles) -> str:
        """
        Wrap tables in a responsive scroll container and apply full inline styling.

        - table_responsive (theme flag): wraps in overflow-x:auto div
        - table_striped (theme flag): alternates even row background
        - Header cells (<th>) always get the theme's table_header_bg / text colors
        - Data cells (<td>) get consistent padding and bottom border

        Zebra striping counts all <tr> elements — header rows are not special-cased
        because the explicit <th> background overrides the <tr> background.
        """
        t = styles.theme

        def _process_table(m: re.Match) -> str:
            tbl = m.group(0)

            tbl = re.sub(
                r'<th\b[^>]*>',
                f'<th style="{styles.th}">',
                tbl,
            )
            tbl = re.sub(
                r'<td\b[^>]*>',
                f'<td style="{styles.td}">',
                tbl,
            )

            if t.table_striped:
                row_counter = [0]
                def _stripe(rm: re.Match) -> str:
                    row_counter[0] += 1
                    bg = styles.tr_even if row_counter[0] % 2 == 0 else styles.tr_odd
                    return f'<tr style="{bg}">'
                tbl = re.sub(r'<tr\b[^>]*>', _stripe, tbl)

            tbl = re.sub(
                r'<table\b[^>]*>',
                f'<table class="wp-block-table" style="{styles.table}">',
                tbl,
            )
            return f'<div style="{styles.table_wrap}">{tbl}</div>'

        return re.sub(
            r'<table\b[^>]*>.*?</table>',
            _process_table,
            html,
            flags=re.DOTALL,
        )

    # ── Pass 3: Code ──────────────────────────────────────────────────────────

    def _pass_code(self, html: str, styles: _ThemeStyles) -> str:
        """
        Style fenced code blocks (<pre><code>) and inline <code> elements.

        The pre pass runs first; any remaining <code> elements without a style
        attribute are inline (the fenced-code pass already added style= to pre's
        inner <code>), so no lookbehind is needed.
        """
        # Fenced blocks
        html = re.sub(
            r'<pre\b[^>]*>(\s*<code\b[^>]*>)(.*?)(</code>)\s*</pre>',
            lambda m: (
                f'<pre style="{styles.pre}">'
                f'<code style="{styles.pre_code}">'
                f'{m.group(2)}'
                f'</code></pre>'
            ),
            html,
            flags=re.DOTALL,
        )
        # Inline code — any <code> still lacking style= is inline
        html = re.sub(
            r'<code\b(?![^>]*style=)[^>]*>',
            f'<code style="{styles.code_inline}">',
            html,
        )
        return html

    # ── Pass 4: Typography ────────────────────────────────────────────────────

    def _pass_typography(self, html: str, styles: _ThemeStyles) -> str:
        """
        Apply editorial typography: heading hierarchy, paragraph rhythm, lists.

        Lead paragraph: the first <p> (excluding figcaptions) receives the
        theme's lead style (larger, softer color) to create a visual entry point.
        All subsequent paragraphs get the standard body style.

        Headings H1–H4 descend in size and weight. H2 carries a left accent bar
        (accent color from theme) as the primary section separator signal.
        """
        def _add_style(tag: str, css: str) -> str:
            if 'style=' in tag:
                return tag
            close = tag.rindex('>')
            return tag[:close] + f' style="{css}"' + tag[close:]

        for tag, key in (("h1", styles.h1), ("h2", styles.h2),
                          ("h3", styles.h3), ("h4", styles.h4)):
            html = re.sub(
                rf'<{tag}\b[^>]*>',
                lambda m, css=key: _add_style(m.group(0), css),
                html,
            )

        _first_p = [True]

        def _style_p(m: re.Match) -> str:
            tag = m.group(0)
            if 'style=' in tag:
                return tag
            if _first_p[0]:
                _first_p[0] = False
                return _add_style(tag, styles.p_lead)
            return _add_style(tag, styles.p)

        html = re.sub(
            r'<p\b(?![^>]*class="wp-element-caption")[^>]*>',
            _style_p,
            html,
        )

        html = re.sub(
            r'<ul\b[^>]*>',
            lambda m: _add_style(m.group(0), styles.ul),
            html,
        )
        html = re.sub(
            r'<ol\b[^>]*>',
            lambda m: _add_style(m.group(0), styles.ol),
            html,
        )
        html = re.sub(
            r'<li\b[^>]*>',
            lambda m: _add_style(m.group(0), styles.li),
            html,
        )
        html = re.sub(
            r'<strong\b(?![^>]*style=)[^>]*>',
            lambda m: _add_style(m.group(0), styles.strong),
            html,
        )
        return html

    # ── Pass 5: Links ─────────────────────────────────────────────────────────

    def _pass_links(self, html: str, site_url: str, styles: _ThemeStyles) -> str:
        """
        Style and annotate hyperlinks.

        External links (different host from site_url): theme external_link color,
        optional target="_blank" (controlled by theme.external_links_new_tab).

        Internal links (same host, relative, or fragment): theme internal_link
        color with weighted underline in the accent color.

        Anchor-only (#fragment), mailto:, and tel: links are left unstyled.
        """
        from urllib.parse import urlparse

        site_host = ""
        if site_url:
            site_host = urlparse(site_url.rstrip("/")).netloc.lstrip("www.")

        t = styles.theme

        def _style_link(m: re.Match) -> str:
            tag = m.group(0)
            href_m = re.search(r'href="([^"]*)"', tag)
            if not href_m:
                return tag
            href = href_m.group(1)
            if not href or href.startswith(("#", "mailto:", "tel:")):
                return tag

            parsed = urlparse(href)
            is_external = (
                bool(parsed.scheme)
                and parsed.scheme in ("http", "https")
                and (not site_host or site_host not in parsed.netloc)
            )

            if 'style=' in tag:
                if is_external and t.external_links_new_tab and 'target=' not in tag:
                    return tag.rstrip('>') + ' target="_blank" rel="noopener noreferrer">'
                return tag

            if is_external:
                base = tag.rstrip('>')
                if t.external_links_new_tab and 'target=' not in base:
                    base += ' target="_blank" rel="noopener noreferrer"'
                return base + f' style="{styles.link_ext}">'
            else:
                return tag.rstrip('>') + f' style="{styles.link_int}">'

        return re.sub(r'<a\b[^>]*>', _style_link, html)

    # ── Pass 6: Inline images ─────────────────────────────────────────────────

    def _pass_inline_images(self, html: str) -> str:
        """
        Add loading="lazy" and decoding="async" to any <img> from Markdown source.

        Images injected by PublisherAgent._inject_images() already have these
        attributes from render_figure(); this pass is a safety net for any
        ![alt](url) images in the Markdown body.
        """
        def _add_lazy(m: re.Match) -> str:
            tag = m.group(0)
            if 'loading=' not in tag:
                tag = tag[:-2] + ' loading="lazy"' + tag[-2:]
            if 'decoding=' not in tag:
                tag = tag[:-2] + ' decoding="async"' + tag[-2:]
            return tag

        return re.sub(r'<img\b[^>]*/>', _add_lazy, html)

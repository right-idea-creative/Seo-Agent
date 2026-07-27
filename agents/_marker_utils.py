"""
Shared marker utilities for SEO_AGENT_IMAGE comment handling.

Used by both ImageResolverAgent (planning phase) and DualQAAgent (revision
integrity checks). Neither module owns the insertion logic; it lives here
as the single source of truth.
"""
from __future__ import annotations

import re


def insert_marker_at_section(
    markdown: str,
    marker: str,
    section_title: str | None,
) -> str:
    """
    Insert an image marker before the best-matching heading in the markdown.

    Uses words longer than 3 characters from section_title to find the nearest
    matching H1/H2/H3 heading. Falls back to appending at the end when no heading
    matches — the marker will be visible in the article but at least the image
    won't silently vanish.
    """
    if section_title:
        ref_words = [w for w in re.split(r'\W+', section_title.lower()) if len(w) > 3]
        if ref_words:
            lines = markdown.splitlines(keepends=True)
            for i, line in enumerate(lines):
                m = re.match(r'^#{1,3}\s+(.+)', line)
                if m and any(w in m.group(1).lower() for w in ref_words):
                    lines.insert(i, f'{marker}\n\n')
                    return ''.join(lines)
    return markdown.rstrip() + f'\n\n{marker}\n'

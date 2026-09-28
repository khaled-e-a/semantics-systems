"""Assemble the final markdown post from an LLM-written PostDraft.

The "**[Title](url)** — " prefix of each paper paragraph is built here from
the parsed digest report, never by the LLM, so every link is exact by
construction. An optional image (the paper's main figure) goes directly
above its paragraph.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import DigestEntry, PostDraft

PREFIX_SEPARATOR = " — "


def paper_prefix(entry: DigestEntry) -> str:
    """The Python-built lead-in of a paper paragraph: ``**[Title](url)** — ``."""
    return f"**[{entry.title}]({entry.url})**{PREFIX_SEPARATOR}"


def render_markdown(
    post: PostDraft,
    keep_entries: List[DigestEntry],
    images: Optional[Dict[str, Tuple[str, str]]] = None,
) -> str:
    """Render the post as plain markdown (no frontmatter).

    ``images`` maps paper URL -> (image src, caption). Papers appear in the
    given ``keep_entries`` order.
    """
    images = images or {}
    out = f"# {post.title}\n\n{post.intro}\n\n"
    for entry in keep_entries:
        if entry.url in images:
            src, caption = images[entry.url]
            out += f"![{caption}]({src})\n\n"
        out += f"{paper_prefix(entry)}{post.bodies.get(entry.url, '')}\n\n"
    return out.rstrip() + "\n"

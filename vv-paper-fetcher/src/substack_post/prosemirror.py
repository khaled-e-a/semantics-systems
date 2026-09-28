"""Convert the post markdown our pipeline renders into Substack's ProseMirror doc JSON.

This is deliberately NOT a general markdown parser. It understands only what
write_post.py produces:

- blocks separated by blank lines;
- ``## text`` / ``### text`` headings (level 1 is reserved for the post title,
  which goes in ``draft_title``; a stray ``# `` in the body becomes level 2);
- a line that is exactly ``![caption](src)`` → ``captionedImage > image2 + caption``
  (bare ``image2``/``image`` nodes are silently dropped by Substack);
- everything else → a paragraph with inline ``**strong**`` and ``[text](url)``
  links, which nest, so ``**[Title](url)**`` gives one text node carrying both
  the strong and the link marks. Newlines inside a paragraph become one space.

Standard markdown backslash escapes (``\\*``, ``\\[`` …) are honoured in text.
The caller ``json.dumps`` the returned doc into ``draft_body``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from .models import UploadedImage

logger = logging.getLogger(__name__)

# Substack's editor column is 728 px wide; wider images are displayed resized to it.
EDITOR_WIDTH_PX = 728

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
# Greedy caption so a caption containing "](" still anchors on the final "](src)".
_IMAGE_RE = re.compile(r"^!\[(.*)\]\((\S+)\)$")
_INLINE_RE = re.compile(
    # **strong** (not preceded by a backslash escape); content may hold a link.
    r"(?<!\\)\*\*(?P<strong>.+?)(?<!\\)\*\*"
    # [text](href): text may contain escapes or one level of balanced brackets
    # (e.g. "[Re] Title"); href may contain one level of balanced parentheses.
    r"|(?<!\\)\[(?P<ltext>(?:\\.|[^\[\]\\]|\[[^\[\]]*\])+)\]"
    r"\((?P<href>(?:[^()\s]|\([^()\s]*\))+)\)"
)
_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_MARK_ORDER = {"strong": 0, "em": 1, "code": 2, "link": 3}


def split_title(markdown: str) -> Tuple[str, str]:
    """Return ``(title, body_markdown)``.

    The title comes from a leading ``# `` line (leading blank lines allowed);
    the body is everything after it with leading blank lines removed. Without a
    leading ``# `` line the result is ``("", markdown)`` unchanged.
    """
    lines = markdown.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx == len(lines) or not lines[idx].startswith("# "):
        return "", markdown

    title = lines[idx][2:].strip()
    rest = lines[idx + 1 :]
    while rest and not rest[0].strip():
        rest = rest[1:]
    return title, "\n".join(rest)


def markdown_to_doc(markdown: str, images: Dict[str, UploadedImage]) -> Dict[str, Any]:
    """Build ``{"type": "doc", "content": [...]}`` from body markdown.

    ``images`` maps each ``![caption](src)`` ``src`` to its already-uploaded
    image; an unknown ``src`` raises ``KeyError`` (upload before building).
    """
    content: List[Dict[str, Any]] = []
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    for block in re.split(r"\n[ \t]*\n", text):
        paragraph_lines: List[str] = []
        for raw_line in block.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            heading = _HEADING_RE.match(line)
            image = _IMAGE_RE.match(line)
            if heading or image:
                _flush_paragraph(paragraph_lines, content)
                if heading:
                    content.append(_heading_node(len(heading.group(1)), heading.group(2)))
                else:
                    content.append(_captioned_image_node(image.group(1), image.group(2), images))
            else:
                paragraph_lines.append(line)
        _flush_paragraph(paragraph_lines, content)

    if not content:
        content.append({"type": "paragraph"})
    return {"type": "doc", "content": content}


def _flush_paragraph(lines: List[str], content: List[Dict[str, Any]]) -> None:
    if not lines:
        return
    inline = _parse_inline(" ".join(lines), [])
    lines.clear()
    if inline:
        content.append({"type": "paragraph", "content": inline})


def _heading_node(hashes: int, text: str) -> Dict[str, Any]:
    node: Dict[str, Any] = {"type": "heading", "attrs": {"level": max(2, hashes)}}
    inline = _parse_inline(text, [])
    if inline:
        node["content"] = inline
    return node


def _captioned_image_node(caption: str, src: str, images: Dict[str, UploadedImage]) -> Dict[str, Any]:
    if src not in images:
        raise KeyError(f"image {src!r} has not been uploaded (no entry in images)")
    img = images[src]
    caption_inline = _parse_inline(caption, [])
    alt = "".join(node["text"] for node in caption_inline)
    width = int(img.width)

    caption_node: Dict[str, Any] = {"type": "caption"}
    if caption_inline:
        caption_node["content"] = caption_inline

    return {
        "type": "captionedImage",
        "content": [
            {
                "type": "image2",
                "attrs": {
                    "src": img.url,
                    "srcNoWatermark": None,
                    "fullscreen": False,
                    "imageSize": "normal",
                    "width": width,
                    "height": int(img.height),
                    "resizeWidth": min(width, EDITOR_WIDTH_PX) if width > 0 else EDITOR_WIDTH_PX,
                    "bytes": int(img.bytes),
                    "alt": alt or None,
                    "title": None,
                    "type": img.content_type,
                    "href": None,
                    "belowTheFold": False,
                    "topImage": False,
                    "internalRedirect": None,
                    "isProcessing": False,
                },
            },
            caption_node,
        ],
    }


def _parse_inline(text: str, marks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split ``text`` into ProseMirror text nodes, each carrying ``marks`` plus its own."""
    nodes: List[Dict[str, Any]] = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        _append_text(nodes, text[pos : match.start()], marks)
        if match.group("strong") is not None:
            inner_marks = marks + [{"type": "strong"}]
            for node in _parse_inline(match.group("strong"), inner_marks):
                _append_node(nodes, node)
        else:
            link = {"type": "link", "attrs": {"href": match.group("href")}}
            for node in _parse_inline(match.group("ltext"), marks + [link]):
                _append_node(nodes, node)
        pos = match.end()
    _append_text(nodes, text[pos:], marks)
    return nodes


def _append_text(nodes: List[Dict[str, Any]], raw: str, marks: List[Dict[str, Any]]) -> None:
    text = _ESCAPE_RE.sub(r"\1", raw)
    if not text:
        return  # ProseMirror rejects empty text nodes
    node: Dict[str, Any] = {"type": "text", "text": text}
    if marks:
        node["marks"] = sorted((dict(m) for m in marks), key=lambda m: _MARK_ORDER.get(m["type"], 99))
    _append_node(nodes, node)


def _append_node(nodes: List[Dict[str, Any]], node: Dict[str, Any]) -> None:
    """Append, merging into the previous text node when the marks are identical."""
    if nodes and nodes[-1].get("marks") == node.get("marks"):
        nodes[-1] = {**nodes[-1], "text": nodes[-1]["text"] + node["text"]}
    else:
        nodes.append(node)

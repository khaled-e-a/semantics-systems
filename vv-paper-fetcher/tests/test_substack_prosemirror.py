import json

import pytest

from src.substack_post.models import UploadedImage
from src.substack_post.prosemirror import markdown_to_doc, split_title

IMAGE = UploadedImage(
    url="https://substack-post-media.s3.amazonaws.com/public/images/abc_1200x800.png",
    width=1200,
    height=800,
    bytes=123456,
    content_type="image/png",
)
CAPTION = "Figure 1 from “T” (arXiv:1)"
PAPER_URL = "https://arxiv.org/abs/2609.28614v1"
BODY = (
    "This week's intro.\n\n"
    f"![{CAPTION}](fig-1.png)\n\n"
    f"**[Title]({PAPER_URL})** — body text with **bold** and a [link](https://x)."
)


# ---- split_title -----------------------------------------------------------


def test_split_title_takes_leading_h1_and_strips_blank_lines():
    title, body = split_title("# 15 Ways to Verify AI\n\n\nIntro paragraph.\n\nSecond.")
    assert title == "15 Ways to Verify AI"
    assert body == "Intro paragraph.\n\nSecond."


def test_split_title_allows_leading_blank_lines_and_trims_title():
    title, body = split_title("\n\n#   Spaced title   \nIntro.")
    assert title == "Spaced title"
    assert body == "Intro."


def test_split_title_without_h1_returns_markdown_unchanged():
    md = "## Not a title\n\nIntro."
    assert split_title(md) == ("", md)
    assert split_title("Intro only.") == ("", "Intro only.")


# ---- markdown_to_doc -------------------------------------------------------


def test_renderer_format_produces_expected_tree():
    doc = markdown_to_doc(BODY, {"fig-1.png": IMAGE})
    assert doc["type"] == "doc"
    intro, figure, paper = doc["content"]

    assert intro == {"type": "paragraph", "content": [{"type": "text", "text": "This week's intro."}]}

    assert figure["type"] == "captionedImage"
    image2, caption = figure["content"]
    assert image2["type"] == "image2"
    attrs = image2["attrs"]
    assert attrs["src"] == IMAGE.url
    assert (attrs["width"], attrs["height"], attrs["bytes"]) == (1200, 800, 123456)
    assert attrs["type"] == "image/png"
    assert attrs["resizeWidth"] == 728
    assert attrs["alt"] == CAPTION
    assert attrs["imageSize"] == "normal"
    assert attrs["srcNoWatermark"] is None and attrs["href"] is None
    assert attrs["fullscreen"] is False and attrs["isProcessing"] is False
    assert caption == {"type": "caption", "content": [{"type": "text", "text": CAPTION}]}

    assert paper["type"] == "paragraph"
    assert paper["content"] == [
        {
            "type": "text",
            "text": "Title",
            "marks": [{"type": "strong"}, {"type": "link", "attrs": {"href": PAPER_URL}}],
        },
        {"type": "text", "text": " — body text with "},
        {"type": "text", "text": "bold", "marks": [{"type": "strong"}]},
        {"type": "text", "text": " and a "},
        {"type": "text", "text": "link", "marks": [{"type": "link", "attrs": {"href": "https://x"}}]},
        {"type": "text", "text": "."},
    ]


def test_small_image_is_not_upscaled():
    small = UploadedImage(url="https://cdn/x_300x200.png", width=300, height=200, bytes=10, content_type="image/png")
    doc = markdown_to_doc("![c](s.png)", {"s.png": small})
    assert doc["content"][0]["content"][0]["attrs"]["resizeWidth"] == 300


def test_unknown_image_src_raises_key_error():
    with pytest.raises(KeyError):
        markdown_to_doc("![caption](missing.png)", {"fig-1.png": IMAGE})


def test_heading_levels():
    doc = markdown_to_doc("## Section\n\n### Sub **bold**\n\n# Stray h1\n\nText", {})
    h2, h3, stray, para = doc["content"]
    assert h2 == {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": "Section"}]}
    assert h3["attrs"] == {"level": 3}
    assert h3["content"] == [
        {"type": "text", "text": "Sub "},
        {"type": "text", "text": "bold", "marks": [{"type": "strong"}]},
    ]
    assert stray["attrs"] == {"level": 2}  # level 1 is reserved for draft_title
    assert para["type"] == "paragraph"


def test_newlines_inside_paragraph_become_single_space():
    doc = markdown_to_doc("line one\nline two\r\n  line three", {})
    assert doc["content"] == [
        {"type": "paragraph", "content": [{"type": "text", "text": "line one line two line three"}]}
    ]


def test_image_line_without_blank_separator_still_becomes_block():
    doc = markdown_to_doc(f"Intro.\n![{CAPTION}](fig-1.png)\n**[T]({PAPER_URL})** — body", {"fig-1.png": IMAGE})
    assert [n["type"] for n in doc["content"]] == ["paragraph", "captionedImage", "paragraph"]


def test_title_with_brackets_and_escapes():
    md = r"**[[Re] A \*star\* study](https://arxiv.org/abs/1)** — see 2 \* 3."
    content = markdown_to_doc(md, {})["content"][0]["content"]
    assert content[0]["text"] == "[Re] A *star* study"
    assert content[0]["marks"][1] == {"type": "link", "attrs": {"href": "https://arxiv.org/abs/1"}}
    assert content[1] == {"type": "text", "text": " — see 2 * 3."}


def test_unmatched_markup_stays_literal():
    content = markdown_to_doc("a ** b [c] (d) [e]", {})["content"][0]["content"]
    assert content == [{"type": "text", "text": "a ** b [c] (d) [e]"}]


def test_link_href_with_parentheses():
    content = markdown_to_doc("[wiki](https://en.wikipedia.org/wiki/A_(b))", {})["content"][0]["content"]
    assert content[0]["marks"][0]["attrs"]["href"] == "https://en.wikipedia.org/wiki/A_(b)"


def test_empty_markdown_gives_single_empty_paragraph():
    assert markdown_to_doc("\n\n", {}) == {"type": "doc", "content": [{"type": "paragraph"}]}


def test_no_empty_text_nodes_anywhere():
    doc = markdown_to_doc(f"**[T]({PAPER_URL})**\n\n**x****y**", {})

    def walk(node):
        if node.get("type") == "text":
            assert node["text"]
        for child in node.get("content", []):
            walk(child)

    walk(doc)


def test_json_round_trip():
    doc = markdown_to_doc(BODY, {"fig-1.png": IMAGE})
    encoded = json.dumps(doc)
    assert isinstance(encoded, str)
    assert json.loads(encoded) == doc

"""render.render_markdown: exact output format."""
import re

from src.config import PROJECT_ROOT
from src.substack_post.models import DigestEntry, PostDraft
from src.substack_post.render import render_markdown

A = DigestEntry(rank=1, section="top", title="Paper A", url="http://arxiv.org/abs/2609.00001v1")
B = DigestEntry(rank=5, section="honorable", title="Paper B", url="https://huggingface.co/papers/2609.00002")


def _post() -> PostDraft:
    return PostDraft(
        title="Two Ways to Check",
        intro="Intro text.",
        bodies={A.url: "Body A.", B.url: "Body B."},
    )


def test_render_without_images():
    expected = (
        "# Two Ways to Check\n\n"
        "Intro text.\n\n"
        "**[Paper A](http://arxiv.org/abs/2609.00001v1)** — Body A.\n\n"
        "**[Paper B](https://huggingface.co/papers/2609.00002)** — Body B.\n"
    )
    assert render_markdown(_post(), [A, B]) == expected
    assert render_markdown(_post(), [A, B], images={}) == expected


def test_render_with_images_only_above_papers_that_have_one():
    images = {B.url: ("https://cdn.example/fig1.png", "Figure 1 from “Paper B” (arXiv:2609.00002)")}
    expected = (
        "# Two Ways to Check\n\n"
        "Intro text.\n\n"
        "**[Paper A](http://arxiv.org/abs/2609.00001v1)** — Body A.\n\n"
        "![Figure 1 from “Paper B” (arXiv:2609.00002)](https://cdn.example/fig1.png)\n\n"
        "**[Paper B](https://huggingface.co/papers/2609.00002)** — Body B.\n"
    )
    assert render_markdown(_post(), [A, B], images=images) == expected


def test_render_follows_keep_entry_order_and_ignores_extra_bodies():
    post = _post()
    post.bodies["http://arxiv.org/abs/2609.99999v1"] = "Not a KEEP paper."
    out = render_markdown(post, [B, A])
    assert out.index("Paper B") < out.index("Paper A")
    assert "Not a KEEP paper" not in out


def test_render_strips_trailing_whitespace_and_ends_with_one_newline():
    post = PostDraft(title="One Way", intro="Intro.", bodies={A.url: "Body A.  \n\n\n"})
    out = render_markdown(post, [A])
    assert out.endswith("Body A.\n")
    assert not out.endswith("\n\n")


def test_render_reproduces_the_approved_post_exactly():
    path = PROJECT_ROOT / "reports" / "2026-09-07-substack-post-ste100.md"
    text = path.read_text(encoding="utf-8")
    paragraphs = re.split(r"\n\s*\n", text.strip())
    entries, bodies = [], {}
    for rank, paragraph in enumerate(paragraphs[2:], start=1):
        m = re.match(r"^\*\*\[(?P<t>.+?)\]\((?P<u>\S+?)\)\*\* — (?P<b>.*)$", paragraph, re.DOTALL)
        entries.append(DigestEntry(rank=rank, section="top", title=m["t"], url=m["u"]))
        bodies[m["u"]] = m["b"]
    post = PostDraft(title=paragraphs[0][2:], intro=paragraphs[1], bodies=bodies)
    assert render_markdown(post, entries) == text

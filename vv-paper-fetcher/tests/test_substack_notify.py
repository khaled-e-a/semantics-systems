from pathlib import Path

from src.substack_post import notify
from src.substack_post.models import DigestEntry, DraftResult, Figure, Verdict, VerifyResult

DATE = "2026-09-28"
EDIT_URL = "https://example.substack.com/publish/post/101"


def _entry(rank, title=None):
    return DigestEntry(rank=rank, section="top", title=title or f"Paper {rank}", url=f"http://arxiv.org/abs/2609.0000{rank}v1", arxiv_id=f"2609.0000{rank}")


def _fig(entry, openly_licensed=True):
    return Figure(
        url=entry.url,
        path=Path(f"/tmp/{entry.arxiv_id}.png"),
        figure_number="2",
        credit_caption=f"Figure 2 from “{entry.title}” (arXiv:{entry.arxiv_id})",
        source="pdf",
        openly_licensed=openly_licensed,
        width=800,
        height=600,
        mime="image/png",
    )


A = _entry(1, "Calibrating <Agents> & Friends")
B = _entry(2)
C = _entry(3)
ENTRIES = [A, B, C]
VERDICTS = {
    A.url: Verdict(A.url, True, "keeps", "S", "uncertainty-quantification"),
    B.url: Verdict(B.url, True, "keeps", "S", "validation"),
    C.url: Verdict(C.url, False, "Benchmark only; no <verification> method."),
}
KEEP = [A, B]
FIGS = {A.url: _fig(A, openly_licensed=False)}  # B has no figure


# ---------------------------------------------------------------- converter


def test_markdown_to_html_subset():
    md = (
        "# 2 Ways & Means\n\n"
        "Intro with **bold** and a [link](https://example.com/x?a=1&b=2).\n\n"
        "![Figure 1 from “T” (arXiv:2609.1)](fig-2609.1.png)\n\n"
        "**[Paper <One>](http://arxiv.org/abs/2609.1v1)** — Body text.\n"
    )
    out = notify.markdown_to_html(md)
    assert "<h1" in out and "2 Ways &amp; Means</h1>" in out
    assert "<strong>bold</strong>" in out
    assert 'href="https://example.com/x?a=1&amp;b=2"' in out
    assert "\N{PAPERCLIP} figure attached: <strong>fig-2609.1.png</strong> — Figure 1 from “T” (arXiv:2609.1)" in out
    assert '<a href="http://arxiv.org/abs/2609.1v1" style="color:#1a56db"><strong>Paper &lt;One&gt;</strong></a> — Body text.' in out
    assert out.count("<p") == 3  # intro, attachment note, paper paragraph


def test_markdown_to_html_cdn_image_and_nested_bold_link():
    out = notify.markdown_to_html(
        "![cap](https://substackcdn.com/img.png)\n\n**see [here](https://x.org)**"
    )
    assert '<img src="https://substackcdn.com/img.png" alt="cap"' in out
    assert "<strong>see <a href=\"https://x.org\"" in out


def test_markdown_to_html_escapes_and_refuses_non_http_links():
    out = notify.markdown_to_html("Hi <script>alert(1)</script> [x](javascript:alert(1))")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "javascript:" not in out.split("href=")[0] and 'href="javascript' not in out


# ---------------------------------------------------------------- success


def test_success_email():
    verify = VerifyResult(errors=["Author name found: <Ada>"], warnings=["Paragraph 2 has 230 words"])
    subject, html = notify.build_success_email(
        DATE, "2 Ways to <Check> AI", DraftResult(101, EDIT_URL, True), KEEP, ENTRIES, VERDICTS, FIGS, verify
    )
    assert subject == "Substack draft ready: 2 Ways to <Check> AI"
    assert f'href="{EDIT_URL}"' in html
    assert "2 Ways to &lt;Check&gt; AI" in html
    assert "Kept <strong>2</strong> of <strong>3</strong>" in html
    assert "Calibrating &lt;Agents&gt; &amp; Friends" in html
    assert "uncertainty-quantification" in html and "validation" in html
    assert "Figure 2 (from PDF)" in html
    assert "Figure not openly licensed" in html
    assert "No figure found" in html
    # verify errors come before the editor button
    assert html.index("Author name found: &lt;Ada&gt;") < html.index(EDIT_URL)
    assert "Paragraph 2 has 230 words" in html
    assert "<details" in html and "Benchmark only; no &lt;verification&gt; method." in html


def test_success_email_update_and_clean_verify():
    _, html = notify.build_success_email(
        DATE, "T", DraftResult(55, EDIT_URL, False), KEEP, ENTRIES, VERDICTS, {}, VerifyResult()
    )
    assert "updated" in html
    assert "still fail" not in html
    assert "Warnings" not in html


# ---------------------------------------------------------------- failure


def _failure_html(failure, verify=None):
    return notify.build_failure_email(
        DATE,
        failure,
        "# Title\n\nIntro.\n\n![Figure 2 from “A”](fig-2609.00001.png)\n\n**[A](http://arxiv.org/abs/2609.00001v1)** — Body.",
        KEEP,
        ENTRIES,
        VERDICTS,
        FIGS,
        verify or VerifyResult(),
        attachment_names={A.url: "fig-2609.00001.png"},
    )


def test_failure_email_auth_or_block():
    failure = notify.SubstackFailure(notify.AUTH_OR_BLOCK, "401 Unauthorized <html>")
    subject, html = _failure_html(failure, VerifyResult(errors=["Paragraph count 3 != 4"]))
    assert subject == f"Substack draft NOT created — {DATE}"
    assert "401 Unauthorized &lt;html&gt;" in html
    assert html.index("SUBSTACK_COOKIE") < html.index("SUBSTACK_USE_WARP")  # cookie first for a plain 401
    assert "SUBSTACK_PROXY" in html
    assert "Paragraph count 3 != 4" in html
    assert "attached as fig-2609.00001.png" in html
    assert "figure attached: <strong>fig-2609.00001.png</strong>" in html
    assert "Figure not openly licensed" in html and "No figure found" in html
    assert "Benchmark only" in html


def test_failure_hints_order_for_cloudflare_and_proxy():
    hints = notify.fix_hints(notify.SubstackFailure(notify.AUTH_OR_BLOCK, "Cloudflare challenge (Just a moment...)", proxy_in_use=True))
    assert "SUBSTACK_USE_WARP" in hints[0]
    assert "SUBSTACK_COOKIE" in hints[1]
    assert any("residential proxy" in h for h in hints)


def test_failure_hints_prefer_exception_kind_over_message():
    cookie = notify.fix_hints(notify.SubstackFailure(notify.AUTH_OR_BLOCK, "Cloudflare said no", block_kind="cookie"))
    assert "SUBSTACK_COOKIE" in cookie[0]
    cloudflare = notify.fix_hints(notify.SubstackFailure(notify.AUTH_OR_BLOCK, "403", block_kind="cloudflare"))
    assert "SUBSTACK_USE_WARP" in cloudflare[0]


def test_failure_hint_for_redirect():
    hints = notify.fix_hints(notify.SubstackFailure(notify.ERROR, "302 redirect", status_code=302))
    assert "canonical" in hints[0] and "SUBSTACK_PUBLICATION_URL" in hints[0]


def test_failure_hints_for_other_kinds():
    assert any("SUBSTACK_PUBLICATION_URL" in h for h in notify.fix_hints(notify.SubstackFailure(notify.NOT_CONFIGURED, "x")))
    assert "--skip-substack" in notify.fix_hints(notify.SubstackFailure(notify.SKIPPED, "x"))[0]
    updating = notify.fix_hints(notify.SubstackFailure(notify.ERROR, "404", updating=True))
    assert any("state/substack_drafts.json" in h for h in updating)


def test_writer_failure_email():
    subject, html = notify.build_writer_failure_email(
        DATE, "bad <json>", KEEP, ENTRIES, VERDICTS, FIGS, attachment_names={A.url: "fig-2609.00001.png"}
    )
    assert subject == f"Substack post NOT written — {DATE}"
    assert "post generation failed: bad &lt;json&gt;" in html
    assert "SUBSTACK_WRITER_MODEL" in html
    assert "Papers that passed the filter (2)" in html
    assert "attached as fig-2609.00001.png" in html
    assert "Benchmark only" in html


# ---------------------------------------------------------------- quiet


def test_quiet_email_for_quiet_week():
    subject, html = notify.build_quiet_email(DATE, [], {})
    assert subject == f"No Substack post this week — {DATE}"
    assert "quiet week" in html


def test_quiet_email_for_zero_keep_lists_drop_rationales():
    verdicts = {e.url: Verdict(e.url, False, f"why <{e.rank}>") for e in ENTRIES}
    del verdicts[C.url]  # a missing verdict still gets listed
    subject, html = notify.build_quiet_email(DATE, ENTRIES, verdicts)
    assert subject == f"No Substack post this week — {DATE}"
    assert "None of the 3 papers" in html
    assert "why &lt;1&gt;" in html and "why &lt;2&gt;" in html
    assert "no verdict returned" in html
    assert "Calibrating &lt;Agents&gt; &amp; Friends" in html

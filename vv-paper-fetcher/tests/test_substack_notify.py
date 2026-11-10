from pathlib import Path
from types import SimpleNamespace

from src.substack_post import notify
from src.substack_post.models import DigestEntry, DraftResult, Figure, Verdict, VerifyResult

DATE = "2026-09-28"
EDIT_URL = "https://example.substack.com/publish/post/101"
HOSTED = "https://raw.githubusercontent.com/khaled-e-a/semantics-systems/figures/2026-09-28"


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
        "![cap <1>](https://substackcdn.com/img.png)\n\n**see [here](https://x.org)**"
    )
    assert '<img src="https://substackcdn.com/img.png" alt="cap &lt;1&gt;" style="max-width:100%;height:auto;display:block">' in out
    # caption: a small italic paragraph right below the image, no wrapper element
    assert out.index("<img") < out.index("<em>cap &lt;1&gt;</em></p>")
    assert "<div" not in out and "<table" not in out
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
    misconfigured = notify.fix_hints(notify.SubstackFailure(notify.NOT_CONFIGURED, "x"))
    assert "SUBSTACK_PUBLICATION_URL" in misconfigured[0] and "SUBSTACK_SETUP.md" in misconfigured[0]
    assert "paste-ready" in misconfigured[0]  # deleting the secrets is a valid way out
    assert not hasattr(notify, "SKIPPED")  # --skip-substack is paste mode now, not a failure
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
    assert "could not be hosted" not in html


def test_writer_failure_email_links_hosted_figures():
    _, html = notify.build_writer_failure_email(
        DATE, "x", KEEP, ENTRIES, VERDICTS, FIGS, attachment_names={}, hosted_urls={A.url: f"{HOSTED}/fig-2609.00001.png"}
    )
    assert f'<a href="{HOSTED}/fig-2609.00001.png" style="color:#1a56db">image</a>' in html
    assert "attached as" not in html


def test_writer_failure_email_hosting_failed():
    _, html = notify.build_writer_failure_email(
        DATE, "x", KEEP, ENTRIES, VERDICTS, FIGS, attachment_names={A.url: "fig-2609.00001.png"}, hosting_error="push <rejected>"
    )
    assert "Figures could not be hosted (push &lt;rejected&gt;) — they're attached instead." in html
    assert "pasted images" not in html  # there is no post to paste
    assert "attached as fig-2609.00001.png" in html


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


# ---------------------------------------------------------------- paste-ready layout

HOSTED_MD = (
    "# 2 Ways to <Check> AI\n\n"
    "Intro with **bold**.\n\n"
    f"![Figure 2 from “Calibrating <Agents> & Friends” (arXiv:2609.00001)]({HOSTED}/fig-2609.00001.png)\n\n"
    "**[Calibrating <Agents> & Friends](http://arxiv.org/abs/2609.00001v1)** — Body A.\n\n"
    "**[Paper 2](http://arxiv.org/abs/2609.00002v1)** — Body B.\n"
)


def _split_copy_block(html):
    """(before the start marker, the copy block, after the end marker)."""
    start = html.index(notify.COPY_START)
    end = html.index(notify.COPY_END)
    assert start < end
    before, rest = html[:start], html[start + len(notify.COPY_START) :]
    block, after = rest.split(notify.COPY_END, 1)
    # the block runs from the rule below the start marker to the rule above the end marker
    block = block.split("<hr", 1)[1].split(">", 1)[1].rsplit("<hr", 1)[0]
    return before, block, after


def test_paste_email_layout():
    verify = VerifyResult(errors=["Link appears twice: <X>"], warnings=["Paragraph 2 has 230 words"])
    subject, html = notify.build_paste_email(DATE, HOSTED_MD, KEEP, ENTRIES, VERDICTS, FIGS, verify, attachment_names={})
    assert subject == "Substack post ready to paste: 2 Ways to <Check> AI"
    before, block, after = _split_copy_block(html)

    # Instructions first, then the title on its own, both outside the copy block.
    assert before.index(notify.PASTE_INSTRUCTIONS) < before.index("2 Ways to &lt;Check&gt; AI")
    assert "Title (paste it into the title field)" in before
    assert "2 Ways to &lt;Check&gt; AI" not in block and "<h1" not in html

    # Only the post body inside: intro, image, caption, paragraphs, in that order.
    img = f'<img src="{HOSTED}/fig-2609.00001.png" alt="Figure 2 from “Calibrating &lt;Agents&gt; &amp; Friends” (arXiv:2609.00001)" style="max-width:100%;height:auto;display:block">'
    caption = "<em>Figure 2 from “Calibrating &lt;Agents&gt; &amp; Friends” (arXiv:2609.00001)</em></p>"
    paper_link = '<a href="http://arxiv.org/abs/2609.00001v1" style="color:#1a56db"><strong>Calibrating &lt;Agents&gt; &amp; Friends</strong></a> — Body A.'
    assert block.index("Intro with <strong>bold</strong>.") < block.index(img) < block.index(caption) < block.index(paper_link)
    assert block.index(paper_link) < block.index("Body B.")
    for tag in ("<div", "<table", "<ul", "<details", "<hr", "<h1", "<h3"):
        assert tag not in block
    for note in ("not openly licensed", "No figure found", "Contribution", "Link appears twice", "figure attached", "Benchmark only"):
        assert note not in block

    # Review facts after the end marker, unchanged in content.
    assert "Kept <strong>2</strong> of <strong>3</strong>" in after
    assert "uncertainty-quantification" in after and "Figure 2 (from PDF)" in after
    assert "Figure not openly licensed" in after and "No figure found" in after
    assert "Link appears twice: &lt;X&gt;" in after and "Paragraph 2 has 230 words" in after
    assert "<details" in after and "Benchmark only; no &lt;verification&gt; method." in after
    assert "1 check(s) still fail" in before  # pointer to them above the post

    # Paste mode is a normal outcome: no error box, no fix hints; the setup note is last.
    assert "How to fix" not in html and "not created" not in html and "Reason:" not in html
    assert "attached" not in html and "could not be hosted" not in html
    assert html.rindex(notify.SETUP_NOTE) > html.index("Benchmark only")


def test_failure_email_has_hints_then_the_same_paste_layout():
    failure = notify.SubstackFailure(notify.AUTH_OR_BLOCK, "401 Unauthorized")
    subject, html = notify.build_failure_email(DATE, failure, HOSTED_MD, KEEP, ENTRIES, VERDICTS, FIGS, VerifyResult())
    assert subject == f"Substack draft NOT created — {DATE}"
    before, block, after = _split_copy_block(html)
    assert before.index("The Substack draft was not created.") < before.index("How to fix") < before.index(notify.PASTE_INSTRUCTIONS)
    assert "Refresh SUBSTACK_COOKIE" in before and "401 Unauthorized" in before
    assert "Title (paste it into the title field)" in before and "2 Ways to &lt;Check&gt; AI" not in block
    assert f'<img src="{HOSTED}/fig-2609.00001.png"' in block
    assert "Figure not openly licensed" in after and "Benchmark only" in after
    assert notify.SETUP_NOTE not in html
    assert "figure attached" not in html and "attached as" not in html


def test_hosting_failed_note_and_attachment_placeholders():
    md = HOSTED_MD.replace(f"{HOSTED}/fig-2609.00001.png", "fig-2609.00001.png")
    _, html = notify.build_paste_email(
        DATE, md, KEEP, ENTRIES, VERDICTS, FIGS, VerifyResult(),
        attachment_names={A.url: "fig-2609.00001.png"}, hosting_error="git push <rejected>",
    )
    before, block, after = _split_copy_block(html)
    assert (
        "Figures could not be hosted (git push &lt;rejected&gt;) — they're attached instead; "
        "pasted images won't carry over." in before
    )
    assert "<img" not in html
    assert "\N{PAPERCLIP} figure attached: <strong>fig-2609.00001.png</strong>" in block
    assert "attached as fig-2609.00001.png" in after


def test_only_unhosted_figures_are_marked_attached():
    figs = {A.url: _fig(A), B.url: _fig(B)}
    md = (
        "# T\n\nIntro.\n\n"
        f"![cap A]({HOSTED}/fig-2609.00001.png)\n\n**[A](http://arxiv.org/abs/2609.00001v1)** — Body A.\n\n"
        "![cap B](fig-2609.00002.png)\n\n**[B](http://arxiv.org/abs/2609.00002v1)** — Body B.\n"
    )
    failure = notify.SubstackFailure(notify.ERROR, "500")
    for build in (
        lambda **kw: notify.build_paste_email(DATE, md, KEEP, ENTRIES, VERDICTS, figs, VerifyResult(), **kw),
        lambda **kw: notify.build_failure_email(DATE, failure, md, KEEP, ENTRIES, VERDICTS, figs, VerifyResult(), **kw),
    ):
        _, html = build(attachment_names={B.url: "fig-2609.00002.png"}, hosting_error="no public URL for 1 of 2 figure(s)")
        _, block, after = _split_copy_block(html)
        assert block.count("<img") == 1 and f'<img src="{HOSTED}/fig-2609.00001.png"' in block
        assert block.count("figure attached") == 1 and "<strong>fig-2609.00002.png</strong>" in block
        assert "attached as fig-2609.00002.png" in after and "attached as fig-2609.00001.png" not in after
        assert "no public URL for 1 of 2 figure(s)" in html


# ---------------------------------------------------------------- paper limit (cut papers, institutions)

D = DigestEntry(rank=4, section="top", title="Cut <Paper> D", url="http://arxiv.org/abs/2609.00004v1", arxiv_id="2609.00004", max_h_index=82)
E = DigestEntry(rank=5, section="top", title="Cut Paper E", url="http://arxiv.org/abs/2609.00005v1", arxiv_id="2609.00005")
CAP_ENTRIES = ENTRIES + [D, E]
CAP_VERDICTS = {
    **VERDICTS,
    D.url: Verdict(D.url, True, "Proposes a <calibration> check.", "S", "uncertainty-quantification"),
    E.url: Verdict(E.url, True, "Validates a simulator.", "S", "validation"),
}


def _aff(institutions, tier):
    return SimpleNamespace(url="", institutions=institutions, matched=institutions[:1], tier=tier, source="html")


CAP_AFFS = {
    A.url: _aff(["MIT", "Google DeepMind"], 1),
    B.url: _aff([], 0),
    D.url: _aff(["Acme <Labs>"], 0),
    E.url: _aff(["Stanford University", "Mila"], 2),
}


def test_paste_email_lists_cut_papers_and_institutions_outside_the_copy_block():
    _, html = notify.build_paste_email(
        DATE, HOSTED_MD, KEEP, CAP_ENTRIES, CAP_VERDICTS, FIGS, VerifyResult(),
        cut_entries=[E, D], affiliations=CAP_AFFS,
    )
    before, block, after = _split_copy_block(html)
    for text in ("Institutions", "Not included", "Cut Paper E", "h-index"):
        assert text not in block and text not in before

    assert (
        "<strong>4</strong> of <strong>5</strong> papers from the 2026-09-28 digest passed the filter; "
        "the <strong>2</strong> best are in the post." in after
    )
    assert "Kept <strong>" not in html
    # per-paper facts for the papers in the post
    assert "Institutions: MIT, Google DeepMind (tier 1)" in after
    assert "Institutions: not found" in after  # B: lookup found nothing
    # the cut papers, best first, after the papers in the post and before the dropped ones
    assert "Not included: over the 2-paper limit (2)" in after
    assert after.index("Papers in the post (2)") < after.index("Not included") < after.index("Dropped papers (1)")
    assert after.index("Cut Paper E") < after.index("Cut &lt;Paper&gt; D")
    assert '<a href="http://arxiv.org/abs/2609.00005v1" style="color:#1a56db"><strong>Cut Paper E</strong></a>' in after
    assert "Institutions: Stanford University, Mila (tier 2) · author h-index unknown · #5 in the digest" in after
    assert "Institutions: Acme &lt;Labs&gt; (not on the trusted list) · max author h-index 82 · #4 in the digest" in after
    assert "Filter: Proposes a &lt;calibration&gt; check." in after and "Filter: Validates a simulator." in after
    assert "swap it into the post by hand" in after
    # cut papers passed the filter: they are not in the dropped list
    dropped = after.split("Dropped papers", 1)[1]
    assert "Cut Paper E" not in dropped and "Benchmark only" in dropped


def test_success_and_failure_emails_list_cut_papers():
    _, success = notify.build_success_email(
        DATE, "T", DraftResult(101, EDIT_URL, True), KEEP, CAP_ENTRIES, CAP_VERDICTS, FIGS, VerifyResult(),
        cut_entries=[D], affiliations=CAP_AFFS,
    )
    assert "Not included: over the 2-paper limit (1)" in success
    assert "Institutions: MIT, Google DeepMind (tier 1)" in success
    assert "the <strong>2</strong> best are in the post" in success

    failure = notify.SubstackFailure(notify.AUTH_OR_BLOCK, "401")
    _, failed = notify.build_failure_email(
        DATE, failure, HOSTED_MD, KEEP, CAP_ENTRIES, CAP_VERDICTS, FIGS, VerifyResult(), cut_entries=[D], affiliations=CAP_AFFS
    )
    _, block, after = _split_copy_block(failed)
    assert "Not included" in after and "Not included" not in block


def test_writer_failure_email_lists_cut_papers():
    _, html = notify.build_writer_failure_email(
        DATE, "x", KEEP, CAP_ENTRIES, CAP_VERDICTS, FIGS, cut_entries=[D], affiliations=CAP_AFFS
    )
    assert "Papers picked for the post (2)" in html
    assert "the <strong>2</strong> best were picked for the post" in html
    assert "Not included: over the 2-paper limit (1)" in html


def test_cut_paper_without_lookup_and_long_institution_list():
    many = _aff([f"Uni {i}" for i in range(1, 9)], 1)
    _, html = notify.build_success_email(
        DATE, "T", DraftResult(101, EDIT_URL, True), KEEP, CAP_ENTRIES, CAP_VERDICTS, FIGS, VerifyResult(),
        cut_entries=[D], affiliations={A.url: many},
    )
    assert "Institutions: Uni 1, Uni 2, Uni 3, Uni 4, Uni 5, Uni 6, +2 more (tier 1)" in html
    assert "Institutions: not looked up · max author h-index 82" in html  # lookup failed: no affiliations


def test_no_cut_means_no_institutions_or_cut_section():
    _, html = notify.build_paste_email(DATE, HOSTED_MD, KEEP, ENTRIES, VERDICTS, FIGS, VerifyResult())
    assert "Not included" not in html and "Institutions" not in html
    assert "Kept <strong>2</strong> of <strong>3</strong>" in html

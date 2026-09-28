"""Orchestrator tests for write_post.py with every pipeline stage faked.

write_post.py imports the input/writing/Substack modules, which land in
separate PRs; until they are all present this module is skipped.
"""
import base64
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_DEPENDENCIES = [
    f"src.substack_post.{name}"
    for name in (
        "report_parser",
        "paper_details",
        "figures",
        "writer",
        "prompts",
        "render",
        "prosemirror",
        "substack_client",
    )
]
for _name in _DEPENDENCIES:
    try:
        importlib.import_module(_name)
    except Exception as _exc:  # ImportError before merge; OSError if e.g. libcairo is missing locally
        pytest.skip(f"{_name} not importable yet: {_exc}", allow_module_level=True)

import write_post  # noqa: E402
from src.substack_post.models import (  # noqa: E402
    DigestEntry,
    DraftResult,
    Figure,
    PaperDetails,
    PostDraft,
    UploadedImage,
    Verdict,
    VerifyResult,
)

# The real renderer and ProseMirror builder, captured before any test patches them.
REAL_RENDER = write_post.render.render_markdown
REAL_SPLIT_TITLE = write_post.prosemirror.split_title
REAL_MARKDOWN_TO_DOC = write_post.prosemirror.markdown_to_doc

DATE = "2026-09-28"
PUB = "https://example.substack.com"
ENV = {
    "OPENROUTER_API_KEY": "or-key",
    "OPENROUTER_MODEL": "base/model",
    "SUBSTACK_WRITER_MODEL": "writer/model",
    "RESEND_API_KEY": "re-key",
    "REPORT_EMAIL_FROM": "bot@example.com",
    "REPORT_EMAIL_TO": "me@example.com",
    "SUBSTACK_PUBLICATION_URL": PUB,
    "SUBSTACK_COOKIE": "substack.sid=abc",
    "SUBSTACK_PROXY": "socks5h://127.0.0.1:40000",
}


def _entry(rank, arxiv_id):
    return DigestEntry(
        rank=rank,
        section="top",
        title=f"Paper {rank} <T&C>",
        url=f"http://arxiv.org/abs/{arxiv_id}v1",
        arxiv_id=arxiv_id,
        version="v1",
    )


ENTRIES = [_entry(1, "2609.00001"), _entry(2, "2609.00002"), _entry(3, "2609.00003")]


def _fake_render(post, keep_entries, images=None):
    images = images or {}
    parts = [f"# {post.title}", post.intro]
    for e in keep_entries:
        if e.url in images:
            src, caption = images[e.url]
            parts.append(f"![{caption}]({src})")
        parts.append(f"**[{e.title}]({e.url})** — {post.bodies[e.url]}")
    return "\n\n".join(parts) + "\n"


def _substack_error(status_code, message="Substack error"):
    """A real SubstackError subclass instance, without depending on its constructor signature."""

    class _Err(write_post.substack_client.SubstackError):
        def __init__(self):
            Exception.__init__(self, message)
            self.status_code = status_code

    return _Err()


def _auth_error(kind, message):
    class _AuthErr(write_post.substack_client.SubstackAuthOrBlockError):
        def __init__(self):
            Exception.__init__(self, message)
            self.kind = kind
            self.status_code = 403

    return _AuthErr()


class FakeSubstackClient:
    instances = []
    fail_with = None  # exception instance raised by upload_image
    update_fail_with = None  # exception instance raised by update_draft

    def __init__(self, publication_url, cookie, proxy=None, session=None):
        if cookie.strip() == "":
            raise ValueError("SUBSTACK_COOKIE is empty")  # like the real client's validation
        self.args = (publication_url, cookie, proxy)
        self.uploads, self.created, self.updated = [], [], []
        FakeSubstackClient.instances.append(self)

    def upload_image(self, path):
        if FakeSubstackClient.fail_with is not None:
            raise FakeSubstackClient.fail_with
        self.uploads.append(Path(path))
        return UploadedImage(
            url=f"https://substack-post-media.s3.amazonaws.com/public/images/{Path(path).name}",
            width=800,
            height=600,
            bytes=Path(path).stat().st_size,
            content_type="image/png",
        )

    def create_draft(self, title, subtitle, doc):
        self.created.append((title, subtitle, doc))
        return DraftResult(draft_id=101, edit_url=f"{PUB}/publish/post/101", created=True)

    def update_draft(self, draft_id, title, subtitle, doc):
        self.updated.append((draft_id, title, subtitle, doc))
        if FakeSubstackClient.update_fail_with is not None:
            raise FakeSubstackClient.update_fail_with
        return DraftResult(draft_id=draft_id, edit_url=f"{PUB}/publish/post/{draft_id}", created=False)


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    """Fake every stage; knobs and call records live on the returned namespace."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / f"{DATE}.md").write_text("# digest\n", encoding="utf-8")
    state_path = tmp_path / "state" / "substack_drafts.json"

    p = SimpleNamespace(
        reports=reports,
        state_path=state_path,
        post_path=reports / f"{DATE}-substack-post-ste100.md",
        env=dict(ENV),
        entries=list(ENTRIES),
        keep_urls={ENTRIES[0].url, ENTRIES[2].url},
        no_figure_urls={ENTRIES[2].url},
        verify=VerifyResult(errors=["Link appears twice: X"], warnings=["Long sentence in paragraph 2"]),
        emails=[],
        calls={},
        figdir=None,
        config={},
        writer_raises=None,
    )

    FakeSubstackClient.instances = []
    FakeSubstackClient.fail_with = None
    FakeSubstackClient.update_fail_with = None

    def record(name, *args, **kwargs):
        p.calls.setdefault(name, []).append((args, kwargs))

    def find_report(reports_dir, date=None):
        record("find_report", reports_dir, date)
        path = Path(reports_dir) / f"{date or DATE}.md"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def parse_report(path):
        record("parse_report", path)
        return p.entries

    def fetch_details(entries, session=None):
        return {e.url: PaperDetails(e.arxiv_id, "v1", e.title, "Abstract.", ["Ada Lovelace"]) for e in entries}

    def fetch_licenses(details, urls, session=None):
        record("fetch_licenses", list(urls))

    def filter_papers(client, model, entries, details):
        record("filter_papers", model)
        return {
            e.url: Verdict(
                url=e.url,
                keep=e.url in p.keep_urls,
                rationale=f"rationale {e.rank}",
                summary="S.",
                contribution_type="validation" if e.url in p.keep_urls else "",
            )
            for e in entries
        }

    def choose_figures(client, model, captions_by_url):
        record("choose_figures", captions_by_url)
        return {url: 0 for url in captions_by_url}

    def fetch_main_figures(entries, details, out_dir, choose=None, session=None):
        p.figdir = Path(out_dir)
        assert choose is not None
        choose({e.url: ["Figure 1: Overview."] for e in entries})
        figs = {}
        for e in entries:
            if e.url in p.no_figure_urls:
                continue
            path = Path(out_dir) / f"{e.arxiv_id}-main.png"
            path.write_bytes(b"\x89PNG fake " + e.arxiv_id.encode())
            figs[e.url] = Figure(
                url=e.url,
                path=path,
                figure_number="1",
                credit_caption=f"Figure 1 from “{e.title}” (arXiv:{e.arxiv_id})",
                source="html",
                openly_licensed=False,
                width=800,
                height=600,
                mime="image/png",
            )
        return figs

    def load_style_example(reports_dir, exclude_date=None):
        record("load_style_example", reports_dir, exclude_date)
        return "STYLE"

    def fake_write_post(client, model, keep_entries, verdicts, details, all_entries, style_example=None, max_retries=2):
        record("write_post", model, [e.url for e in keep_entries], style_example)
        if p.writer_raises is not None:
            raise p.writer_raises
        post = PostDraft(
            title=f"{len(keep_entries)} Ways to Check AI",
            intro="This week covers a few papers.",
            bodies={e.url: f"Body for {e.rank}. The validation contribution is X." for e in keep_entries},
        )
        return post, p.verify

    def split_title(markdown):
        first, _, rest = markdown.partition("\n")
        return first.lstrip("# ").strip(), rest.strip()

    def markdown_to_doc(markdown, images):
        record("markdown_to_doc", markdown, sorted(images))
        return {"type": "doc", "content": []}

    def send(api_key, sender, recipients, subject, html_body, attachments=None):
        p.emails.append(SimpleNamespace(subject=subject, html=html_body, attachments=attachments))
        return True

    monkeypatch.setattr(write_post, "REPORTS_DIR", reports)
    monkeypatch.setattr(write_post, "DRAFTS_STATE_PATH", state_path)
    monkeypatch.setattr(write_post, "load_env_vars", lambda required, optional=(): dict(p.env))
    monkeypatch.setattr(write_post, "load_config", lambda: p.config)
    monkeypatch.setattr(write_post, "send_digest_email", send)
    monkeypatch.setattr(write_post.report_parser, "find_report", find_report)
    monkeypatch.setattr(write_post.report_parser, "parse_report", parse_report)
    monkeypatch.setattr(write_post.paper_details, "fetch_details", fetch_details)
    monkeypatch.setattr(write_post.paper_details, "fetch_licenses", fetch_licenses)
    monkeypatch.setattr(write_post.writer, "make_client", lambda api_key: object())
    monkeypatch.setattr(write_post.writer, "filter_papers", filter_papers)
    monkeypatch.setattr(write_post.writer, "choose_figures", choose_figures)
    monkeypatch.setattr(write_post.writer, "write_post", fake_write_post)
    monkeypatch.setattr(write_post.figures, "fetch_main_figures", fetch_main_figures)
    monkeypatch.setattr(write_post.prompts, "load_style_example", load_style_example)
    monkeypatch.setattr(write_post.render, "render_markdown", _fake_render)
    monkeypatch.setattr(write_post.prosemirror, "split_title", split_title)
    monkeypatch.setattr(write_post.prosemirror, "markdown_to_doc", markdown_to_doc)
    monkeypatch.setattr(write_post.substack_client, "SubstackClient", FakeSubstackClient)
    return p


def _state(p):
    return json.loads(p.state_path.read_text(encoding="utf-8"))


def _seed_state(p, draft_id=55):
    p.state_path.parent.mkdir(parents=True)
    p.state_path.write_text(
        json.dumps(
            {DATE: {"draft_id": draft_id, "edit_url": f"{PUB}/publish/post/{draft_id}", "title": "Old", "updated_at": "2026-09-28T00:00:00+00:00"}}
        ),
        encoding="utf-8",
    )


def test_quiet_week_sends_quiet_email_only(pipe):
    pipe.entries = []
    assert write_post.main([]) == 0
    assert [e.subject for e in pipe.emails] == [f"No Substack post this week — {DATE}"]
    assert "quiet week" in pipe.emails[0].html
    assert "filter_papers" not in pipe.calls
    assert not pipe.post_path.exists() and not pipe.state_path.exists()


def test_zero_keep_sends_quiet_email_with_drop_rationales(pipe):
    pipe.keep_urls = set()
    assert write_post.main([]) == 0
    assert len(pipe.emails) == 1
    email = pipe.emails[0]
    assert email.subject == f"No Substack post this week — {DATE}"
    for e in ENTRIES:
        assert f"rationale {e.rank}" in email.html
    assert "write_post" not in pipe.calls and not FakeSubstackClient.instances
    assert not pipe.post_path.exists() and not pipe.state_path.exists()


def test_success_creates_draft_writes_cdn_post_and_state(pipe):
    assert write_post.main([]) == 0

    (client,) = FakeSubstackClient.instances
    assert client.args == (PUB, "substack.sid=abc", "socks5h://127.0.0.1:40000")
    assert len(client.created) == 1 and not client.updated
    title, subtitle, _doc = client.created[0]
    assert title == "2 Ways to Check AI" and subtitle == ""
    assert [path.name for path in client.uploads] == ["2609.00001-main.png"]

    # Writer model preference, licenses for KEEP only, style example excludes this week.
    assert pipe.calls["filter_papers"][0][0] == ("writer/model",)
    assert pipe.calls["fetch_licenses"][0][0] == ([ENTRIES[0].url, ENTRIES[2].url],)
    assert pipe.calls["load_style_example"][0][0] == (pipe.reports, DATE)
    assert pipe.calls["choose_figures"]
    # ProseMirror gets the body without the title, images keyed by CDN URL.
    body_md, image_keys = pipe.calls["markdown_to_doc"][0][0]
    assert not body_md.startswith("#")
    assert image_keys == ["https://substack-post-media.s3.amazonaws.com/public/images/2609.00001-main.png"]

    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert "![Figure 1 from “Paper 1 <T&C>” (arXiv:2609.00001)](https://substack-post-media" in post_md
    assert "fig-" not in post_md

    record = _state(pipe)[DATE]
    assert record["draft_id"] == 101
    assert record["edit_url"] == f"{PUB}/publish/post/101"
    assert record["title"] == "2 Ways to Check AI"
    assert record["updated_at"].endswith("+00:00")

    (email,) = pipe.emails
    assert email.subject == "Substack draft ready: 2 Ways to Check AI"
    assert f"{PUB}/publish/post/101" in email.html
    assert email.attachments is None
    assert "Link appears twice: X" in email.html  # remaining verify error surfaced
    assert "No figure found" in email.html  # paper 3 has no figure
    assert "not openly licensed" in email.html
    assert pipe.figdir is not None and not pipe.figdir.exists()  # temp figures cleaned up


def test_already_drafted_skips_without_email(pipe):
    _seed_state(pipe)
    assert write_post.main([]) == 0
    assert pipe.emails == []
    assert "parse_report" not in pipe.calls
    assert not FakeSubstackClient.instances
    assert not pipe.post_path.exists()


def test_force_updates_existing_draft(pipe):
    _seed_state(pipe, draft_id=55)
    assert write_post.main(["--force"]) == 0

    (client,) = FakeSubstackClient.instances
    assert not client.created
    assert [u[0] for u in client.updated] == [55]
    record = _state(pipe)[DATE]
    assert record["draft_id"] == 55 and record["title"] == "2 Ways to Check AI"
    assert record["updated_at"] != "2026-09-28T00:00:00+00:00"
    (email,) = pipe.emails
    assert email.subject.startswith("Substack draft ready:")
    assert "updated" in email.html


def test_force_falls_back_to_create_when_draft_was_deleted(pipe):
    _seed_state(pipe, draft_id=55)
    FakeSubstackClient.update_fail_with = _substack_error(404, "draft not found")
    assert write_post.main(["--force"]) == 0

    (client,) = FakeSubstackClient.instances
    assert [u[0] for u in client.updated] == [55]
    assert len(client.created) == 1
    assert _state(pipe)[DATE]["draft_id"] == 101
    assert pipe.emails[0].subject.startswith("Substack draft ready:")


def test_force_update_other_error_is_a_failure(pipe):
    _seed_state(pipe, draft_id=55)
    FakeSubstackClient.update_fail_with = _substack_error(500, "server exploded")
    assert write_post.main(["--force"]) == 0

    (client,) = FakeSubstackClient.instances
    assert not client.created
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "server exploded" in email.html
    assert _state(pipe)[DATE]["draft_id"] == 55  # state untouched


def test_cloudflare_block_puts_warp_hint_first(pipe):
    FakeSubstackClient.fail_with = _auth_error("cloudflare", "403 from Substack")
    assert write_post.main([]) == 0
    html = pipe.emails[0].html
    assert html.index("SUBSTACK_USE_WARP") < html.index("Refresh SUBSTACK_COOKIE")
    assert "residential proxy" in html  # SUBSTACK_PROXY was set for this run


def test_auth_failure_emails_post_with_attachments(pipe):
    FakeSubstackClient.fail_with = _auth_error("cookie", "401 Unauthorized")
    assert write_post.main(["--date", DATE]) == 0

    assert not pipe.state_path.exists()
    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert "](fig-2609.00001.png)" in post_md
    assert "substack-post-media" not in post_md

    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "401 Unauthorized" in email.html
    assert email.html.index("Refresh SUBSTACK_COOKIE") < email.html.index("SUBSTACK_USE_WARP")
    assert "figure attached: <strong>fig-2609.00001.png</strong>" in email.html
    assert "Paper 1 &lt;T&amp;C&gt;" in email.html
    (attachment,) = email.attachments
    assert attachment["filename"] == "fig-2609.00001.png"
    assert base64.b64decode(attachment["content"]) == b"\x89PNG fake 2609.00001"


def test_missing_substack_config_is_a_failure_not_a_crash(pipe):
    del pipe.env["SUBSTACK_COOKIE"]
    assert write_post.main([]) == 0
    assert not FakeSubstackClient.instances
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "Substack not configured" in email.html
    assert pipe.post_path.exists() and not pipe.state_path.exists()


def test_malformed_substack_config_is_a_failure_not_a_crash(pipe):
    pipe.env["SUBSTACK_COOKIE"] = "   "
    assert write_post.main([]) == 0
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "Substack misconfigured: SUBSTACK_COOKIE is empty" in email.html


def test_skip_substack_flag(pipe):
    assert write_post.main(["--skip-substack"]) == 0
    assert not FakeSubstackClient.instances
    (email,) = pipe.emails
    assert "--skip-substack" in email.html
    assert len(email.attachments) == 1


def test_unexpected_substack_exception_still_emails_then_exits_nonzero(pipe):
    FakeSubstackClient.fail_with = RuntimeError("boom")
    assert write_post.main([]) == 1
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "boom" in email.html
    assert pipe.post_path.exists()


def _writer_error(message):
    base = getattr(write_post.writer, "WriterError", RuntimeError)

    class _Err(base):
        def __init__(self):
            Exception.__init__(self, message)

    return _Err()


def test_writer_failure_emails_and_exits_nonzero(pipe):
    pipe.writer_raises = _writer_error("no parseable JSON after 3 attempts")
    assert write_post.main([]) == 1

    (email,) = pipe.emails
    assert email.subject == f"Substack post NOT written — {DATE}"
    assert "post generation failed: no parseable JSON after 3 attempts" in email.html
    assert [a["filename"] for a in email.attachments] == ["fig-2609.00001.png"]
    assert not FakeSubstackClient.instances
    assert not pipe.post_path.exists() and not pipe.state_path.exists()


def test_writer_failure_in_dry_run_sends_nothing(pipe):
    pipe.writer_raises = _writer_error("boom")
    assert write_post.main(["--dry-run"]) == 1
    assert pipe.emails == []


def test_configured_style_example_is_used(pipe, tmp_path):
    approved = tmp_path / "2026-09-07-substack-post-ste100.md"
    approved.write_text("APPROVED STYLE", encoding="utf-8")
    pipe.config = {"substack_post": {"style_example": str(approved)}}
    assert write_post.main([]) == 0
    assert pipe.calls["write_post"][0][0][2] == "APPROVED STYLE"
    assert "load_style_example" not in pipe.calls


def test_configured_style_example_for_this_date_falls_back(pipe):
    pipe.config = {"substack_post": {"style_example": f"reports/{DATE}-substack-post-ste100.md"}}
    assert write_post.main([]) == 0
    assert pipe.calls["load_style_example"][0][0] == (pipe.reports, DATE)
    assert pipe.calls["write_post"][0][0][2] == "STYLE"


def test_unreadable_configured_style_example_gives_none(pipe, tmp_path):
    pipe.config = {"substack_post": {"style_example": str(tmp_path / "missing.md")}}
    assert write_post.main([]) == 0
    assert pipe.calls["write_post"][0][0][2] is None
    assert "load_style_example" not in pipe.calls


def test_dry_run_writes_and_sends_nothing(pipe, capsys):
    _seed_state(pipe)  # dry run previews even an already-drafted date
    before = pipe.state_path.read_text(encoding="utf-8")
    assert write_post.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# 2 Ways to Check AI")
    assert str(pipe.figdir / "2609.00001-main.png") in out
    assert pipe.emails == []
    assert not FakeSubstackClient.instances
    assert not pipe.post_path.exists()
    assert pipe.state_path.read_text(encoding="utf-8") == before


def test_missing_report_exits_nonzero(pipe):
    assert write_post.main(["--date", "2026-01-05"]) == 1
    assert pipe.emails == []


def test_bad_date_argument_is_rejected():
    with pytest.raises(SystemExit):
        write_post.parse_args(["--date", "28-09-2026"])


def test_real_render_and_prosemirror_glue(pipe, monkeypatch):
    """CDN-image markdown from the real renderer must build a doc with the uploaded images."""
    monkeypatch.setattr(write_post.render, "render_markdown", REAL_RENDER)
    monkeypatch.setattr(write_post.prosemirror, "split_title", REAL_SPLIT_TITLE)
    monkeypatch.setattr(write_post.prosemirror, "markdown_to_doc", REAL_MARKDOWN_TO_DOC)
    assert write_post.main([]) == 0

    (client,) = FakeSubstackClient.instances
    title, _subtitle, doc = client.created[0]
    assert title == "2 Ways to Check AI"
    assert doc["type"] == "doc"
    image_srcs = [
        child["attrs"]["src"]
        for node in doc["content"]
        if node["type"] == "captionedImage"
        for child in node["content"]
        if child["type"] == "image2"
    ]
    assert image_srcs == ["https://substack-post-media.s3.amazonaws.com/public/images/2609.00001-main.png"]
    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert post_md.startswith("# 2 Ways to Check AI\n")
    assert f"**[Paper 1 <T&C>]({ENTRIES[0].url})** — Body for 1." in post_md

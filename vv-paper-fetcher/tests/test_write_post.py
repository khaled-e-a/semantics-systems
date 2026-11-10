"""Orchestrator tests for write_post.py with every pipeline stage faked.

write_post.py imports the input/writing/Substack modules, figure_host
(which pushes figures to the public `figures` branch), and affiliations
(institution lookup for the paper limit), which land in separate PRs; until
they are all present this module is skipped.
figure_host.publish_figures and affiliations.lookup_affiliations are always
faked: these tests never push and never call an LLM.
"""
import base64
import importlib
import json
import logging
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
        "figure_host",
        "affiliations",
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
HOSTED = f"https://raw.githubusercontent.com/khaled-e-a/semantics-systems/figures/{DATE}"
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
        host_raises=None,  # exception raised by figure_host.publish_figures
        host_skip=set(),  # paper URLs publish_figures leaves out of its result
        tiers={},  # paper URL -> institution tier returned by the affiliations lookup (default 0)
        lookup_raises=None,  # exception raised by affiliations.lookup_affiliations
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
        record("fetch_main_figures", [e.url for e in entries])
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

    def lookup_affiliations(client, model, entries, details, trusted, session=None):
        record("lookup_affiliations", model, [e.url for e in entries], sorted(details), trusted)
        if p.lookup_raises is not None:
            raise p.lookup_raises
        return {
            e.url: SimpleNamespace(
                url=e.url,
                institutions=[f"Inst {e.rank}"],
                matched=[f"Inst {e.rank}"] if p.tiers.get(e.url) else [],
                tier=p.tiers.get(e.url, 0),
                source="html",
            )
            for e in entries
        }

    def publish_figures(files, report_date, repo_dir, **kwargs):
        record("publish_figures", dict(files), report_date, repo_dir)
        for path, _name in files.values():
            assert Path(path).exists()  # hosted before the temp figure dir is removed
        if p.host_raises is not None:
            raise p.host_raises
        return {url: f"{HOSTED}/{name}" for url, (_path, name) in files.items() if url not in p.host_skip}

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
    monkeypatch.setattr(write_post.figure_host, "publish_figures", publish_figures)
    monkeypatch.setattr(write_post.affiliations, "lookup_affiliations", lookup_affiliations)
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
    assert "publish_figures" not in pipe.calls  # the draft has Substack-hosted images already
    assert "lookup_affiliations" not in pipe.calls  # within the paper limit: no institution lookup
    assert "Not included" not in email.html and "Institutions" not in email.html


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


def test_auth_failure_emails_paste_ready_post_with_hosted_figures(pipe):
    FakeSubstackClient.fail_with = _auth_error("cookie", "401 Unauthorized")
    assert write_post.main(["--date", DATE]) == 0

    assert not pipe.state_path.exists()
    ((files, report_date, repo_dir), _kw), = pipe.calls["publish_figures"]
    assert files == {ENTRIES[0].url: (pipe.figdir / "2609.00001-main.png", "fig-2609.00001.png")}
    assert report_date == DATE and repo_dir == write_post.PROJECT_ROOT
    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert f"]({HOSTED}/fig-2609.00001.png)" in post_md
    assert "substack-post-media" not in post_md and "](fig-" not in post_md

    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "401 Unauthorized" in email.html
    assert email.html.index("Refresh SUBSTACK_COOKIE") < email.html.index("SUBSTACK_USE_WARP")
    assert email.html.index("SUBSTACK_USE_WARP") < email.html.index(write_post.notify.COPY_START)
    assert f'<img src="{HOSTED}/fig-2609.00001.png"' in email.html
    assert "figure attached" not in email.html
    assert "Paper 1 &lt;T&amp;C&gt;" in email.html
    assert email.attachments is None


@pytest.mark.parametrize("unset", ["SUBSTACK_COOKIE", "SUBSTACK_PUBLICATION_URL"])
def test_missing_substack_config_sends_paste_email(pipe, unset):
    del pipe.env[unset]
    assert write_post.main([]) == 0
    assert not FakeSubstackClient.instances
    assert len(pipe.calls["publish_figures"]) == 1

    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert f"![Figure 1 from “Paper 1 <T&C>” (arXiv:2609.00001)]({HOSTED}/fig-2609.00001.png)" in post_md
    assert not pipe.state_path.exists()

    (email,) = pipe.emails
    assert email.subject == "Substack post ready to paste: 2 Ways to Check AI"
    assert email.attachments is None
    html = email.html
    assert html.index(write_post.notify.PASTE_INSTRUCTIONS) < html.index(write_post.notify.COPY_START)
    block = html.split(write_post.notify.COPY_START, 1)[1].split(write_post.notify.COPY_END, 1)[0]
    assert f'<img src="{HOSTED}/fig-2609.00001.png" alt="Figure 1 from “Paper 1 &lt;T&amp;C&gt;” (arXiv:2609.00001)"' in block
    assert "2 Ways to Check AI" not in block  # the title goes in Substack's title field
    assert "How to fix" not in html and "not configured" not in html and "figure attached" not in html
    assert write_post.notify.SETUP_NOTE in html
    assert "Link appears twice: X" in html  # remaining verify error still surfaced
    assert pipe.figdir is not None and not pipe.figdir.exists()


@pytest.mark.parametrize("exc", ["figure_host_error", "unexpected"])
def test_hosting_failure_falls_back_to_attachments(pipe, exc):
    del pipe.env["SUBSTACK_COOKIE"]
    pipe.host_raises = (
        write_post.figure_host.FigureHostError("git push rejected") if exc == "figure_host_error" else OSError("git push rejected")
    )
    assert write_post.main([]) == 0

    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert "](fig-2609.00001.png)" in post_md and HOSTED not in post_md

    (email,) = pipe.emails
    assert email.subject == "Substack post ready to paste: 2 Ways to Check AI"
    assert "Figures could not be hosted (git push rejected)" in email.html
    assert "pasted images won't carry over" in email.html
    assert "figure attached: <strong>fig-2609.00001.png</strong>" in email.html
    assert "<img" not in email.html
    (attachment,) = email.attachments
    assert attachment["filename"] == "fig-2609.00001.png"
    assert base64.b64decode(attachment["content"]) == b"\x89PNG fake 2609.00001"


def test_partly_hosted_figures_attach_only_the_rest(pipe):
    FakeSubstackClient.fail_with = _auth_error("cookie", "401 Unauthorized")
    pipe.no_figure_urls = set()  # papers 1 and 3 both have a figure
    pipe.host_skip = {ENTRIES[2].url}
    assert write_post.main([]) == 0

    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert f"]({HOSTED}/fig-2609.00001.png)" in post_md and "](fig-2609.00003.png)" in post_md
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert f'<img src="{HOSTED}/fig-2609.00001.png"' in email.html
    assert email.html.count("figure attached") == 1 and "<strong>fig-2609.00003.png</strong>" in email.html
    assert "no public URL for 1 of 2 figure(s)" in email.html
    assert [a["filename"] for a in email.attachments] == ["fig-2609.00003.png"]


def test_malformed_substack_config_is_a_failure_not_a_crash(pipe):
    pipe.env["SUBSTACK_COOKIE"] = "   "
    assert write_post.main([]) == 0
    (email,) = pipe.emails
    assert email.subject == f"Substack draft NOT created — {DATE}"
    assert "Substack misconfigured: SUBSTACK_COOKIE is empty" in email.html


def test_skip_substack_flag_sends_paste_email(pipe):
    assert write_post.main(["--skip-substack"]) == 0  # Substack secrets are set, but not used
    assert not FakeSubstackClient.instances
    assert len(pipe.calls["publish_figures"]) == 1
    (email,) = pipe.emails
    assert email.subject == "Substack post ready to paste: 2 Ways to Check AI"
    assert "How to fix" not in email.html
    assert f'<img src="{HOSTED}/fig-2609.00001.png"' in email.html
    assert email.attachments is None


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
    assert len(pipe.calls["publish_figures"]) == 1
    assert f'href="{HOSTED}/fig-2609.00001.png"' in email.html
    assert email.attachments is None
    assert not FakeSubstackClient.instances
    assert not pipe.post_path.exists() and not pipe.state_path.exists()


def test_writer_failure_with_hosting_failure_attaches_figures(pipe):
    pipe.writer_raises = _writer_error("boom")
    pipe.host_raises = write_post.figure_host.FigureHostError("no network")
    assert write_post.main([]) == 1
    (email,) = pipe.emails
    assert "Figures could not be hosted (no network)" in email.html
    assert [a["filename"] for a in email.attachments] == ["fig-2609.00001.png"]


def test_writer_failure_without_figures_does_not_host(pipe):
    pipe.writer_raises = _writer_error("boom")
    pipe.no_figure_urls = {e.url for e in ENTRIES}
    assert write_post.main([]) == 1
    assert "publish_figures" not in pipe.calls
    assert pipe.emails[0].attachments is None


def test_writer_failure_in_dry_run_sends_nothing(pipe):
    pipe.writer_raises = _writer_error("boom")
    assert write_post.main(["--dry-run"]) == 1
    assert pipe.emails == []
    assert "publish_figures" not in pipe.calls


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
    assert "publish_figures" not in pipe.calls
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


def test_real_render_paste_email_glue(pipe, monkeypatch):
    """The real renderer's hosted-image markdown must come out as <img> + caption inside the copy block."""
    monkeypatch.setattr(write_post.render, "render_markdown", REAL_RENDER)
    del pipe.env["SUBSTACK_COOKIE"]
    assert write_post.main([]) == 0

    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert post_md.startswith("# 2 Ways to Check AI\n")
    (email,) = pipe.emails
    block = email.html.split(write_post.notify.COPY_START, 1)[1].split(write_post.notify.COPY_END, 1)[0]
    assert block.index("This week covers a few papers.") < block.index(f'<img src="{HOSTED}/fig-2609.00001.png"')
    assert "<em>Figure 1 from “Paper 1 &lt;T&amp;C&gt;” (arXiv:2609.00001)</em>" in block
    paper_link = f'<a href="{ENTRIES[0].url}" style="color:#1a56db"><strong>Paper 1 &lt;T&amp;C&gt;</strong></a>'
    assert f"{paper_link} — Body for 1." in block


# --------------------------------------------------------------------------- paper limit

TRUSTED = {"tier1": ["Massachusetts Institute of Technology (MIT)"], "tier2": ["Mila"]}
DROPPED_RANKS = (4, 11)  # the filter drops these, so 18 of the 20 papers pass
CAP_H_INDEX = {2: None, 3: 1, 5: 2, 20: None}  # every other paper: 50
# By tier (#20 is tier 1, #19 tier 2), then h-index: #5, #3 and #2 are cut, in that order.
CAP_SELECTED = [1, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def _cap_week(pipe, config=None):
    """20 papers, 18 of which pass the filter; returns {rank: entry}."""
    entries = [_entry(r, f"2609.{r:05d}") for r in range(1, 21)]
    for e in entries:
        e.max_h_index = CAP_H_INDEX.get(e.rank, 50)
    by_rank = {e.rank: e for e in entries}
    pipe.entries = entries
    pipe.keep_urls = {e.url for e in entries if e.rank not in DROPPED_RANKS}
    pipe.no_figure_urls = set()
    pipe.tiers = {by_rank[20].url: 1, by_rank[19].url: 2}
    pipe.config = {"substack_post": {"max_papers": 15, "trusted_institutions": TRUSTED}} if config is None else config
    return by_rank


def _urls(by_rank, ranks):
    return [by_rank[r].url for r in ranks]


def _written_urls(pipe):
    return pipe.calls["write_post"][0][0][1]


def test_cap_keeps_15_of_18_by_tier_then_h_index(pipe, caplog):
    caplog.set_level(logging.INFO, logger="write_post")
    by_rank = _cap_week(pipe)
    assert write_post.main([]) == 0

    # One lookup, for the papers that passed the filter only, with the writer model and the config lists.
    (((model, urls, detail_urls, trusted), _kw),) = pipe.calls["lookup_affiliations"]
    passed_ranks = [r for r in range(1, 21) if r not in DROPPED_RANKS]
    assert model == "writer/model"
    assert urls == _urls(by_rank, passed_ranks)
    assert set(detail_urls) >= set(urls)
    assert trusted == TRUSTED

    # Figures, licenses and the writer get exactly the 15 selected papers, in report order.
    selected = _urls(by_rank, CAP_SELECTED)
    assert pipe.calls["fetch_main_figures"][0][0][0] == selected
    assert pipe.calls["fetch_licenses"][0][0][0] == selected
    assert _written_urls(pipe) == selected
    (client,) = FakeSubstackClient.instances
    assert client.created[0][0] == "15 Ways to Check AI"
    assert len(client.uploads) == 15
    post_md = pipe.post_path.read_text(encoding="utf-8")
    assert all(f"]({by_rank[r].url})" not in post_md for r in (2, 3, 5))

    (email,) = pipe.emails
    html = email.html
    assert "<strong>18</strong> of <strong>20</strong> papers" in html and "the <strong>15</strong> best are in the post" in html
    assert "Institutions: Inst 20 (tier 1)" in html and "Institutions: Inst 19 (tier 2)" in html
    assert "Not included: over the 15-paper limit (3)" in html
    cut_section = html.split("Not included", 1)[1]
    assert cut_section.index("Paper 5 &lt;") < cut_section.index("Paper 3 &lt;") < cut_section.index("Paper 2 &lt;")
    assert "Filter: rationale 5" in cut_section

    assert "Selected 15 of 18 KEEP papers (cap 15); cut: #5 " in caplog.text
    assert "keep=18 selected=15 figures=15/15" in caplog.text


def test_within_cap_skips_the_lookup(pipe, caplog):
    caplog.set_level(logging.INFO, logger="write_post")
    pipe.entries = [_entry(r, f"2609.{r:05d}") for r in range(1, 11)]
    pipe.keep_urls = {e.url for e in pipe.entries}
    pipe.no_figure_urls = set()
    pipe.config = {"substack_post": {"max_papers": 15, "trusted_institutions": TRUSTED}}
    assert write_post.main([]) == 0

    assert "lookup_affiliations" not in pipe.calls
    assert _written_urls(pipe) == [e.url for e in pipe.entries]
    (email,) = pipe.emails
    assert "Kept <strong>10</strong> of <strong>10</strong>" in email.html
    assert "Not included" not in email.html and "Institutions" not in email.html
    assert "keep=10 selected=10" in caplog.text


def test_lookup_failure_falls_back_to_h_index_then_rank(pipe, caplog):
    caplog.set_level(logging.INFO, logger="write_post")
    by_rank = _cap_week(pipe)
    pipe.lookup_raises = RuntimeError("OpenRouter down")
    assert write_post.main([]) == 0

    assert len(pipe.calls["lookup_affiliations"]) == 1
    assert "Institution lookup failed (OpenRouter down)" in caplog.text
    # Without tiers, #20 (h-index unknown) loses to #5 (h-index 2).
    assert _written_urls(pipe) == _urls(by_rank, [1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19])
    (email,) = pipe.emails
    assert "Not included: over the 15-paper limit (3)" in email.html
    assert "Institutions: not looked up" in email.html and "Institutions: Inst" not in email.html
    cut_section = email.html.split("Not included", 1)[1]
    assert cut_section.index("Paper 3 &lt;") < cut_section.index("Paper 2 &lt;") < cut_section.index("Paper 20 &lt;")


def test_config_without_cap_keys_uses_defaults(pipe):
    by_rank = _cap_week(pipe, config={"substack_post": {"style_example": None}})
    assert write_post.main([]) == 0
    (((_model, _urls_, _details, trusted), _kw),) = pipe.calls["lookup_affiliations"]
    assert trusted == {"tier1": [], "tier2": []}
    assert _written_urls(pipe) == _urls(by_rank, CAP_SELECTED)


def test_max_papers_zero_means_no_cap(pipe):
    by_rank = _cap_week(pipe, config={"substack_post": {"max_papers": 0, "trusted_institutions": TRUSTED}})
    assert write_post.main([]) == 0
    assert "lookup_affiliations" not in pipe.calls
    assert _written_urls(pipe) == _urls(by_rank, [r for r in range(1, 21) if r not in DROPPED_RANKS])


def test_writer_failure_email_lists_cut_papers(pipe):
    _cap_week(pipe)
    pipe.writer_raises = _writer_error("boom")
    assert write_post.main([]) == 1
    (email,) = pipe.emails
    assert "Papers picked for the post (15)" in email.html
    assert "Not included: over the 15-paper limit (3)" in email.html


def test_dry_run_with_cap_sends_nothing(pipe, capsys, caplog):
    caplog.set_level(logging.INFO, logger="write_post")
    _cap_week(pipe)
    assert write_post.main(["--dry-run"]) == 0
    assert capsys.readouterr().out.startswith("# 15 Ways to Check AI")
    assert pipe.emails == []
    assert "keep=18 selected=15" in caplog.text


def test_paper_cap_settings(caplog):
    defaults = (15, {"tier1": [], "tier2": []})
    assert write_post.paper_cap_settings({}) == defaults
    assert write_post.paper_cap_settings({"substack_post": None}) == defaults
    assert write_post.paper_cap_settings({"substack_post": {"max_papers": None, "trusted_institutions": None}}) == defaults
    assert write_post.paper_cap_settings({"substack_post": {"max_papers": "lots"}}) == defaults
    assert "max_papers is not a number" in caplog.text
    assert write_post.paper_cap_settings(
        {"substack_post": {"max_papers": "8", "trusted_institutions": {"tier1": ["MIT", " "], "tier2": "Mila"}}}
    ) == (8, {"tier1": ["MIT"], "tier2": ["Mila"]})


def test_committed_config_has_the_cap_and_institution_lists():
    max_papers, trusted = write_post.paper_cap_settings(importlib.import_module("src.config").load_config())
    assert max_papers == 15
    assert "Anthropic" in trusted["tier1"] and "University of California, Berkeley" in trusted["tier1"]
    assert "Hugging Face" in trusted["tier2"] and "University of Tübingen" in trusted["tier2"]
    assert len(trusted["tier1"]) == 35 and len(trusted["tier2"]) == 31
    assert not set(trusted["tier1"]) & set(trusted["tier2"])

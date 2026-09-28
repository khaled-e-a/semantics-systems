"""Notification emails for write_post.py: one (subject, html) builder per outcome.

- success: the draft exists on Substack; the email links to the editor.
- failure: the draft could not be created (auth, Cloudflare, not configured,
  other Substack error); the email carries the whole post as HTML and the
  figures go as attachments, so nothing is lost.
- quiet: no post this week (quiet-week report, or no paper passed the filter).

The HTML is deliberately plain and inline-styled (email clients handle
<style> blocks inconsistently). Every piece of text goes through
html.escape(), and links are only emitted for http(s) URLs.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List, Mapping, Optional, Sequence, Tuple

from .models import DigestEntry, DraftResult, Figure, Verdict, VerifyResult

# Failure kinds, in the order write_post.py checks for them.
SKIPPED = "skipped"  # --skip-substack
NOT_CONFIGURED = "not_configured"  # SUBSTACK_PUBLICATION_URL / SUBSTACK_COOKIE missing
AUTH_OR_BLOCK = "auth_or_block"  # SubstackAuthOrBlockError: expired cookie or Cloudflare block
ERROR = "error"  # any other SubstackError (or unexpected error while talking to Substack)


@dataclass
class SubstackFailure:
    """Why the draft was not created, for the failure email."""

    kind: str  # one of SKIPPED, NOT_CONFIGURED, AUTH_OR_BLOCK, ERROR
    message: str
    proxy_in_use: bool = False  # a proxy (WARP or SUBSTACK_PROXY) was set for this run
    updating: bool = False  # --force: the failing call was an update of an existing draft
    block_kind: Optional[str] = None  # SubstackAuthOrBlockError.kind: "cloudflare" | "cookie"
    status_code: Optional[int] = None  # SubstackError.status_code, when known


_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
_LINK_STYLE = "color:#1a56db"
_MUTED = "color:#555;font-size:13px"
_FIG_SOURCE_LABEL = {"html": "arXiv HTML", "pdf": "PDF"}
_CLOUDFLARE_HINT_RE = re.compile(r"cloudflare|just a moment|challenge|\b1010\b|\b1020\b", re.IGNORECASE)


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _is_web_url(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def _a(url: str, inner_html: str) -> str:
    """A link, or just the text when ``url`` is not http(s) (never emit javascript: etc.)."""
    if not _is_web_url(url):
        return inner_html
    return f'<a href="{_e(url)}" style="{_LINK_STYLE}">{inner_html}</a>'


# --------------------------------------------------------------------------
# Markdown (our subset only) → HTML
# --------------------------------------------------------------------------

_IMAGE_LINE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*$")
_INLINE_RE = re.compile(
    r"\*\*\[(?P<bl_text>[^\]]+)\]\((?P<bl_url>[^)\s]+)\)\*\*"  # **[text](url)**
    r"|\[(?P<l_text>[^\]]+)\]\((?P<l_url>[^)\s]+)\)"  # [text](url)
    r"|\*\*(?P<b>.+?)\*\*"  # **bold**
)
_P_STYLE = "margin:0 0 16px;line-height:1.6"
_HEADING_STYLES = {
    1: "font-size:24px;line-height:1.3;margin:0 0 16px",
    2: "font-size:19px;line-height:1.3;margin:24px 0 12px",
}


def _inline_to_html(text: str) -> str:
    out: List[str] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        out.append(_e(text[pos : m.start()]))
        if m.group("bl_text") is not None:
            out.append(_a(m.group("bl_url"), f"<strong>{_e(m.group('bl_text'))}</strong>"))
        elif m.group("l_text") is not None:
            out.append(_a(m.group("l_url"), _e(m.group("l_text"))))
        else:
            out.append(f"<strong>{_inline_to_html(m.group('b'))}</strong>")
        pos = m.end()
    out.append(_e(text[pos:]))
    return "".join(out)


def _image_to_html(alt: str, src: str) -> str:
    if _is_web_url(src):
        return (
            f'<div style="margin:0 0 12px"><img src="{_e(src)}" alt="{_e(alt)}" '
            f'style="max-width:100%;height:auto;display:block">'
            f'<div style="{_MUTED};margin-top:4px">{_e(alt)}</div></div>'
        )
    filename = PurePosixPath(src).name or src
    return (
        '<p style="margin:0 0 12px;padding:8px 12px;background:#f3f4f6;border-radius:4px;font-size:13px">'
        f"\N{PAPERCLIP} figure attached: <strong>{_e(filename)}</strong> — {_e(alt)}</p>"
    )


def markdown_to_html(markdown: str) -> str:
    """Convert the markdown write_post.py produces (and only that) to HTML.

    Supported: ``# heading`` lines, paragraphs, ``**[t](u)**``, ``**b**``,
    ``[t](u)``, and image lines ``![caption](src)``. An image whose src is an
    http(s) URL renders as <img>; any other src is an attachment filename and
    renders as a "figure attached" note.
    """
    blocks: List[str] = []
    paragraph: List[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(f'<p style="{_P_STYLE}">{_inline_to_html(" ".join(paragraph))}</p>')
            paragraph.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        image = _IMAGE_LINE_RE.match(line)
        heading = _HEADING_RE.match(line)
        if image:
            flush()
            blocks.append(_image_to_html(image.group("alt"), image.group("src")))
        elif heading:
            flush()
            level = len(heading.group("hashes"))
            style = _HEADING_STYLES.get(level, _HEADING_STYLES[2])
            blocks.append(f'<h{level} style="{style}">{_inline_to_html(heading.group("text"))}</h{level}>')
        else:
            paragraph.append(line)
    flush()
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Shared sections
# --------------------------------------------------------------------------


def _page(inner: str) -> str:
    return (
        f'<div style="font-family:{_FONT};max-width:680px;margin:0 auto;color:#1a1a1a;'
        f'font-size:15px;line-height:1.5">\n{inner}\n</div>'
    )


def _section_title(text: str) -> str:
    return f'<h3 style="font-size:16px;margin:24px 0 8px">{_e(text)}</h3>'


def _box(inner: str, border: str, background: str) -> str:
    return (
        f'<div style="border-left:4px solid {border};background:{background};padding:12px 16px;'
        f'margin:0 0 16px;border-radius:4px">{inner}</div>'
    )


def _ul(items: Sequence[str]) -> str:
    lis = "".join(f'<li style="margin:0 0 6px">{item}</li>' for item in items)
    return f'<ul style="margin:8px 0 0;padding-left:20px">{lis}</ul>'


def _verify_errors_block(verify: VerifyResult) -> str:
    if not verify.errors:
        return ""
    inner = (
        f'<strong style="color:#b91c1c">{len(verify.errors)} check(s) still fail after the automatic '
        "rewrites. Fix these before publishing:</strong>" + _ul([_e(err) for err in verify.errors])
    )
    return _box(inner, "#dc2626", "#fef2f2")


def _verify_warnings_block(verify: VerifyResult) -> str:
    if not verify.warnings:
        return ""
    return _section_title(f"Warnings ({len(verify.warnings)})") + _ul([_e(w) for w in verify.warnings])


def _drop_entries(all_entries: Sequence[DigestEntry], verdicts: Mapping[str, Verdict]) -> List[DigestEntry]:
    return [e for e in all_entries if not (verdicts.get(e.url) and verdicts[e.url].keep)]


def _drop_items(drops: Sequence[DigestEntry], verdicts: Mapping[str, Verdict]) -> List[str]:
    items = []
    for e in drops:
        verdict = verdicts.get(e.url)
        rationale = verdict.rationale if verdict and verdict.rationale else "no verdict returned"
        items.append(f"{_a(e.url, _e(e.title))}<br><span style=\"{_MUTED}\">{_e(rationale)}</span>")
    return items


def _drop_section(all_entries: Sequence[DigestEntry], verdicts: Mapping[str, Verdict]) -> str:
    """DROP list inside <details> (clients without support just show it expanded)."""
    drops = _drop_entries(all_entries, verdicts)
    if not drops:
        return ""
    return (
        '<details style="margin:24px 0 0">'
        f'<summary style="cursor:pointer;font-weight:600">Dropped papers ({len(drops)}) and why</summary>'
        f"{_ul(_drop_items(drops, verdicts))}</details>"
    )


def _papers_section(
    keep_entries: Sequence[DigestEntry],
    verdicts: Mapping[str, Verdict],
    figs: Mapping[str, Figure],
    attachment_names: Optional[Mapping[str, str]] = None,
    heading: str = "Papers in the post",
) -> str:
    items = []
    for e in keep_entries:
        verdict = verdicts.get(e.url)
        facts = [f"Contribution: {_e(verdict.contribution_type if verdict and verdict.contribution_type else 'unspecified')}"]
        flags: List[str] = []
        fig = figs.get(e.url)
        if fig is None:
            flags.append("No figure found — the paragraph has no image.")
        else:
            source = _FIG_SOURCE_LABEL.get(fig.source, fig.source)
            facts.append(f"Figure {_e(fig.figure_number)} (from {_e(source)})")
            if attachment_names and e.url in attachment_names:
                facts.append(f"attached as {_e(attachment_names[e.url])}")
            if not fig.openly_licensed:
                flags.append("Figure not openly licensed — check the paper's license before publishing.")
        line = f"{_a(e.url, f'<strong>{_e(e.title)}</strong>')}<br><span style=\"{_MUTED}\">{' · '.join(facts)}</span>"
        for flag in flags:
            line += f'<br><span style="color:#b45309;font-size:13px;font-weight:600">{_e(flag)}</span>'
        items.append(line)
    return _section_title(f"{heading} ({len(keep_entries)})") + _ul(items)


def _kept_line(report_date: str, keep_count: int, total: int) -> str:
    return (
        f'<p style="margin:0 0 8px">Kept <strong>{keep_count}</strong> of <strong>{total}</strong> '
        f"papers from the {_e(report_date)} digest.</p>"
    )


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def build_success_email(
    report_date: str,
    post_title: str,
    draft: DraftResult,
    keep_entries: Sequence[DigestEntry],
    all_entries: Sequence[DigestEntry],
    verdicts: Mapping[str, Verdict],
    figs: Mapping[str, Figure],
    verify: VerifyResult,
) -> Tuple[str, str]:
    subject = f"Substack draft ready: {post_title}"
    action = "created" if draft.created else "updated (existing draft, --force)"
    parts = [
        f'<p style="{_MUTED};margin:0 0 8px">Substack draft {_e(action)} for the {_e(report_date)} digest.</p>',
        _verify_errors_block(verify),
        f'<h1 style="font-size:22px;line-height:1.3;margin:0 0 16px">{_e(post_title)}</h1>',
        f'<p style="margin:0 0 8px"><a href="{_e(draft.edit_url)}" style="display:inline-block;padding:12px 22px;'
        'background:#ff6719;color:#ffffff;text-decoration:none;border-radius:6px;font-weight:600;font-size:17px">'
        "Open the draft in the Substack editor</a></p>",
        f'<p style="{_MUTED};margin:0 0 16px">{_a(draft.edit_url, _e(draft.edit_url))}</p>',
        _kept_line(report_date, len(keep_entries), len(all_entries)),
        _papers_section(keep_entries, verdicts, figs),
        _verify_warnings_block(verify),
        _drop_section(all_entries, verdicts),
    ]
    return subject, _page("\n".join(p for p in parts if p))


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------

_REFRESH_COOKIE = (
    "Refresh SUBSTACK_COOKIE: while signed in to Substack, open your browser's DevTools → Network tab, "
    "click any substack.com request, copy the whole Cookie request header, and paste it into the "
    "SUBSTACK_COOKIE repo secret. The cookie expires, and signing out of Substack invalidates it."
)
_USE_WARP = (
    "If the cookie is fresh, Substack's Cloudflare is probably blocking the GitHub runner's datacenter IP: "
    "set the repo variable SUBSTACK_USE_WARP=true (routes Substack calls through Cloudflare WARP), "
    "or set the SUBSTACK_PROXY secret to a proxy URL."
)
_RERUN = (
    "After fixing it, re-run: GitHub → Actions → Substack Draft Post → Run workflow, with report_date set to "
    "this date. Or publish by hand: the post is below and the figures are attached."
)


def fix_hints(failure: SubstackFailure) -> List[str]:
    """Concrete next steps for the failure email, most likely fix first."""
    if failure.kind == SKIPPED:
        return [
            "Nothing to fix: this run used --skip-substack. Paste the post below into a new Substack "
            "draft, or re-run without --skip-substack."
        ]
    if failure.kind == NOT_CONFIGURED:
        return [
            "Set the repo secrets SUBSTACK_PUBLICATION_URL (your https://<name>.substack.com address, not a "
            "custom domain) and SUBSTACK_COOKIE (see the 'Substack draft post' section of the README).",
            _RERUN,
        ]
    if failure.kind == AUTH_OR_BLOCK:
        hints = [_REFRESH_COOKIE, _USE_WARP]
        cloudflare = (
            failure.block_kind == "cloudflare"
            if failure.block_kind in ("cloudflare", "cookie")
            else bool(_CLOUDFLARE_HINT_RE.search(failure.message))
        )
        if cloudflare:
            hints.reverse()
        if failure.proxy_in_use:
            hints.append(
                "A proxy was already in use for this run and Substack still refused the request. If the "
                "cookie is fresh, WARP's IPs are probably blocked too: try a residential proxy in "
                "SUBSTACK_PROXY, or a self-hosted runner."
            )
        hints.append(_RERUN)
        return hints
    hints = []
    if failure.status_code is not None and 300 <= failure.status_code < 400:
        hints.append(
            "Substack redirected the request: set SUBSTACK_PUBLICATION_URL to the canonical "
            "https://<name>.substack.com address (not a custom domain, no trailing path)."
        )
    hints.append("Check the workflow run's log for the full error, then re-run the workflow for this date.")
    if failure.updating:
        hints.append(
            "The error happened while updating the existing draft (--force). If you already published that "
            "draft, remove this date from state/substack_drafts.json and re-run to create a new one."
        )
    hints.append("Or publish by hand: the post is below and the figures are attached.")
    return hints


def build_failure_email(
    report_date: str,
    failure: SubstackFailure,
    post_markdown: str,
    keep_entries: Sequence[DigestEntry],
    all_entries: Sequence[DigestEntry],
    verdicts: Mapping[str, Verdict],
    figs: Mapping[str, Figure],
    verify: VerifyResult,
    attachment_names: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str]:
    subject = f"Substack draft NOT created — {report_date}"
    reason_box = _box(
        f'<strong style="color:#b91c1c;font-size:16px">The Substack draft was not created.</strong>'
        f'<p style="margin:8px 0 0"><strong>Reason:</strong> {_e(failure.message)}</p>'
        f'<p style="margin:8px 0 0"><strong>How to fix:</strong></p>'
        + _ul([_e(h) for h in fix_hints(failure)]),
        "#dc2626",
        "#fef2f2",
    )
    post_block = (
        _section_title("The post (copy it into Substack; figures are attached)")
        + '<div style="border:1px solid #e5e7eb;border-radius:6px;padding:20px">'
        + markdown_to_html(post_markdown)
        + "</div>"
    )
    parts = [
        f'<p style="{_MUTED};margin:0 0 8px">Substack draft post for the {_e(report_date)} digest.</p>',
        reason_box,
        _verify_errors_block(verify),
        _kept_line(report_date, len(keep_entries), len(all_entries)),
        _papers_section(keep_entries, verdicts, figs, attachment_names),
        _verify_warnings_block(verify),
        post_block,
        _drop_section(all_entries, verdicts),
    ]
    return subject, _page("\n".join(p for p in parts if p))


def build_writer_failure_email(
    report_date: str,
    reason: str,
    keep_entries: Sequence[DigestEntry],
    all_entries: Sequence[DigestEntry],
    verdicts: Mapping[str, Verdict],
    figs: Mapping[str, Figure],
    attachment_names: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str]:
    """The LLM never produced a usable post: no draft, no post file."""
    subject = f"Substack post NOT written — {report_date}"
    reason_box = _box(
        '<strong style="color:#b91c1c;font-size:16px">Post generation failed, so no draft was created.</strong>'
        f'<p style="margin:8px 0 0"><strong>Reason:</strong> post generation failed: {_e(reason)}</p>'
        '<p style="margin:8px 0 0"><strong>How to fix:</strong></p>'
        + _ul(
            [
                _e(
                    "LLM errors are often temporary: re-run it (GitHub → Actions → Substack Draft Post → Run "
                    "workflow, with report_date set to this date)."
                ),
                _e(
                    "If it fails again, check the workflow log and try another model in the "
                    "SUBSTACK_WRITER_MODEL secret."
                ),
            ]
        ),
        "#dc2626",
        "#fef2f2",
    )
    parts = [
        f'<p style="{_MUTED};margin:0 0 8px">Substack draft post for the {_e(report_date)} digest.</p>',
        reason_box,
        _kept_line(report_date, len(keep_entries), len(all_entries)),
        _papers_section(keep_entries, verdicts, figs, attachment_names, heading="Papers that passed the filter"),
        _drop_section(all_entries, verdicts),
    ]
    return subject, _page("\n".join(p for p in parts if p))


# --------------------------------------------------------------------------
# Quiet
# --------------------------------------------------------------------------


def build_quiet_email(
    report_date: str,
    all_entries: Sequence[DigestEntry],
    verdicts: Optional[Mapping[str, Verdict]] = None,
) -> Tuple[str, str]:
    """No post: a quiet-week report (no entries) or no paper passed the filter."""
    verdicts = verdicts or {}
    subject = f"No Substack post this week — {report_date}"
    if not all_entries:
        parts = [
            f'<p style="margin:0 0 8px">The {_e(report_date)} digest was a quiet week: no papers matched, '
            "so there is no Substack post and no draft was created.</p>"
        ]
    else:
        drops = _drop_entries(all_entries, verdicts)
        parts = [
            f'<p style="margin:0 0 8px">None of the {len(all_entries)} papers in the {_e(report_date)} digest '
            "passed the strict VV/UQ filter, so no draft was created.</p>",
            _section_title(f"Dropped papers ({len(drops)}) and why"),
            _ul(_drop_items(drops, verdicts)),
        ]
    return subject, _page("\n".join(parts))

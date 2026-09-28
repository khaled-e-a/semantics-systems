#!/usr/bin/env python3
"""Weekly digest → Substack draft post — orchestrator entrypoint.

Turns reports/YYYY-MM-DD.md into a post that follows WRITING_SUBSTACK_POST.md,
puts each paper's main figure above its paragraph, saves it as a Substack
DRAFT, and emails a link to review and publish. If Substack refuses (expired
cookie, Cloudflare block, not configured) the email carries the whole post
with the figures attached instead. See README.md ("Substack draft post").

Run `python write_post.py --dry-run --verbose` to print the post without
touching Substack, email, the post file, or state.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import PROJECT_ROOT, load_config, load_env_vars
from src.email_sender import attachment_from_path, send_digest_email
from src.substack_post import (
    draft_state,
    figures,
    notify,
    paper_details,
    prompts,
    prosemirror,
    render,
    report_parser,
    substack_client,
    writer,
)
from src.substack_post.models import DigestEntry, DraftResult, Figure, PostDraft, UploadedImage, VerifyResult

REPORTS_DIR_NAME = "reports"
POST_SUFFIX = "-substack-post-ste100.md"

# Module-level so tests can point them at tmp_path.
REPORTS_DIR = PROJECT_ROOT / REPORTS_DIR_NAME
DRAFTS_STATE_PATH = draft_state.DRAFTS_STATE_PATH

REQUIRED_ENV_VARS = ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"]
EMAIL_ENV_VARS = ["RESEND_API_KEY", "REPORT_EMAIL_FROM", "REPORT_EMAIL_TO"]
OPTIONAL_ENV_VARS = ["SUBSTACK_WRITER_MODEL", "SUBSTACK_PUBLICATION_URL", "SUBSTACK_COOKIE", "SUBSTACK_PROXY"]

logger = logging.getLogger("write_post")


def _report_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}")
    return value


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=_report_date, default=None, help="Report date (YYYY-MM-DD). Default: newest report in reports/.")
    parser.add_argument("--force", action="store_true", help="Run even if this date already has a draft; UPDATES that draft instead of creating a duplicate.")
    parser.add_argument("--dry-run", action="store_true", help="Print the post (local figure paths) to stdout; no Substack, email, post file, or state writes.")
    parser.add_argument("--skip-substack", action="store_true", help="Don't call Substack; write the post file and send the 'not created' email with figures attached.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def attachment_name(entry: DigestEntry, fig: Figure) -> str:
    """Stable, filesystem-safe attachment filename: fig-<arxiv id or rank>.<ext>."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", entry.arxiv_id or f"rank{entry.rank}")
    return f"fig-{stem}{fig.path.suffix.lower() or '.png'}"


def _publish_draft(
    client: "substack_client.SubstackClient",
    post: PostDraft,
    keep: List[DigestEntry],
    figs: Dict[str, Figure],
    existing: Optional[Dict],
    force: bool,
) -> Tuple[DraftResult, str, str]:
    """Upload figures, build the ProseMirror doc, create or update the draft.

    Returns (draft result, CDN-image markdown, post title).
    """
    uploaded: Dict[str, UploadedImage] = {}
    for entry in keep:
        fig = figs.get(entry.url)
        if fig is None:
            continue
        uploaded[entry.url] = client.upload_image(fig.path)
        logger.info("Uploaded figure for %s -> %s", entry.url, uploaded[entry.url].url)

    md_cdn = render.render_markdown(post, keep, {url: (img.url, figs[url].credit_caption) for url, img in uploaded.items()})
    title, body = prosemirror.split_title(md_cdn)
    doc = prosemirror.markdown_to_doc(body, {img.url: img for img in uploaded.values()})

    if force and existing:
        draft_id = int(existing["draft_id"])
        try:
            result = client.update_draft(draft_id, title, "", doc)
        except substack_client.SubstackError as exc:
            # The user deleted the draft in Substack: make a new one instead of failing.
            if isinstance(exc, substack_client.SubstackAuthOrBlockError) or getattr(exc, "status_code", None) != 404:
                raise
            logger.warning("Draft %s no longer exists on Substack (404); creating a new draft", draft_id)
            result = client.create_draft(title, "", doc)
    else:
        result = client.create_draft(title, "", doc)
    return result, md_cdn, title


def load_style_example(config: Dict, report_date: str) -> Optional[str]:
    """The writer's style example: config's substack_post.style_example (a hand-approved
    post) when set, else the newest *-substack-post-ste100.md other than this date's.

    A configured file for this same date is skipped (it would hand the writer the answer).
    A configured file that can't be read gives no example rather than the newest post,
    which may be the pipeline's own unreviewed output.
    """
    configured = (config.get("substack_post") or {}).get("style_example")
    if not configured:
        return prompts.load_style_example(REPORTS_DIR, exclude_date=report_date)
    path = Path(configured)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.name.startswith(report_date):
        logger.info("Configured style example %s is this week's post; using the newest other post", path.name)
        return prompts.load_style_example(REPORTS_DIR, exclude_date=report_date)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Configured style example unreadable (%s); writing without one", exc)
        return None


def _send(env: Dict[str, str], subject: str, html: str, attachments: Optional[List[Dict[str, str]]] = None) -> bool:
    sent = send_digest_email(
        env["RESEND_API_KEY"], env["REPORT_EMAIL_FROM"], env["REPORT_EMAIL_TO"], subject, html, attachments=attachments
    )
    logger.info("Email sent: %s (%s)", sent, subject)
    return sent


def _quiet(env: Dict[str, str], dry_run: bool, report_date: str, entries: List[DigestEntry], verdicts: Dict) -> int:
    subject, html = notify.build_quiet_email(report_date, entries, verdicts)
    sent: Optional[bool] = None
    if dry_run:
        logger.info("Dry run: would send %r", subject)
    else:
        sent = _send(env, subject, html)
    _log_summary(report_date, entries, [], {}, VerifyResult(), "none(no post)", sent)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Fail fast, before any network call. Email vars are only needed when we send.
    required = REQUIRED_ENV_VARS + ([] if args.dry_run else EMAIL_ENV_VARS)
    optional = OPTIONAL_ENV_VARS + (EMAIL_ENV_VARS if args.dry_run else [])
    env = load_env_vars(required, optional)
    config = load_config()
    model = env.get("SUBSTACK_WRITER_MODEL") or env["OPENROUTER_MODEL"]

    try:
        report_path = report_parser.find_report(REPORTS_DIR, args.date)
    except FileNotFoundError as exc:
        logger.error("No digest report found: %s", exc)
        return 1
    report_date = report_path.stem
    logger.info("Report: %s (writer model: %s)", report_path, model)

    drafts = draft_state.load_drafts(DRAFTS_STATE_PATH)
    existing = draft_state.get_draft(drafts, report_date)
    if existing and not args.force and not args.dry_run:
        logger.info("Already drafted: %s (use --force to update it)", existing.get("edit_url"))
        return 0

    # 1. Parse the digest.
    entries = report_parser.parse_report(report_path)
    logger.info("Parsed %d entries", len(entries))
    if not entries:
        logger.info("Quiet week — no post")
        return _quiet(env, args.dry_run, report_date, [], {})

    # 2. Real abstracts/authors, then the strict KEEP/DROP filter.
    details = paper_details.fetch_details(entries)
    llm = writer.make_client(env["OPENROUTER_API_KEY"])
    verdicts = writer.filter_papers(llm, model, entries, details)
    missing_verdicts = [e.url for e in entries if e.url not in verdicts]
    if missing_verdicts:
        logger.warning("No verdict for %d entries (treated as DROP): %s", len(missing_verdicts), missing_verdicts)
    keep = [e for e in entries if e.url in verdicts and verdicts[e.url].keep]
    logger.info("KEEP %d of %d", len(keep), len(entries))
    if not keep:
        logger.info("No paper passed the filter — no post")
        return _quiet(env, args.dry_run, report_date, entries, verdicts)

    # 3. Licenses (email flags only) and main figures. Figure files live in a temp dir: never committed.
    paper_details.fetch_licenses(details, [e.url for e in keep])
    figdir = Path(tempfile.mkdtemp(prefix="substack-figures-"))
    figs = figures.fetch_main_figures(
        keep, details, figdir, choose=lambda captions: writer.choose_figures(llm, model, captions)
    )
    logger.info("Figures: %d of %d KEEP papers", len(figs), len(keep))

    names = {e.url: attachment_name(e, figs[e.url]) for e in keep if e.url in figs}

    # 4. Write + verify (the writer retries on hard verify errors).
    style_example = load_style_example(config, report_date)
    writer_error = getattr(writer, "WriterError", RuntimeError)
    try:
        post, verify = writer.write_post(llm, model, keep, verdicts, details, entries, style_example=style_example)
    except writer_error as exc:
        # No LLM attempt returned a usable post. Tell the user, then fail the run (red in Actions).
        logger.error("Post generation failed: %s", exc)
        if not args.dry_run:
            subject, html = notify.build_writer_failure_email(
                report_date, str(exc) or type(exc).__name__, keep, entries, verdicts, figs, attachment_names=names
            )
            _send(env, subject, html, [attachment_from_path(figs[url].path, name) for url, name in names.items()])
            shutil.rmtree(figdir, ignore_errors=True)
        return 1
    logger.info("Post: %r — verify errors=%d warnings=%d", post.title, len(verify.errors), len(verify.warnings))

    if args.dry_run:
        print(render.render_markdown(post, keep, {url: (str(fig.path), fig.credit_caption) for url, fig in figs.items()}))
        logger.info("Dry run: figures in %s", figdir)
        logger.info("Dry run: verify errors=%s warnings=%s", verify.errors, verify.warnings)
        _log_summary(report_date, entries, keep, figs, verify, "dry-run", None)
        return 0

    # 5. Substack draft. Every handled failure falls through to the failure email.
    pub = env.get("SUBSTACK_PUBLICATION_URL")
    cookie = env.get("SUBSTACK_COOKIE")
    proxy = env.get("SUBSTACK_PROXY")
    updating = bool(args.force and existing)
    failure: Optional[notify.SubstackFailure] = None
    draft: Optional[DraftResult] = None
    unexpected = False
    client = None
    md_cdn = title = ""
    if args.skip_substack:
        failure = notify.SubstackFailure(notify.SKIPPED, "Substack upload skipped (--skip-substack).")
    elif not (pub and cookie):
        failure = notify.SubstackFailure(
            notify.NOT_CONFIGURED, "Substack not configured: SUBSTACK_PUBLICATION_URL and/or SUBSTACK_COOKIE is not set."
        )
    else:
        try:
            client = substack_client.SubstackClient(pub, cookie, proxy=proxy)
        except ValueError as exc:  # empty or malformed URL/cookie: a config problem, not a crash
            failure = notify.SubstackFailure(notify.NOT_CONFIGURED, f"Substack misconfigured: {exc}")
    if client is not None:
        try:
            draft, md_cdn, title = _publish_draft(client, post, keep, figs, existing, args.force)
            logger.info("Draft %s: %s", "created" if draft.created else "updated", draft.edit_url)
        except substack_client.SubstackError as exc:
            auth_or_block = isinstance(exc, substack_client.SubstackAuthOrBlockError)
            logger.error("Substack %s: %s", "refused the request (auth or Cloudflare)" if auth_or_block else "error", exc)
            failure = notify.SubstackFailure(
                notify.AUTH_OR_BLOCK if auth_or_block else notify.ERROR,
                str(exc) or type(exc).__name__,
                proxy_in_use=bool(proxy),
                updating=updating,
                block_kind=getattr(exc, "kind", None),
                status_code=getattr(exc, "status_code", None),
            )
        except Exception as exc:  # noqa: BLE001 — salvage the post by email, then still exit non-zero
            logger.exception("Unexpected error while creating the Substack draft")
            failure = notify.SubstackFailure(
                notify.ERROR,
                f"Unexpected error while talking to Substack: {exc!r}",
                proxy_in_use=bool(proxy),
                updating=updating,
            )
            unexpected = True
    if failure:
        logger.warning("Draft NOT created: %s", failure.message)

    # 6. Post file: CDN images on success, attachment filenames otherwise.
    if draft is not None:
        post_md = md_cdn
    else:
        post_md = render.render_markdown(post, keep, {url: (names[url], figs[url].credit_caption) for url in names})
    post_path = REPORTS_DIR / f"{report_date}{POST_SUFFIX}"
    post_path.write_text(post_md if post_md.endswith("\n") else post_md + "\n", encoding="utf-8")
    logger.info("Wrote %s", post_path)

    # 7. Idempotency state (before the email, so a re-run never duplicates the draft).
    if draft is not None:
        draft_state.record_draft(drafts, report_date, draft, title or post.title)
        draft_state.save_drafts(drafts, DRAFTS_STATE_PATH)
        logger.info("State updated")

    # 8. Email (best effort).
    if draft is not None:
        subject, html = notify.build_success_email(
            report_date, title or post.title, draft, keep, entries, verdicts, figs, verify
        )
        sent = _send(env, subject, html)
    else:
        subject, html = notify.build_failure_email(
            report_date, failure, post_md, keep, entries, verdicts, figs, verify, attachment_names=names
        )
        attachments = [attachment_from_path(figs[url].path, name) for url, name in names.items()]
        sent = _send(env, subject, html, attachments)

    shutil.rmtree(figdir, ignore_errors=True)

    outcome = ("updated" if not draft.created else "created") if draft is not None else f"failed({failure.kind})"
    _log_summary(report_date, entries, keep, figs, verify, outcome, sent)
    return 1 if unexpected else 0


def _log_summary(
    report_date: str,
    entries: List[DigestEntry],
    keep: List[DigestEntry],
    figs: Dict[str, Figure],
    verify: VerifyResult,
    substack: str,
    email_sent: Optional[bool],
) -> None:
    logger.info(
        "Run summary: date=%s entries=%d keep=%d figures=%d/%d verify_errors=%d verify_warnings=%d "
        "substack=%s email_sent=%s",
        report_date,
        len(entries),
        len(keep),
        len(figs),
        len(keep),
        len(verify.errors),
        len(verify.warnings),
        substack,
        email_sent,
    )


if __name__ == "__main__":
    sys.exit(main())

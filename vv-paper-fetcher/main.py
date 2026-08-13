#!/usr/bin/env python3
"""Weekly VV/UQ/evals paper fetcher — orchestrator entrypoint.

See README.md and config.yaml. Run `python main.py --dry-run --verbose` for
a full pipeline smoke test that skips email sending and state/report writes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from src.collectors.base import collect_all
from src.config import load_config, load_env
from src.dedup import merge_duplicates
from src.keyword_filter import apply_keyword_filter
from src.llm_triage import triage_papers
from src.models import RunStats
from src.ranking import score_and_rank
from src.reputation import enrich_with_reputation
from src.report import build_report_context, render_html, render_markdown
from src.state import filter_unseen, load_state, record_seen, save_state
from src.email_sender import send_digest_email

REPORTS_DIR_NAME = "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip email sending and report/state writes; print report to stdout.")
    parser.add_argument("--since", type=str, default=None, help="Window start date (YYYY-MM-DD). Default: 7 days before --until.")
    parser.add_argument("--until", type=str, default=None, help="Window end date (YYYY-MM-DD). Default: today (UTC).")
    parser.add_argument("--top-n", type=int, default=None, help="Override config's report.top_n.")
    parser.add_argument("--sources", type=str, default=None, help="Comma-separated list of collectors to run (default: all enabled).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def compute_window(since: str | None, until: str | None) -> tuple[date, date]:
    window_end = datetime.strptime(until, "%Y-%m-%d").date() if until else datetime.utcnow().date()
    window_start = datetime.strptime(since, "%Y-%m-%d").date() if since else window_end - timedelta(days=7)
    return window_start, window_end


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("main")

    env = load_env(dry_run=args.dry_run)
    config = load_config()

    window_start, window_end = compute_window(args.since, args.until)
    logger.info("Window: %s to %s", window_start, window_end)

    only_sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    stats = RunStats()
    state = load_state()

    papers = collect_all(config, window_start, window_end, state, stats, only_sources)
    logger.info("Collected %d raw papers (%s)", len(papers), stats.collected_per_source)

    papers = merge_duplicates(papers)
    stats.after_in_run_dedup = len(papers)
    logger.info("After in-run dedup: %d", len(papers))

    papers = apply_keyword_filter(papers, config.get("keyword_filter", {}))
    stats.after_keyword_filter = len(papers)
    logger.info("After keyword filter: %d", len(papers))

    papers = filter_unseen(papers, state)
    stats.after_cross_run_dedup = len(papers)
    logger.info("After cross-run dedup: %d", len(papers))

    triaged = triage_papers(papers, env["OPENROUTER_API_KEY"], env["OPENROUTER_MODEL"], config.get("llm_triage", {}), stats)
    relevant_count = sum(1 for t in triaged if t.relevant)
    logger.info("After LLM triage: %d relevant of %d triaged", relevant_count, len(triaged))

    reputation = enrich_with_reputation(triaged, env.get("SEMANTIC_SCHOLAR_API_KEY"), stats)

    report_cfg = dict(config.get("report", {}))
    if args.top_n is not None:
        report_cfg["top_n"] = args.top_n

    ranked = score_and_rank(triaged, reputation, config.get("scoring", {}), report_cfg, stats)
    logger.info("Final top-N: %d", len(ranked["top"]))

    context = build_report_context(ranked, stats, window_start, window_end)
    markdown = render_markdown(context)

    if args.dry_run:
        print(markdown)
    else:
        report_path = f"{REPORTS_DIR_NAME}/{window_end.isoformat()}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Wrote %s", report_path)

        html = render_html(context)
        subject = f"VV/UQ/Evals Paper Digest — {window_end.isoformat()}"
        sent = send_digest_email(env["RESEND_API_KEY"], env["REPORT_EMAIL_FROM"], env["REPORT_EMAIL_TO"], subject, html)
        logger.info("Email sent: %s", sent)

        record_seen(triaged, state, window_end.isoformat())
        save_state(state)
        logger.info("State updated")

    logger.info(
        "Run summary: collected=%s errors=%s in_run_dedup=%d keyword_filter=%d cross_run_dedup=%d "
        "llm_batches=%d/%d_failed relevant=%d final_top_n=%d reputation_errors=%d",
        stats.collected_per_source,
        stats.source_errors,
        stats.after_in_run_dedup,
        stats.after_keyword_filter,
        stats.after_cross_run_dedup,
        stats.llm_batches_total,
        stats.llm_batches_failed,
        relevant_count,
        stats.final_top_n,
        stats.reputation_lookup_errors,
    )

    total_sources = len(stats.collected_per_source) or 1
    if len(stats.source_errors) >= total_sources and total_sources > 0:
        logger.error("Every source failed — likely a systemic issue (network, config). Failing the run.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

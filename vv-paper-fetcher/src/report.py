"""Report rendering: one shared context feeds two separate Jinja2 templates
(Markdown for the committed report file, HTML for the email body). Kept as
two templates rather than converting Markdown->HTML, because the email body
needs inline-style/table markup for email-client compatibility that a
generic converter wouldn't give cleanly.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import RunStats, ScoredPaper

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(enabled_extensions=("html",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_report_context(
    ranked: Dict[str, List[ScoredPaper]],
    stats: RunStats,
    window_start: date,
    window_end: date,
) -> Dict[str, Any]:
    top = ranked["top"]
    honorable = ranked["honorable_mentions"]

    return {
        "window_start": window_start,
        "window_end": window_end,
        "generated_at": datetime.utcnow(),
        "top": top,
        "honorable_mentions": honorable,
        "collected_per_source": stats.collected_per_source,
        "source_errors": stats.source_errors,
        "llm_batches_total": stats.llm_batches_total,
        "llm_batches_failed": stats.llm_batches_failed,
        "quiet_week": len(top) == 0,
    }


def render_markdown(context: Dict[str, Any]) -> str:
    template = _env.get_template("report.md.j2")
    return template.render(**context)


def render_html(context: Dict[str, Any]) -> str:
    template = _env.get_template("report.html.j2")
    return template.render(**context)

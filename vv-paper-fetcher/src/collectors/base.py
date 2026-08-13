"""Fault-isolated fan-out across all enabled source collectors.

Each collector's fetch() is called independently; a timeout, malformed
response, or any other exception in one source is caught here, logged, and
contributes zero papers — it never aborts collection from the other sources.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Tuple

from ..models import Paper, RunStats
from . import arxiv, dblp, hf_papers, openreview

logger = logging.getLogger(__name__)

COLLECTORS = {
    "arxiv": arxiv.fetch,
    "hf_papers": hf_papers.fetch,
    "openreview": openreview.fetch,
    "dblp": dblp.fetch,
}


def collect_all(
    config: Dict[str, Any],
    window_start: date,
    window_end: date,
    state: Dict[str, Any],
    stats: RunStats,
    only_sources: List[str] | None = None,
) -> List[Paper]:
    all_papers: List[Paper] = []
    sources_cfg = config.get("sources", {})

    for name, fetch_fn in COLLECTORS.items():
        if only_sources and name not in only_sources:
            continue
        src_cfg = sources_cfg.get(name, {})
        if not src_cfg.get("enabled", True):
            continue
        try:
            papers = fetch_fn(src_cfg, window_start, window_end, state)
            stats.collected_per_source[name] = len(papers)
            all_papers.extend(papers)
        except Exception as exc:  # noqa: BLE001 - deliberate: one source must never kill the run
            logger.warning("Collector %s failed: %s", name, exc)
            stats.collected_per_source[name] = 0
            stats.source_errors[name] = str(exc)

    return all_papers

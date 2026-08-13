"""Config-driven keyword pre-filter applied to title+abstract before the
(expensive) LLM triage step, to control cost/volume.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .models import Paper


def apply_keyword_filter(papers: List[Paper], filter_cfg: Dict[str, Any]) -> List[Paper]:
    terms = filter_cfg.get("terms", [])
    min_matches = filter_cfg.get("min_matches", 1)
    case_sensitive = filter_cfg.get("case_sensitive", False)
    skip_for_sources = set(filter_cfg.get("skip_for_sources", []))

    if not case_sensitive:
        terms = [t.lower() for t in terms]

    kept: List[Paper] = []
    for paper in papers:
        if paper.source in skip_for_sources:
            kept.append(paper)
            continue

        haystack = f"{paper.title} {paper.abstract}"
        if not case_sensitive:
            haystack = haystack.lower()

        matches = sum(1 for term in terms if term in haystack)
        if matches >= min_matches:
            kept.append(paper)

    return kept

"""Cross-run dedup state: which papers have already been reported.

Also tracks DBLP's per-venue known-publication-key sets, since DBLP has no
per-paper date to window by — new papers there are found by set difference
against previously-seen keys (see collectors/dblp.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import Paper, TriagedPaper

STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "seen_papers.json"


def load_state(path: Path | None = None) -> Dict[str, Any]:
    p = path or STATE_PATH
    if not p.exists():
        return {"seen_papers": {}, "dblp_seen_keys": {}}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("seen_papers", {})
    data.setdefault("dblp_seen_keys", {})
    return data


def save_state(state: Dict[str, Any], path: Path | None = None) -> None:
    p = path or STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def filter_unseen(papers: List[Paper], state: Dict[str, Any]) -> List[Paper]:
    seen = state["seen_papers"]
    return [p for p in papers if p.dedup_key() not in seen]


def record_seen(triaged: Iterable[TriagedPaper], state: Dict[str, Any], report_date: str) -> None:
    """Record every triaged-relevant paper as seen, not just the top-N that made the cut.

    This is a deliberate tradeoff (documented in the design plan): a paper that
    narrowly misses the report cutoff is never re-reported even if a later
    week's reputation data would rank it higher.
    """
    for t in triaged:
        if not t.relevant:
            continue
        key = t.paper.dedup_key()
        state["seen_papers"][key] = {"first_seen_report_date": report_date, "title": t.paper.title}

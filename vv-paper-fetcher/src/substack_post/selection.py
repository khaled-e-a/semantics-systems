"""Cap the post at ``max_papers``: keep the KEEP papers from the most trusted institutions.

Ranking, best first:
1. institution tier: 1, then 2, then 0 (no trusted institution, or unknown);
2. max author h-index from the digest report (higher first; unknown last);
3. report rank (lower first).

The post itself keeps the report's ranked order, so the selected papers are
returned sorted by report rank. The cut papers are returned best first, so
the email can list the best candidates to swap in by hand.

``affiliations`` maps paper URL to any object with a ``tier`` attribute
(affiliations.Affiliations). This module does not import it, so it can be
tested on its own.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Tuple

from .models import DigestEntry

_TIER_ORDER = {1: 0, 2: 1}  # any other tier (0 = no match or unknown) sorts after these


def tier_of(affiliation: Any) -> int:
    """1, 2, or 0 (no match, unknown, missing, or a value we don't understand)."""
    tier = getattr(affiliation, "tier", 0)
    try:
        tier = int(tier)
    except (TypeError, ValueError):
        return 0
    return tier if tier in _TIER_ORDER else 0


def ranking_key(entry: DigestEntry, affiliations: Mapping[str, Any]) -> Tuple[int, bool, int, int]:
    """Sort key, best first: tier, then h-index (None last), then report rank."""
    tier = tier_of(affiliations.get(entry.url))
    h_index = entry.max_h_index
    return (_TIER_ORDER.get(tier, len(_TIER_ORDER)), h_index is None, -(h_index or 0), entry.rank)


def select_papers(
    keep: List[DigestEntry], affiliations: Mapping[str, Any], max_papers: int
) -> Tuple[List[DigestEntry], List[DigestEntry]]:
    """Return (selected in report order, cut in ranking order).

    ``max_papers <= 0`` means no cap. With ``len(keep) <= max_papers`` every
    paper is selected, unchanged, and nothing is cut.
    """
    if max_papers <= 0 or len(keep) <= max_papers:
        return list(keep), []
    affiliations = affiliations or {}
    ranked = sorted(keep, key=lambda e: ranking_key(e, affiliations))
    selected = sorted(ranked[:max_papers], key=lambda e: e.rank)
    return selected, ranked[max_papers:]

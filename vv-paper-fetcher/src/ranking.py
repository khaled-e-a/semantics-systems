"""Composite scoring and ranking.

LLM relevance is a gate (papers below min_relevance_score are excluded
before this point, in llm_triage.py) — reputation signals here only reorder
the already-relevant set. Score combines LLM relevance, author h-index
(the load-bearing reputation signal, since a brand-new paper has ~0
citations of its own), the paper's own citation count, HF upvotes (0 if not
HF-sourced), and a flat bonus for configured top venues.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .models import RunStats, ScoredPaper, TriagedPaper


def _normalize(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return min(value, cap) / cap


def score_and_rank(
    triaged: List[TriagedPaper],
    reputation: Dict[str, Dict[str, Any]],
    scoring_cfg: Dict[str, Any],
    report_cfg: Dict[str, Any],
    stats: RunStats,
) -> Dict[str, List[ScoredPaper]]:
    weights = scoring_cfg.get("weights", {})
    caps = scoring_cfg.get("caps", {})
    venue_bonus_cfg = scoring_cfg.get("venue_tier_bonus", {})
    top_venues = {v.lower() for v in venue_bonus_cfg.get("top_venues", [])}
    bonus = venue_bonus_cfg.get("bonus", 0.0)

    scored: List[ScoredPaper] = []
    for t in triaged:
        if not t.relevant:
            continue
        rep = reputation.get(t.paper.dedup_key(), {})
        max_h = rep.get("max_h_index", 0)
        citations = rep.get("citationCount", 0)
        upvotes = t.paper.extra.get("hf_upvotes", 0) or 0

        composite = (
            weights.get("llm_relevance", 0) * (t.relevance_score / 5.0)
            + weights.get("author_h_index", 0) * _normalize(max_h, caps.get("hindex_cap", 80))
            + weights.get("paper_citations", 0) * _normalize(citations, caps.get("citation_cap", 200))
            + weights.get("hf_upvotes", 0) * _normalize(upvotes, caps.get("upvotes_cap", 100))
        )
        if t.paper.venue.lower() in top_venues:
            composite += bonus

        scored.append(
            ScoredPaper(
                triaged=t,
                max_author_h_index=max_h,
                avg_author_h_index=rep.get("avg_h_index", 0.0),
                paper_citation_count=citations,
                composite_score=composite,
                reputation_unmatched=rep.get("unmatched", False),
            )
        )

    scored.sort(key=lambda s: s.composite_score, reverse=True)

    top_n = report_cfg.get("top_n", 20)
    honorable_n = report_cfg.get("include_honorable_mentions_count", 10)

    top = scored[:top_n]
    honorable = scored[top_n : top_n + honorable_n]
    stats.final_top_n = len(top)

    return {"top": top, "honorable_mentions": honorable}

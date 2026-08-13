from datetime import date

from src.models import Paper, RunStats, TriagedPaper
from src.ranking import score_and_rank

SCORING_CFG = {
    "weights": {"llm_relevance": 0.4, "author_h_index": 0.3, "paper_citations": 0.1, "hf_upvotes": 0.1},
    "caps": {"hindex_cap": 80, "citation_cap": 200, "upvotes_cap": 100},
    "venue_tier_bonus": {"bonus": 0.15, "top_venues": ["NeurIPS"]},
}
REPORT_CFG = {"top_n": 2, "include_honorable_mentions_count": 5}


def _triaged(title, venue="cs.AI", relevance_score=5, relevant=True, key_suffix="1"):
    paper = Paper(
        title=title,
        authors=["Someone"],
        abstract="",
        url="https://example.com",
        venue=venue,
        source="arxiv",
        published_date=date(2026, 8, 4),
        arxiv_id=f"2401.0000{key_suffix}",
    )
    return TriagedPaper(paper=paper, relevant=relevant, relevance_score=relevance_score, summary="s", tags=[])


def test_irrelevant_papers_are_excluded():
    t = _triaged("Not relevant", relevant=False)
    result = score_and_rank([t], {}, SCORING_CFG, REPORT_CFG, RunStats())
    assert result["top"] == []


def test_higher_h_index_ranks_above_lower_h_index():
    low = _triaged("Low reputation paper", key_suffix="1")
    high = _triaged("High reputation paper", key_suffix="2")
    reputation = {
        low.paper.dedup_key(): {"citationCount": 0, "max_h_index": 2, "avg_h_index": 2, "unmatched": False},
        high.paper.dedup_key(): {"citationCount": 0, "max_h_index": 60, "avg_h_index": 60, "unmatched": False},
    }

    result = score_and_rank([low, high], reputation, SCORING_CFG, REPORT_CFG, RunStats())

    assert [s.triaged.paper.title for s in result["top"]] == ["High reputation paper", "Low reputation paper"]


def test_venue_bonus_applied_for_top_venue():
    t = _triaged("NeurIPS paper", venue="NeurIPS", key_suffix="3")
    result = score_and_rank([t], {}, SCORING_CFG, REPORT_CFG, RunStats())
    assert result["top"][0].composite_score > SCORING_CFG["weights"]["llm_relevance"] * 1.0


def test_top_n_and_honorable_mentions_cutoff():
    triaged = [_triaged(f"Paper {i}", key_suffix=str(i)) for i in range(5)]
    stats = RunStats()

    result = score_and_rank(triaged, {}, SCORING_CFG, REPORT_CFG, stats)

    assert len(result["top"]) == 2
    assert len(result["honorable_mentions"]) == 3
    assert stats.final_top_n == 2

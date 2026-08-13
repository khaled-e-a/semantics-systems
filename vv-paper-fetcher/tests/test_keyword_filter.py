from datetime import date

from src.keyword_filter import apply_keyword_filter
from src.models import Paper

FILTER_CFG = {
    "min_matches": 1,
    "case_sensitive": False,
    "skip_for_sources": ["dblp"],
    "terms": ["uncertainty quantification", "formal verification", "agent"],
}


def _paper(**overrides):
    defaults = dict(
        title="A Survey",
        authors=["Someone"],
        abstract="",
        url="https://example.com",
        venue="cs.AI",
        source="arxiv",
        published_date=date(2026, 8, 4),
    )
    defaults.update(overrides)
    return Paper(**defaults)


def test_keeps_paper_matching_a_term_case_insensitively():
    p = _paper(title="Uncertainty Quantification for Autonomous AI Agents", abstract="")
    kept = apply_keyword_filter([p], FILTER_CFG)
    assert kept == [p]


def test_drops_paper_matching_no_terms():
    p = _paper(title="A Study of Sorting Algorithms", abstract="Nothing relevant here.")
    kept = apply_keyword_filter([p], FILTER_CFG)
    assert kept == []


def test_dblp_source_bypasses_filter_regardless_of_content():
    p = _paper(title="Irrelevant Title", abstract="", source="dblp")
    kept = apply_keyword_filter([p], FILTER_CFG)
    assert kept == [p]


def test_min_matches_requires_multiple_hits():
    cfg = dict(FILTER_CFG, min_matches=2)
    one_match = _paper(title="Formal Verification Only")
    two_matches = _paper(title="Formal Verification of Agent Behavior")

    kept = apply_keyword_filter([one_match, two_matches], cfg)

    assert one_match not in kept
    assert two_matches in kept

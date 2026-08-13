from datetime import date

from src.dedup import merge_duplicates
from src.models import Paper


def _paper(**overrides):
    defaults = dict(
        title="Formal Verification of LLM Agents",
        authors=["Alice Smith", "Bob Jones"],
        abstract="short",
        url="https://arxiv.org/abs/2401.00001",
        venue="cs.AI",
        source="arxiv",
        published_date=date(2026, 8, 4),
        arxiv_id="2401.00001",
    )
    defaults.update(overrides)
    return Paper(**defaults)


def test_merges_papers_sharing_arxiv_id_and_unions_extra():
    arxiv_paper = _paper(source="arxiv", abstract="a short abstract")
    hf_paper = _paper(
        source="hf_papers",
        abstract="a much longer and more complete abstract than the arxiv one",
        extra={"hf_upvotes": 42},
    )

    merged = merge_duplicates([arxiv_paper, hf_paper])

    assert len(merged) == 1
    result = merged[0]
    assert result.extra["hf_upvotes"] == 42
    assert "more complete" in result.abstract
    assert set(result.sources) == {"arxiv", "hf_papers"}


def test_keeps_distinct_papers_separate():
    p1 = _paper(arxiv_id="2401.00001", title="Paper One")
    p2 = _paper(arxiv_id="2401.00002", title="Paper Two")

    merged = merge_duplicates([p1, p2])

    assert len(merged) == 2


def test_falls_back_to_title_author_key_when_no_arxiv_id_or_doi():
    p1 = _paper(arxiv_id=None, doi=None, title="Some DBLP-Only Paper", authors=["Carol Lee"])
    p2 = _paper(arxiv_id=None, doi=None, title="Some DBLP-Only Paper", authors=["Carol Lee"], source="dblp")

    merged = merge_duplicates([p1, p2])

    assert len(merged) == 1
